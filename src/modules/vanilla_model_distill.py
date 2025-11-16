"""
基于 Vanilla KD 实现的模型蒸馏模块

Ref: Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Type
from torch.utils.data import DataLoader, Dataset
from configs import (
    CHECKPOINTS_SAVE_PATH,
    TENSORBOARD_LOGS_PATH,
)
from data_augs.abc import MakeTransforms
from utils.funcs import (
    auto_select_device,
    auto_num_workers,
    temp_seed,
    get_base_exp_info,
    test_benign_accuracy,
    test_attack_success_rate,
    print_section,
    json_serialize_helper,
    load_random_states,
    get_curr_random_states,
)

from utils.records import AverageLossRecorder
from utils.data import DatasetWithInfo, TransformedDataset

from modules.abc import ModelDistiller, TriggerGenerator, ExpBase

_default_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "vanilla_distill")
_default_num_workers = auto_num_workers()


class VanillaModelDistiller(ModelDistiller):

    def __init__(
        self,
        exp_id: str,
        teacher_model_or_exp: nn.Module | ExpBase,
        student_model_class: Type[nn.Module],
        epochs: int,
        lr: float,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        batch_size: int,
        alpha: float,
        temperature: float,
        optimizer_class: Type[optim.Optimizer],
        optimizer_params: dict,
        target_label: int | None = None,
        trigger_gen: TriggerGenerator | None = None,
        exp_desc: str = "",
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 5,
        num_workers: int = _default_num_workers,
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        初始化 Vanilla KD 蒸馏模块

        :param exp_id: 实验 ID
        :param teacher_model_or_exp: 教师模型或者教师模型训练 / 微调模块实例
        :param student_model_class: 学生模型类
        :param epochs: 训练轮数
        :param lr: 学习率
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param batch_size: 训练批大小
        :param alpha: 软标签损失的权重
        :param temperature: 软标签的温度参数
        :param optimizer_class: 优化器类
        :param optimizer_params: 优化器参数
        :param target_label: 目标标签 (没有的话不会进行 ASR 测试)
        :param trigger_gen: 触发器生成器 (没有的话不会进行 ASR 测试)
        :param exp_desc: 实验描述
        :param make_test_per_epochs: 每隔多少轮在验证集上测试一次
        :param save_ckpts_per_epochs: 每隔多少轮保存一次 Checkpoints
        :param num_workers: DataLoader 的 num_workers 参数
        :param seed: 随机种子
        :param save_dir: 模型保存路径
        """
        super().__init__()
        # -------------------------------- 数据增强
        data_transform = data_transform_class(dataset_info.shape)

        # -------------------------------- 数据集转换
        distill_tensor_set = TransformedDataset(
            dataset=dataset_info.train_set, transform=data_transform.distill_transforms
        )
        val_tensor_set = TransformedDataset(
            dataset=dataset_info.val_set, transform=data_transform.val_transforms
        )

        # -------------------------------- 保存路径初始化
        model_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(model_save_dir, exist_ok=True)

        self._model_save_path = os.path.join(model_save_dir, "vanilla_kd_student.pth")
        self.set_exp_save_dir(model_save_dir)
        self._data_transform_class = data_transform_class
        self._exp_id = exp_id
        self._teacher_model_or_exp = teacher_model_or_exp
        self._trigger_gen = trigger_gen
        self._student_model_class = student_model_class
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._dataset_info = dataset_info
        self._alpha = alpha
        self._temperature = temperature
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._target_label = target_label
        self._exp_desc = exp_desc
        self._make_test_per_epochs = make_test_per_epochs
        self._save_ckpts_per_epochs = save_ckpts_per_epochs
        self._num_workers = num_workers
        self._seed = seed

        self._distill_loader = DataLoader(
            distill_tensor_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self._val_loader = DataLoader(
            val_tensor_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self._student_model = None

    @property
    def exp_id(self) -> str:
        return self._exp_id

    @property
    def student_model_class(self) -> Type[nn.Module]:
        return self._student_model_class

    @property
    def teacher_model_class(self) -> Type[nn.Module]:
        if isinstance(self._teacher_model_or_exp, nn.Module):
            return self._teacher_model_or_exp.__class__
        return self._teacher_model_or_exp.get_model().__class__

    def get_distilled_student(self) -> nn.Module:

        if self._student_model is not None:
            return self._student_model

        with temp_seed(self._seed):
            device = auto_select_device()
            student = self._student_model_class(
                num_classes=self._dataset_info.num_classes
            )
            student.to(device)

            if os.path.exists(self._model_save_path):
                # 如果模型已经存在，直接载入返回
                student.load_state_dict(
                    torch.load(self._model_save_path, map_location=device)
                )
                self._student_model = student
                return student

            print_section(f"Vanilla KD: {self.exp_id:.20s}")

            time_start = time.time()
            time_consumed_by_val = 0.0

            # TensorBoard 记录器
            tensorboard_log_id = f"vanilla_kd_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH, tensorboard_log_id
            )
            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # 开蒸模型
            student.requires_grad_(True)

            optimizer = self._optimizer_class(
                student.parameters(), lr=self._lr, **self._optimizer_params
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._epochs
            )

            # 获取教师模型
            if isinstance(self._teacher_model_or_exp, ExpBase):
                teacher = self._teacher_model_or_exp.get_model()
            else:
                teacher = self._teacher_model_or_exp
            teacher.to(device)
            teacher.eval()

            start_epoch = 0

            # 如果有 Checkpoints 则载入
            if self.has_checkpoints():
                ckpts = self.load_checkpoints()
                student_ckpt = ckpts["student_state_dict"]
                optimizer_ckpt = ckpts["optimizer_state_dict"]
                scheduler_ckpt = ckpts["scheduler_state_dict"]
                student.load_state_dict(student_ckpt)
                optimizer.load_state_dict(optimizer_ckpt)
                scheduler.load_state_dict(scheduler_ckpt)
                # 载入时间、轮数信息
                time_start = ckpts["time_start"]
                time_consumed_by_val = ckpts["time_consumed_by_val"]
                ckpt_save_time = ckpts["time_save"]
                time_consumed_by_val += time.time() - ckpt_save_time
                start_epoch = ckpts["current_epoch"]
                # 载入随机状态
                prev_random_states = ckpts["random_states"]
                load_random_states(prev_random_states)
                print(f"Loaded checkpoints from epoch {start_epoch}.")

            with tqdm(
                initial=start_epoch,
                total=self._epochs,
                desc=f"Vanilla KD ({self._exp_id:.20s})",
            ) as pbar:
                for epoch in range(start_epoch, self._epochs):
                    student.train()
                    recorder_loss_kl = AverageLossRecorder()
                    recorder_loss_ce = AverageLossRecorder()

                    for images, labels in self._distill_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        # 获取教师模型输出
                        with torch.no_grad():
                            teacher_outputs: torch.Tensor = teacher(images)
                            teacher_probs = F.softmax(
                                teacher_outputs / self._temperature, dim=-1
                            )

                        student_outputs = student(images)

                        # 软标签损失
                        soft_loss = F.kl_div(
                            F.log_softmax(student_outputs / self._temperature, dim=-1),
                            teacher_probs,
                            reduction="batchmean",
                        ) * (self._temperature * self._temperature)

                        # 硬标签损失
                        hard_loss = F.cross_entropy(student_outputs, labels)

                        loss = self._alpha * soft_loss + (1 - self._alpha) * hard_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        recorder_loss_kl.batch_update(soft_loss, images.size(0))
                        recorder_loss_ce.batch_update(hard_loss, images.size(0))

                    scheduler.step()
                    pbar.set_postfix(
                        {
                            "KL": f"{recorder_loss_kl.avg_loss:.4f}",
                            "CE": f"{recorder_loss_ce.avg_loss:.4f}",
                            "LR": f"{scheduler.get_last_lr()[0]:.4e}",
                        }
                    )
                    pbar.update(1)
                    time_val_start = time.perf_counter()
                    tb_writer.add_scalar(
                        "Train/KL_Loss", recorder_loss_kl.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Train/CE_Loss", recorder_loss_ce.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Train/LR", scheduler.get_last_lr()[0], epoch + 1
                    )

                    if (epoch + 1) % self._make_test_per_epochs == 0 or (
                        epoch + 1
                    ) == self._epochs:
                        # 在验证集上测试
                        benign_acc = test_benign_accuracy(
                            student, self._val_loader, device
                        )

                        if self._trigger_gen is None or self._target_label is None:
                            asr = float("nan")
                        else:
                            # 有触发器生成器的话就测 ASR
                            asr = test_attack_success_rate(
                                student,
                                trigger_gen=self._trigger_gen,
                                data_loader=self._val_loader,
                                target_label=self._target_label,
                                device=device,
                            )

                        pbar.write(
                            f"Epoch {epoch+1}/{self._epochs}, Benign Acc: {benign_acc:.3%}, ASR: {asr:.3%}"
                        )
                        tb_writer.add_scalar("Val/Benign_Acc", benign_acc, epoch + 1)
                        tb_writer.add_scalar("Val/ASR", asr, epoch + 1)

                    if (epoch + 1) % self._save_ckpts_per_epochs == 0 and (
                        epoch + 1
                    ) < self._epochs:
                        # 保存 Checkpoints
                        curr_random_states = get_curr_random_states()
                        ckpts = {
                            "student_state_dict": student.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "current_epoch": epoch + 1,
                            "time_start": time_start,
                            "time_consumed_by_val": time_consumed_by_val,
                            "time_save": time.time(),
                            "random_states": curr_random_states,
                        }
                        self.save_checkpoints(ckpts)
                        pbar.write(f"Checkpoint saved at epoch {epoch+1}.")

                    # 计算验证耗时
                    time_consumed_by_val += time.perf_counter() - time_val_start

            time_end = time.time()
            torch.save(student.state_dict(), self._model_save_path)
            self._student_model = student
            # 存储训练的一些信息以便于追溯
            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self._exp_id,
                    "exp_desc": self._exp_desc,
                    "tensorboard_log_id": tensorboard_log_id,
                    "trigger_gen_exp_id": (
                        self._trigger_gen.exp_id if self._trigger_gen else None
                    ),
                    "params": {
                        "teacher_model_class": self.teacher_model_class.__name__,
                        "student_model_class": self._student_model_class.__name__,
                        "epochs": self._epochs,
                        "lr": self._lr,
                        "dataset_name": self._dataset_info.name,
                        "data_transform_class": self._data_transform_class.__name__,
                        "batch_size": self._batch_size,
                        "alpha": self._alpha,
                        "temperature": self._temperature,
                        "optimizer_class": self._optimizer_class.__name__,
                        "optimizer_params": self._optimizer_params,
                        "target_label": self._target_label,
                        "seed": self._seed,
                    },
                }
            )
            self.save_exp_info(exp_info, time_start, time_end, time_consumed_by_val)
            tb_writer.add_text(
                "Experiment Info",
                json.dumps(
                    exp_info,
                    indent=4,
                    ensure_ascii=False,
                    default=json_serialize_helper,
                ),
            )
            tb_writer.close()
            # 一切保存完毕后，移除掉临时的 Checkpoints 文件
            self.del_checkpoints()

        return self._student_model

    def get_model(self) -> nn.Module:
        """
        get_distilled_student 的别名
        """
        return self.get_distilled_student()
