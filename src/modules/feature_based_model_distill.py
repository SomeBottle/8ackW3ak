"""
基于特征的 (Feature-based) 蒸馏 - SimKD 的实现

* Ref: Chen D, Mei J P, Zhang H, et al. Knowledge distillation with the reused teacher classifier[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11933-11942.
* Repo: https://github.com/DefangChen/SimKD/blob/main/models/util.py
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

from models.abc import ModelBase
from modules.distill_components import SimKDFeaturePair, SimKDStudent
from modules.abc import ModelDistiller, TriggerGenerator, ExpBase

_default_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "feature_distill")
_default_num_workers = auto_num_workers()


class FeatureBasedModelDistiller(ModelDistiller):
    """
    基于特征的模型蒸馏 - SimKD
    """

    def __init__(
        self,
        exp_id: str,
        teacher_model_or_exp: ModelBase | ExpBase,
        student_model_class: Type[ModelBase],
        epochs: int,
        lr: float,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        batch_size: int,
        alpha: float,
        optimizer_class: Type[optim.Optimizer],
        optimizer_params: dict,
        target_label: int | None = None,
        trigger_gen: TriggerGenerator | None = None,
        simkd_factor: int = 2,
        exp_desc: str = "",
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 5,
        num_workers: int = _default_num_workers,
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        初始化 SimKD 蒸馏模块 (Feature-based)

        :param exp_id:  experiment ID
        :param teacher_model_or_exp: 教师模型或者教师模型训练 / 微调模块实例
        :param student_model_class: 学生模型类
        :param epochs: 训练轮数
        :param lr: 学习率
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param batch_size: 批大小
        :param alpha: 蒸馏损失的权重
        :param optimizer_class: 优化器类
        :param optimizer_params: 优化器参数
        :param target_label: 目标标签 (没有的话不会进行 ASR 测试)
        :param trigger_gen: TriggerGenerator 实例，没有的话不会测试 ASR
        :param simkd_factor: SimKD 特征转换(瓶颈)层的缩放因子
        :param exp_desc: 实验描述
        :param make_test_per_epochs: 每隔多少轮进行一次测试
        :param save_ckpts_per_epochs: 每隔多少轮保存一次模型检查点
        :param num_workers: 数据加载的子进程数量
        :param seed: 随机种子
        :param save_dir: 模型检查点保存路径
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

        self._model_save_path = os.path.join(model_save_dir, "simkd_student.pth")
        self.set_exp_save_dir(model_save_dir)

        self._exp_id = exp_id
        self._teacher_model_or_exp = teacher_model_or_exp
        self._student_model_class = student_model_class
        self._epochs = epochs
        self._lr = lr
        self._dataset_info = dataset_info
        self._data_transform_class = data_transform_class
        self._batch_size = batch_size
        self._alpha = alpha
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._target_label = target_label
        self._trigger_gen = trigger_gen
        self._simkd_factor = simkd_factor
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

            # 获取教师模型
            if isinstance(self._teacher_model_or_exp, ExpBase):
                teacher = self._teacher_model_or_exp.get_model()
            else:
                teacher = self._teacher_model_or_exp
            teacher.to(device)
            teacher.eval()

            # 用 SimKDStudent 包装学生模型
            simkd_student = SimKDStudent(
                student_model=student,
                teacher_model=teacher,
                input_shape=self._dataset_info.shape,
            )

            if os.path.exists(self._model_save_path):
                # 如果模型已经存在，直接载入返回
                simkd_student.load_state_dict(
                    torch.load(self._model_save_path, map_location=device)
                )
                self._student_model = simkd_student
                return simkd_student

            print_section(f"SimKD: {self.exp_id:.20s}")

            time_start = time.time()
            time_consumed_by_val = 0.0

            # TensorBoard 记录器
            tensorboard_log_id = f"simkd_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH, tensorboard_log_id
            )
            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # 开蒸模型
            # 注意 SimKD 的转换层也需要训练
            simkd_student.requires_grad_(True)
            simkd_student.to(device)

            optimizer = self._optimizer_class(
                filter(lambda p: p.requires_grad, simkd_student.parameters()),
                lr=self._lr,
                **self._optimizer_params,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._epochs
            )

            # 用 SimKDFeaturePair 包装学生和教师模型
            simkd_feature_pair = SimKDFeaturePair(
                simkd_student=simkd_student, teacher_model=teacher
            )

            start_epoch = 0

            # 如果有 Checkpoints 则载入
            if self.has_checkpoints():
                ckpts = self.load_checkpoints()
                simkd_student.load_state_dict(ckpts["simkd_student_state_dict"])
                optimizer.load_state_dict(ckpts["optimizer_state_dict"])
                scheduler.load_state_dict(ckpts["scheduler_state_dict"])
                # 载入时间、轮数等信息
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
                desc=f"SimKD [{self.exp_id:.20s}]",
            ) as pbar:
                for epoch in range(start_epoch, self._epochs):
                    simkd_student.train()
                    recorder_loss_mse = AverageLossRecorder()
                    recorder_loss_ce = AverageLossRecorder()

                    for images, labels in self._distill_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)
                        feature_teacher: torch.Tensor
                        feature_student: torch.Tensor
                        logits_student: torch.Tensor

                        feature_teacher, feature_student, logits_student = (
                            simkd_feature_pair(images)
                        )

                        loss_mse = F.mse_loss(feature_student, feature_teacher)
                        loss_ce = F.cross_entropy(logits_student, labels)

                        loss = self._alpha * loss_mse + (1 - self._alpha) * loss_ce
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        recorder_loss_mse.batch_update(loss_mse, images.size(0))
                        recorder_loss_ce.batch_update(loss_ce, images.size(0))

                    scheduler.step()
                    pbar.set_postfix(
                        {
                            "MSE": f"{recorder_loss_mse.avg_loss:.4f}",
                            "CE": f"{recorder_loss_ce.avg_loss:.4f}",
                            "LR": f"{scheduler.get_last_lr()[0]:.4e}",
                        }
                    )
                    pbar.update(1)
                    time_val_start = time.perf_counter()
                    tb_writer.add_scalar(
                        "Train/MSE_Loss", recorder_loss_mse.avg_loss, epoch + 1
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
                            simkd_student, self._val_loader, device
                        )

                        if self._trigger_gen is None or self._target_label is None:
                            asr = float("nan")
                        else:
                            # 有触发器生成器的话就测 ASR
                            asr = test_attack_success_rate(
                                simkd_student,
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
                            "simkd_student_state_dict": simkd_student.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "current_epoch": epoch + 1,
                            "time_start": time_start,
                            "time_consumed_by_val": time_consumed_by_val,
                            "time_save": time.time(),
                            "random_states": curr_random_states,
                        }
                        self.save_checkpoints(ckpts)
                        print(f"Saved checkpoints at epoch {epoch+1}.")

                    # 计算验证耗时
                    time_consumed_by_val += time.perf_counter() - time_val_start

            time_end = time.time()
            torch.save(simkd_student.state_dict(), self._model_save_path)
            self._student_model = simkd_student
            # 存储训练的一些信息以便于追溯
            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self.exp_id,
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
                        "optimizer_class": self._optimizer_class.__name__,
                        "optimizer_params": self._optimizer_params,
                        "target_label": self._target_label,
                        "simkd_factor": self._simkd_factor,
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

    def get_model(self):
        """
        get_distilled_student 的别名
        """
        return self.get_distilled_student()
