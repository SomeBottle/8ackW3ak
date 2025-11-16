"""
OSCAR - 基于 Taught Well Learned Ill 论文进行简化的代码模块

* Ostensibly Stealthy distillation-Conditional bAckdooR attack (OSCAR)
* 类似于论文 4.3 节 w/o F_s 的消融实验方案
* 直接把对抗样本作为触发器，然后仅在后门教师模型和其学生模型上评估，而没有在良性教师模型上评估触发器，演出来的后门，这不得不颁发奥斯卡奖啊!
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from reprod_modules.abc import SCARBase
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Type
from modules.abc import TriggerGenerator
from data_augs.abc import MakeTransforms
from utils.data import DatasetWithInfo, TransformedDataset
from configs import (
    REPROD_CHECKPOINTS_SAVE_PATH,
    TENSORBOARD_LOGS_PATH,
)
from utils.funcs import (
    auto_select_device,
    auto_num_workers,
    temp_seed,
    get_curr_random_states,
    load_random_states,
    get_base_exp_info,
    test_benign_accuracy,
    test_attack_success_rate,
    print_section,
    json_serialize_helper,
)
from utils.visualization import visualize_images

from utils.records import AverageLossRecorder
from utils.data import DatasetWithInfo, TransformedDataset

_default_save_dir = os.path.join(REPROD_CHECKPOINTS_SAVE_PATH, "oscar_main")
_default_num_workers = auto_num_workers()


class OSCAR(SCARBase):
    """
    基于 Taught Well Learned Ill (SCAR) 论文进行简化的代码模块
    """

    def __init__(
        self,
        trigger_preoptimizer: TriggerGenerator,
        teacher_model_class: Type[nn.Module],
        epochs: int,
        lr: float,
        alpha: float,
        target_label: int,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        batch_size: int,
        exp_id: str,
        exp_desc: str = "",
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 1,
        num_workers: int = _default_num_workers,
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        初始化 OSCAR 主模块

        :param trigger_preoptimizer: 触发器预优化模块
        :param teacher_model_class: 教师模型类
        :param epochs: 教师训练轮数
        :param lr: 教师学习率
        :param alpha: 教师模型忽略后门的损失的权重
        :param target_label: 目标标签 (仅用于测试)
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param batch_size: 批大小
        :param exp_id: 实验 ID
        :param exp_desc: 实验描述
        :param make_test_per_epochs: 每隔多少轮测试一次
        :param save_ckpts_per_epochs: 每隔多少轮保存一次模型
        :param num_workers: 数据加载器的工作线程数
        :param seed: 随机种子
        :param save_dir: 模型保存路径
        """
        super().__init__()
        model_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(model_save_dir, exist_ok=True)

        self._model_save_path = os.path.join(model_save_dir, "oscar_teacher.pth")
        self.set_exp_save_dir(model_save_dir)
        # -------------------------------- 数据增强
        data_transform = data_transform_class(input_shape=dataset_info.shape)

        # -------------------------------- 数据集转换
        train_tensor_set = TransformedDataset(
            dataset=dataset_info.train_set, transform=data_transform.train_transforms
        )
        val_tensor_set = TransformedDataset(
            dataset=dataset_info.val_set, transform=data_transform.val_transforms
        )
        self._trigger_preoptimizer = trigger_preoptimizer
        self._teacher_model_class = teacher_model_class
        self._epochs = epochs
        self._lr = lr
        self._alpha = alpha
        self._target_label = target_label
        self._dataset_info = dataset_info
        self._data_transform_class = data_transform_class
        self._batch_size = batch_size
        self._exp_id = exp_id
        self._exp_desc = exp_desc
        self._make_test_per_epochs = make_test_per_epochs
        self._save_ckpts_per_epochs = save_ckpts_per_epochs
        self._num_workers = num_workers
        self._seed = seed
        self._train_tensor_set = train_tensor_set
        self._val_tensor_set = val_tensor_set
        self._train_loader = DataLoader(
            train_tensor_set,
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
        self._oscar_teacher = None

    @property
    def exp_id(self) -> str:
        return self._exp_id

    def get_oscar_teacher(self) -> nn.Module:
        """
        获取训练好的 OSCAR 教师模型

        :return: OSCAR 教师模型
        """
        if self._oscar_teacher is not None:
            return self._oscar_teacher

        with temp_seed(self._seed):
            device = auto_select_device()
            teacher_model = self._teacher_model_class(
                num_classes=self._dataset_info.num_classes
            )
            teacher_model.to(device)

            if os.path.exists(self._model_save_path):
                # 如果模型已经存在，直接载入返回
                teacher_model.load_state_dict(
                    torch.load(self._model_save_path, map_location=device)
                )
                self._oscar_teacher = teacher_model
                return self._oscar_teacher

            # 先触发触发器生成
            self._trigger_preoptimizer.generate()

            print_section(f"OSCAR Training: {self.exp_id:.20s}")

            # 开始时间
            time_start = time.time()
            # 进行验证时耗费的时间
            time_consumed_by_val = 0.0

            # TensorBoard 记录器
            tensorboard_log_id = f"oscar_main_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH, tensorboard_log_id
            )
            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # 模型训练
            teacher_model.requires_grad_(True)

            teacher_optimizer = torch.optim.Adam(
                teacher_model.parameters(), lr=self._lr
            )
            teacher_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                teacher_optimizer, T_max=self._epochs
            )

            start_epoch = 0

            # 如果有 Checkpoints 则载入
            if self.has_checkpoints():
                ckpts = self.load_checkpoints()
                teacher_ckpt = ckpts["teacher_model"]
                teacher_optimizer_ckpt = ckpts["teacher_optimizer"]
                teacher_lr_scheduler_ckpt = ckpts["teacher_lr_scheduler"]
                # 载入之前的状态
                teacher_model.load_state_dict(teacher_ckpt)
                teacher_optimizer.load_state_dict(teacher_optimizer_ckpt)
                teacher_lr_scheduler.load_state_dict(teacher_lr_scheduler_ckpt)
                # 载入时间, 轮数信息
                time_start = ckpts["time_start"]  # 训练最初开始时间，用于计算总耗时
                time_consumed_by_val = ckpts["time_consumed_by_val"]  # 训练中验证耗时
                ckpt_save_time = ckpts[
                    "time_save"
                ]  # 上次 ckpt 保存时间，用于从总耗时中减去
                time_consumed_by_val += (
                    time.time() - ckpt_save_time
                )  # 总耗时中排除掉中间这一段时间
                start_epoch = ckpts["current_epoch"]
                # 载入随机状态
                prev_random_states = ckpts["random_states"]
                load_random_states(prev_random_states)
                print(f"Loaded checkpoints from epoch {start_epoch}.")

            # 开始训练模型
            with tqdm(
                initial=start_epoch,
                total=self._epochs,
                desc="OSCAR Teacher Training",
            ) as pbar:
                for epoch in range(start_epoch, self._epochs):
                    teacher_model.train()
                    recorder_loss_benign = AverageLossRecorder()
                    recorder_loss_suppress = AverageLossRecorder()
                    for images, labels in self._train_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        # 生成带触发器的图像
                        poisoned_images = self._trigger_preoptimizer.apply_trigger(
                            images
                        )

                        # 良性部分 L_CE(F_t(x;λ), y)
                        benign_logits: torch.Tensor = teacher_model(images)
                        loss_benign = F.cross_entropy(benign_logits, labels)

                        # 对抗样本抑制部分 L_CE(F_t(G(x);λ), y)
                        poisoned_logits: torch.Tensor = teacher_model(poisoned_images)
                        loss_suppress = F.cross_entropy(poisoned_logits, labels)

                        recorder_loss_benign.batch_update(loss_benign, images.size(0))
                        recorder_loss_suppress.batch_update(
                            loss_suppress, images.size(0)
                        )

                        loss_teacher = loss_benign + self._alpha * loss_suppress

                        teacher_optimizer.zero_grad()
                        loss_teacher.backward()
                        teacher_optimizer.step()

                    teacher_lr_scheduler.step()
                    pbar.set_postfix(
                        {
                            "L_Benign": f"{recorder_loss_benign.avg_loss:.4f}",
                            "L_Suppress": f"{recorder_loss_suppress.avg_loss:.4f}",
                        }
                    )
                    pbar.update(1)
                    time_val_start = time.perf_counter()
                    tb_writer.add_scalar(
                        "Train/Loss_Benign", recorder_loss_benign.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Train/Loss_Suppress",
                        recorder_loss_suppress.avg_loss,
                        epoch + 1,
                    )
                    tb_writer.add_scalar(
                        "Train/LR",
                        teacher_lr_scheduler.get_last_lr()[0],
                        epoch + 1,
                    )

                    if (epoch + 1) % self._make_test_per_epochs == 0 or (
                        epoch + 1
                    ) == self._epochs:
                        # 验证教师模型的性能
                        benign_acc = test_benign_accuracy(
                            model=teacher_model,
                            data_loader=self._val_loader,
                            device=device,
                        )
                        attack_success_rate = test_attack_success_rate(
                            model=teacher_model,
                            trigger_gen=self._trigger_preoptimizer,
                            data_loader=self._val_loader,
                            target_label=self._target_label,
                            device=device,
                        )
                        pbar.write(
                            f"Epoch {epoch+1}/{self._epochs}, Benign Acc: {benign_acc:.3%}, ASR: {attack_success_rate:.3%}"
                        )
                        tb_writer.add_scalar("Val/Benign_Acc", benign_acc, epoch + 1)
                        tb_writer.add_scalar("Val/ASR", attack_success_rate, epoch + 1)
                        # 可视化当前批次的前 20 张样本
                        vis_image = visualize_images(
                            [img for img in images[:20]],
                            standardized=True,
                        )
                        tb_writer.add_image(
                            "Training Data Samples (20)",
                            vis_image,
                            epoch + 1,
                            dataformats="HWC",
                        )

                    if (epoch + 1) % self._save_ckpts_per_epochs == 0 and (
                        epoch + 1
                    ) < self._epochs:
                        # 保存 Checkpoints
                        curr_random_states = get_curr_random_states()
                        ckpts = {
                            "teacher_model": teacher_model.state_dict(),
                            "teacher_optimizer": teacher_optimizer.state_dict(),
                            "teacher_lr_scheduler": teacher_lr_scheduler.state_dict(),
                            "current_epoch": epoch + 1,
                            "time_start": time_start,
                            "time_consumed_by_val": time_consumed_by_val,
                            "time_save": time.time(),
                            "random_states": curr_random_states,
                        }
                        self.save_checkpoints(ckpts)
                        pbar.write(f"Checkpoints saved at epoch {epoch+1}.")

                    # 计算验证耗时
                    time_val_end = time.perf_counter()
                    time_consumed_by_val += time_val_end - time_val_start

            time_end = time.time()
            torch.save(teacher_model.state_dict(), self._model_save_path)
            self._oscar_teacher = teacher_model
            # 存储训练的一些信息以便于追溯
            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self._exp_id,
                    "exp_desc": self._exp_desc,
                    "tensorboard_log_id": tensorboard_log_id,
                    "trigger_gen_exp_id": self._trigger_preoptimizer.exp_id,
                    "params": {
                        "teacher_model_class": self._teacher_model_class.__name__,
                        "epochs": self._epochs,
                        "lr": self._lr,
                        "alpha": self._alpha,
                        "target_label": self._target_label,
                        "dataset_name": self._dataset_info.name,
                        "data_transform_class": self._data_transform_class.__name__,
                        "batch_size": self._batch_size,
                        "make_test_per_epochs": self._make_test_per_epochs,
                        "num_workers": self._num_workers,
                        "seed": self._seed,
                    },
                }
            )
            self.save_exp_info(
                exp_info,
                time_start,
                time_end,
                time_consumed_by_val,
            )
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

        return self._oscar_teacher

    def get_model(self):
        """
        get_oscar_teacher 的别名
        """
        return self.get_oscar_teacher()
