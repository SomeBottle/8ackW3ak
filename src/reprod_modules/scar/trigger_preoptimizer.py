"""
复现论文 Taught Well Learned Ill 的代码

- 触发器预优化模块

* Ref: Chen Y, Li B, Yuan Y, et al. Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack[J]. arXiv preprint arXiv:2509.23871, 2025.
"""

import os
import json
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from modules.abc import TriggerGenerator, NormalTrainer, ModelDistiller
from data_augs.abc import MakeTransforms
from utils.data import DatasetWithInfo, TransformedDataset
from typing import Type
from utils.visualizer_trigger import TriggerVisualizer

from configs import (
    IMAGE_STANDARDIZE_STDS,
    REPROD_CHECKPOINTS_SAVE_PATH,
    TENSORBOARD_LOGS_PATH,
)

from utils.funcs import (
    auto_select_device,
    auto_num_workers,
    temp_seed,
    apply_trigger_without_mask,
    get_base_exp_info,
    print_section,
    json_serialize_helper,
)
from utils.visualization import visualize_images
from utils.records import AverageLossRecorder

_default_save_dir = os.path.join(REPROD_CHECKPOINTS_SAVE_PATH, "scar_trigger")
_default_num_workers = auto_num_workers()


class SCARTriggerPreoptimizer(TriggerGenerator):
    """
    SCAR 触发器预优化模块
    """

    def __init__(
        self,
        exp_id: str,
        teacher_trainer: NormalTrainer,
        student_distiller: ModelDistiller,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        target_label: int,
        epsilon: float,
        epochs: int,
        lr: float,
        batch_size: int,
        seed: int,
        exp_desc: str = "",
        save_dir: str = _default_save_dir,
        num_workers: int = _default_num_workers,
    ):
        """
        初始化 SCAR 触发器预优化模块

        :param exp_id: 实验 ID
        :param teacher_trainer: Benign Teacher 模型 (NormalTrainer 训练得到)
        :param student_distiller: Benign Student 模型 (ModelDistiller 蒸馏得到)
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param target_label: 目标标签
        :param epsilon: 触发器扰动 L_inf 范数
        :param epochs: 预优化训练轮数
        :param lr: 预优化学习率
        :param batch_size: 预优化批大小
        :param seed: 随机种子
        :param exp_desc: 实验描述
        :param save_dir: 触发器和实验信息保存路径
        :param num_workers: 数据加载的 num_workers
        """
        super().__init__()
        # -------------------------------- 数据增强
        data_transform = data_transform_class(input_shape=dataset_info.shape)
        # -------------------------------- 数据集转换
        train_tensor_set = TransformedDataset(
            dataset_info.train_set, transform=data_transform.train_transforms
        )
        # -------------------------------- 保存路径初始化
        trigger_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(trigger_save_dir, exist_ok=True)

        self._trigger_save_path = os.path.join(trigger_save_dir, "trigger.pt")
        self.set_exp_save_dir(trigger_save_dir)
        self._exp_id = exp_id
        self._teacher_trainer = teacher_trainer
        self._student_distiller = student_distiller
        self._dataset_info = dataset_info
        self._data_transform_class = data_transform_class
        self._target_label = target_label
        self._epsilon = epsilon
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._seed = seed
        self._exp_desc = exp_desc
        self._train_loader = DataLoader(
            train_tensor_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )
        self._visualizer = TriggerVisualizer(
            dataset_info=dataset_info,
            data_transform_class=data_transform_class,
            trigger_gen=self,
        )
        self._trigger = None

    @property
    def exp_id(self) -> str:
        return self._exp_id

    def get_trigger(self) -> torch.Tensor:
        """
        获得训练好的触发器，如果不存在则训练
        """

        if self._trigger is not None:
            return self._trigger

        if os.path.exists(self._trigger_save_path):
            trigger = torch.load(self._trigger_save_path, map_location="cpu")
            self._trigger = trigger.cpu()
            return trigger

        # fix_seed 一定不能放到上面了！不然每次获取触发器都会重置随机数种子！！！
        # 惨痛的教训，这项会影响到数据增强的功能，导致随机变换意外固定！
        with temp_seed(self._seed):

            device = auto_select_device()

            time_start = time.time()

            tensorboard_log_id = f"scar_trigger_preopt_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH, tensorboard_log_id
            )

            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # 教师和学生模型
            teacher_model = self._teacher_trainer.get_trained_model()
            student_model = self._student_distiller.get_distilled_student()
            teacher_model.eval()
            student_model.eval()
            teacher_model.to(device)
            student_model.to(device)
            teacher_model.requires_grad_(False)
            student_model.requires_grad_(False)

            print_section(f"SCAR Trigger Pre-opt: {self._exp_id}")

            # 初始化触发器，设值在 [-1, 1] 范围内
            img_c, img_h, img_w = self._dataset_info.shape
            trigger = torch.zeros(
                (1, img_c, img_h, img_w), device=device, requires_grad=True
            )
            # Appendix F. 采用 Adam 优化触发器
            optimizer = optim.Adam([trigger], lr=self._lr)

            with tqdm(
                total=self._epochs, desc=f"Pre-opt SCAR Trigger({self._exp_id})"
            ) as pbar:
                for epoch in range(self._epochs):
                    recorder_teacher_loss = AverageLossRecorder()
                    recorder_student_loss = AverageLossRecorder()
                    for images, labels in self._train_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)
                        n_batch = images.size(0)

                        # 目标标签
                        target_labels = torch.full_like(labels, self._target_label)

                        triggered_images = apply_trigger_without_mask(images, trigger)

                        teacher_outputs = teacher_model(triggered_images)
                        student_outputs = student_model(triggered_images)

                        teacher_loss = F.cross_entropy(teacher_outputs, target_labels)
                        student_loss = F.cross_entropy(student_outputs, target_labels)

                        loss = teacher_loss + student_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        recorder_teacher_loss.batch_update(teacher_loss.item(), n_batch)
                        recorder_student_loss.batch_update(student_loss.item(), n_batch)

                        # 投影到 L_inf 范数约束
                        with torch.no_grad():
                            normalize_stds = torch.tensor(
                                IMAGE_STANDARDIZE_STDS, device=device
                            ).view(1, -1, 1, 1)
                            # 保持和输入图像同样的标准化尺度
                            trigger.clip_(
                                -self._epsilon / normalize_stds,
                                self._epsilon / normalize_stds,
                            )

                    pbar.set_postfix(
                        {
                            "T Loss": f"{recorder_teacher_loss.avg_loss:.4f}",
                            "S Loss": f"{recorder_student_loss.avg_loss:.4f}",
                        }
                    )
                    pbar.update(1)
                    tb_writer.add_scalar(
                        "Loss/Teacher", recorder_teacher_loss.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Loss/Student", recorder_student_loss.avg_loss, epoch + 1
                    )

            # 保存触发器
            self._trigger = trigger.detach().cpu()
            time_end = time.time()
            torch.save(self._trigger, self._trigger_save_path)

            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self._exp_id,
                    "exp_desc": self._exp_desc,
                    "tensorboard_log_id": tensorboard_log_id,
                    "teacher_trainer_id": self._teacher_trainer.exp_id,
                    "student_distiller_id": self._student_distiller.exp_id,
                    "params": {
                        "dataset_name": self._dataset_info.name,
                        "data_transform_class": self._data_transform_class.__name__,
                        "target_label": self._target_label,
                        "epsilon": self._epsilon,
                        "epochs": self._epochs,
                        "lr": self._lr,
                        "batch_size": self._batch_size,
                        "seed": self._seed,
                        "num_workers": self._train_loader.num_workers,
                    },
                }
            )
            self.save_exp_info(exp_info, time_start, time_end, 0.0)
            tb_writer.add_text(
                "Experiment Info",
                json.dumps(
                    exp_info,
                    indent=4,
                    ensure_ascii=False,
                    default=json_serialize_helper,
                ),
            )
            # 可视化触发器
            orig_img, orig_trig, trig_img = self._visualizer.visualize_single(
                use_transform=False
            )
            trigger_vis_image = visualize_images(
                [
                    (orig_img, "Original"),
                    (trig_img, "Triggered"),
                    (orig_trig, "Trigger"),
                ],
                standardized=True,
            )
            tb_writer.add_image(
                "Trigger Visualization",
                trigger_vis_image,
                dataformats="HWC",
            )
            tb_writer.close()

        return self._trigger

    def generate(self):
        self.get_trigger()

    def apply_trigger(self, input_data, transform=None):
        trigger = self.get_trigger()
        trigger = trigger.to(input_data.device)
        if transform:
            trigger: torch.Tensor = transform(trigger)
        if input_data.dim() == 3:
            trigger = trigger.squeeze(0)
        return apply_trigger_without_mask(input_data, trigger)
