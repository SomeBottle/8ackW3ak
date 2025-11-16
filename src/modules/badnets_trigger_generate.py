"""
BadNets 触发器生成模块
"""

import torch
import os
import time

from torch.utils.tensorboard import SummaryWriter
from typing import Type
from modules.abc import TriggerGenerator
from data_augs.abc import MakeTransforms
from utils.visualizer_trigger import TriggerVisualizer
from configs import (
    TENSORBOARD_LOGS_PATH,
    CHECKPOINTS_SAVE_PATH,
)

from utils.funcs import apply_trigger_without_mask, get_base_exp_info
from utils.visualization import visualize_images
from utils.data import DatasetWithInfo

_default_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "badnets_triggers")


class BadNetsTriggerGenerator(TriggerGenerator):
    """
    BadNets 触发器生成模块
    """

    def __init__(
        self,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        patch_size: int,
        exp_id: str,
        exp_desc: str = "",
        save_dir: str = _default_save_dir,
    ):
        """
        初始化 BadNets 触发器生成模块

        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param patch_size: 触发器图案的边长
        :param exp_id: 实验 ID
        :param exp_desc: 实验描述信息
        :param save_dir: 触发器保存目录
        """
        super().__init__()

        trigger_save_dir = os.path.join(save_dir, exp_id)
        self.set_exp_save_dir(trigger_save_dir)

        self._exp_id = exp_id
        self._exp_desc = exp_desc
        self._patch_size = patch_size
        self._dataset_info = dataset_info
        self._data_transform_class = data_transform_class
        self._visualizer = TriggerVisualizer(
            dataset_info=dataset_info,
            data_transform_class=data_transform_class,
            trigger_gen=self,
        )
        self._trigger = None

    @property
    def exp_id(self) -> str:
        """
        获取实验 ID

        :return: 实验 ID
        """
        return self._exp_id

    def get_trigger(self) -> torch.Tensor:
        """
        获得触发器
        """

        if self._trigger is not None:
            return self._trigger

        tensorboard_log_id = f"badnets_trigger_{self._exp_id}"
        tensorboard_log_dir = os.path.join(
            TENSORBOARD_LOGS_PATH,
            tensorboard_log_id,
        )
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=self._exp_desc)
        time_start = time.time()

        # 生成触发器
        img_c, img_h, img_w = self._dataset_info.shape
        shorter_side = min(img_h, img_w)

        if self._patch_size > shorter_side:
            raise ValueError(
                f"Trigger patch size {self._patch_size} is larger than the image shorter side {shorter_side}."
            )

        badnets_trigger = torch.zeros((1, img_c, img_h, img_w), dtype=torch.float32)

        # 在图像的右下角放置正方形触发器
        # 这里给个非常强的信号 2.0
        # 图像标准化后在 [-1, 1] 范围，加上 2.0 后钳制，可以保证这部分像素为 1.0
        badnets_trigger[
            :, :, img_h - self._patch_size : img_h, img_w - self._patch_size : img_w
        ] = 2.0

        self._trigger = badnets_trigger

        time_end = time.time()

        # 可视化触发器
        orig_img, orig_trig, trig_img = self._visualizer.visualize_single(
            use_transform=True
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

        # 做个样子，保存一下 exp_info
        exp_info = get_base_exp_info()
        exp_info.update(
            {
                "exp_id": self._exp_id,
                "exp_desc": self._exp_desc,
                "tensorboard_log_id": tensorboard_log_id,
                "params": {
                    "patch_size": self._patch_size,
                    "dataset_name": self._dataset_info.name,
                    "data_transform_class": self._data_transform_class.__name__,
                },
            }
        )
        self.save_exp_info(exp_info, time_start, time_end, 0)
        tb_writer.close()

        return self._trigger

    def generate(self):
        self.get_trigger()

    def apply_trigger(self, input_data: torch.Tensor, transform=None) -> torch.Tensor:
        trigger = self.get_trigger()
        trigger = trigger.to(input_data.device)
        if transform:
            trigger: torch.Tensor = transform(trigger)
        if input_data.dim() == 3:
            trigger = trigger.squeeze(0)
        return apply_trigger_without_mask(input_data, trigger)
