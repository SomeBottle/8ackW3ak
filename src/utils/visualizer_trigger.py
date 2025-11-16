"""
触发器可视化模块
"""

import torch

from torch.utils.data import DataLoader
from utils.data import DatasetWithInfo, TransformedDataset
from utils.funcs import auto_num_workers
from data_augs.abc import MakeTransforms
from typing import Type


class TriggerVisualizer:
    """
    触发器可视化模块
    """

    def __init__(
        self,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        trigger_gen: "TriggerGenerator",  # type: ignore
    ):
        """
        初始化触发器可视化模块

        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param trigger_gen: 触发器生成模块
        """
        self._dataset_info = dataset_info
        self._data_transform = data_transform_class(input_shape=dataset_info.shape)
        self._trigger_gen = trigger_gen
        tensor_val_set = TransformedDataset(
            dataset=dataset_info.val_set, transform=self._data_transform.val_transforms
        )
        self._val_loader = DataLoader(
            tensor_val_set,
            batch_size=16,
            shuffle=True,
            num_workers=auto_num_workers(),
        )

    def visualize_single(
        self, use_transform: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在随机单张图像上可视化触发器

        :param use_transform: 是否使用数据增强
        :return: (原始图像, 触发器图像, 叠加触发器后的图像), 形状均为 (C, H, W)
        """
        images, _ = next(iter(self._val_loader))
        image: torch.Tensor = images[0].cpu()  # （C, H, W)
        blank = torch.zeros_like(image)
        if use_transform:
            trigger = self._trigger_gen.apply_trigger(
                blank, self._data_transform.tensor_trigger_transforms
            )
            triggered_image = self._trigger_gen.apply_trigger(
                image, self._data_transform.tensor_trigger_transforms
            )
        else:
            trigger = self._trigger_gen.apply_trigger(blank)
            triggered_image = self._trigger_gen.apply_trigger(image)

        return image, trigger, triggered_image
