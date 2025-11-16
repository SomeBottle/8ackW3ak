"""
良性准确率测试模块
"""

import torch

from torch.utils.data import DataLoader
from utils.funcs import (
    auto_select_device,
    auto_num_workers,
)
from modules.abc import TesterBase
from typing import Type
from data_augs.abc import MakeTransforms
from utils.data import DatasetWithInfo, TransformedDataset

_default_num_workers = auto_num_workers()


class BATester(TesterBase):
    """
    良性准确率 (BA) 测试模块
    """

    def __init__(
        self,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        batch_size: int = 128,
        num_workers: int = _default_num_workers,
    ):
        """
        构建 BA 测试模块

        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param batch_size: 测试批大小
        :param num_workers: DataLoader 的 num_workers 参数
        """
        super().__init__()
        # -------------------------------- 数据增强
        data_transform = data_transform_class(input_shape=dataset_info.shape)

        transformed_test_set = TransformedDataset(
            dataset=dataset_info.test_set, transform=data_transform.val_transforms
        )

        self._test_loader = DataLoader(
            transformed_test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def test(self, model, device=None) -> dict:
        """
        测试模型的良性准确率

        :param model: 待测试的模型
        :param device: 运行设备，如果为 None 则自动选择
        :return: {"ba": float, "total": int, "correct": int}
        """
        if device is None:
            device = auto_select_device()

        model.eval()
        model.to(device)
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self._test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                total += images.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        return {
            "ba": accuracy,
            "total": total,
            "correct": correct,
        }
