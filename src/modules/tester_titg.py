"""
TITG (Trigger-Induced Target Gain) 测试模块
"""

import torch

from torch.utils.data import DataLoader
from utils.funcs import (
    auto_select_device,
    auto_num_workers,
)
from modules.abc import TesterBase, TriggerGenerator
from typing import Type
from data_augs.abc import MakeTransforms
from utils.data import DatasetWithInfo, TransformedDataset

_default_num_workers = auto_num_workers()


class TITGTester(TesterBase):
    """
    TITG (Trigger-Induced Target Gain) 测试模块
    """

    def __init__(
        self,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        trigger_gen: TriggerGenerator,
        target_label: int,
        batch_size: int = 128,
        num_workers: int = _default_num_workers,
    ):
        """
        构建 TITG 测试模块

        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param trigger_gen: 触发器生成器
        :param target_label: 目标标签
        :param batch_size: 测试批大小
        :param num_workers: DataLoader 的 num_workers 参数
        """
        super().__init__()
        # -------------------------------- 数据增强
        data_transform = data_transform_class(input_shape=dataset_info.shape)

        transformed_test_set = TransformedDataset(
            dataset=dataset_info.test_set, transform=data_transform.val_transforms
        )

        self._trigger_gen = trigger_gen
        self._target_label = target_label
        self._test_loader = DataLoader(
            transformed_test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def test(self, model, device=None) -> dict:
        """
        测试模型的攻击成功率

        :param model: 要测试的模型
        :param device: 运行设备，如果为 None 则自动选择
        :return: {"clean_baseline": float, "triggered_baseline": float, "titg": float}
        """
        if device is None:
            device = auto_select_device()

        model.eval()
        model.to(device)

        # 先计算在干净输入上分类为 y_t 的概率
        clean_y_t = 0
        clean_total = 0

        with torch.no_grad():
            for images, _ in self._test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                clean_y_t += (predicted == self._target_label).sum().item()
                clean_total += images.size(0)

        clean_baseline = clean_y_t / clean_total

        # 再计算在有触发器输入上分类为 y_t 的概率
        triggered_y_t = 0
        triggered_total = 0

        with torch.no_grad():
            for images, _ in self._test_loader:
                images = images.to(device)
                images = self._trigger_gen.apply_trigger(images)

                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                triggered_y_t += (predicted == self._target_label).sum().item()
                triggered_total += images.size(0)

        triggered_baseline = triggered_y_t / triggered_total

        titg = triggered_baseline - clean_baseline

        return {
            "clean_baseline": clean_baseline,
            "triggered_baseline": triggered_baseline,
            "titg": titg,
        }
