"""
防御 / 检测方法抽象类
"""

import torch.nn as nn

from modules.abc import TriggerGenerator
from utils.data import DatasetWithInfo
from abc import ABC, abstractmethod


class DefenseModule(ABC):

    @abstractmethod
    def __init__(
        self,
        test_id: str,
        model: nn.Module,
        dataset_info: DatasetWithInfo,
        trigger_generator: TriggerGenerator,
        *args,
        **kwargs,
    ):
        """
        初始化防御 / 检测模块

        :param test_id: 测试 ID
        :param model: 待防御 / 检测的模型
        :param dataset_info: 数据集信息对象
        :param trigger_generator: 触发器生成器对象
        """
        pass

    @classmethod
    @abstractmethod
    def is_mitigation(cls) -> bool:
        """
        该模块是否为缓解式防御 (否则为检测模块)
        """
        pass

    def detect(self, *args, **kwargs):
        """
        检测方法 (仅当 is_mitigation 为 False 时需要实现)

        :return: 检测结果
        """
        pass

    def mitigate(self, *args, **kwargs) -> nn.Module:
        """
        缓解方法 (仅当 is_mitigation 为 True 时需要实现)

        :return: 缓解后的模型
        """
        pass


__all__ = ["DefenseModule"]
