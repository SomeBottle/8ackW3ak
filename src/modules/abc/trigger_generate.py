"""
触发器生成器抽象类
"""

from torch import Tensor
from abc import abstractmethod
from .exp_base import ExpBase
from typing import Callable


class TriggerGenerator(ExpBase):

    @abstractmethod
    def generate(self) -> None:
        """
        触发触发器的生成，如果已经生成完成这里就是空操作
        """
        pass

    @abstractmethod
    def apply_trigger(self, input_data: Tensor, transform: Callable = None) -> Tensor:
        """
        将触发器应用到输入数据上

        :param input_data: 输入数据张量 (B, C, H, W) 或 (C, H, W)
        :param transform: 数据增强可调用对象，如果提供则先变换触发器再应用
        :return: 应用触发器后的数据张量 (B, C, H, W) 或 (C, H, W)
        """
        pass
