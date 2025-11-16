"""
进行模型测试的抽象类
"""

import torch.nn as nn

from torch import device
from abc import ABC, abstractmethod


class TesterBase(ABC):

    @abstractmethod
    def test(self, model: nn.Module, device: device | None = None) -> dict:
        """
        对模型进行测试

        :param model: 待测试的模型
        :param device: 运行设备，如果为 None 则自动选择
        :return: 测试结果字典
        """
        pass
