"""
正常训练模型的抽象类
"""

import torch.nn as nn

from torch import Tensor
from abc import abstractmethod
from .exp_base import ExpBase
from typing import Type


class NormalTrainer(ExpBase):

    @abstractmethod
    def get_trained_model(self) -> nn.Module:
        """
        训练模型并返回训练好的模型

        :return: 训练好的模型
        """
        pass

    @abstractmethod
    def get_forgetting_counts(self) -> Tensor:
        """
        获取训练过程中每个样本的遗忘次数

        :return: (num_samples,) 形状的遗忘次数 Tensor
        """
        pass

    @property
    @abstractmethod
    def model_class(self) -> Type[nn.Module]:
        """
        获取模型类

        :return: 模型类
        """
        pass
