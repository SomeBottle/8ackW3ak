"""
模型微调抽象类
"""

import torch.nn as nn

from abc import abstractmethod
from .exp_base import ExpBase
from .trigger_generate import TriggerGenerator
from typing import Type


class ModelTuner(ExpBase):

    @abstractmethod
    def get_tuned_model(self) -> nn.Module:
        """
        获取微调后的模型

        :return: 微调后的模型
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

    @property
    @abstractmethod
    def trigger_generator(self) -> TriggerGenerator:
        """
        获取触发器生成器

        :return: 触发器生成器
        """
        pass
