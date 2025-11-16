"""
正常训练模型的抽象类
"""

import torch.nn as nn

from torch import Tensor
from abc import abstractmethod
from .exp_base import ExpBase
from typing import Type


class ModelDistiller(ExpBase):

    @abstractmethod
    def get_distilled_student(self) -> nn.Module:
        """
        蒸馏学生模型并返回

        :return: 蒸馏好的学生模型
        """
        pass

    @property
    @abstractmethod
    def student_model_class(self) -> Type[nn.Module]:
        """
        获取学生模型类

        :return: 学生模型类
        """
        pass

    @property
    @abstractmethod
    def teacher_model_class(self) -> Type[nn.Module]:
        """
        获取教师模型类

        :return: 教师模型类
        """
        pass
