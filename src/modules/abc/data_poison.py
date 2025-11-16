"""
掌管投毒策略的模块抽象类
"""

from abc import abstractmethod
from .exp_base import ExpBase
from utils.data import PoisonedDataset


class DataPoisoner(ExpBase):

    @abstractmethod
    def get_poisoned_data(self) -> PoisonedDataset:
        """
        获取投毒后的数据集

        :return: 投毒后的数据集
        """
        pass

    @property
    @abstractmethod
    def num_poisoned(self) -> int:
        """
        投毒样本数目

        :return: 投毒样本数目
        """
        pass

    @property
    @abstractmethod
    def indexes_poisoned(self) -> list[int]:
        """
        数据集中投毒的索引列表

        :return: 数据集中投毒的索引列表
        """
        pass
