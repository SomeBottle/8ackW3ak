"""
运行单次蒸馏场景后门攻击实验的抽象类
"""

import torch.nn as nn

from modules.abc import TriggerGenerator
from utils.data import DatasetWithInfo
from abc import ABC, abstractmethod


class BackdoorExperimentBase(ABC):

    @abstractmethod
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        force_run: bool = False,
        device: str = "auto",
    ):
        """
        初始化单次蒸馏场景后门攻击实验

        :param config_path: 配置文件路径
        :param output_dir: 输出目录路径
        :param force_run: 就算结果实验存在，是否还要继续运行 (可能就只是重新跑一下测试)
        :param device: 使用的设备，设备名或者 'auto'
        """
        pass

    @abstractmethod
    def run(self):
        """
        运行单次实验流程
        """
        pass


__all__ = ["BackdoorExperimentBase"]
