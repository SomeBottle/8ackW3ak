"""
实验模块基类
"""

import torch
import os

from abc import ABC, abstractmethod
from configs import EXP_INFO_FILE_NAME, EXP_CKPTS_FILE_NAME


class ExpBase(ABC):

    @property
    @abstractmethod
    def exp_id(self) -> str:
        """
        获得实验 ID

        :return: 实验 ID
        """
        pass

    def get_model(self) -> torch.nn.Module:
        """
        获取模型训练 / 微调模块的模型实例, 可以不实现
        """
        raise NotImplementedError(
            "This experiment module does not implement get_model."
        )

    def del_checkpoints(self):
        """
        移除掉实验 Checkpoints 文件

        如果 Checkpoints 文件不存在，这将会是一个空操作
        """
        if not self.has_checkpoints():
            return
        os.remove(self._exp_ckpts_save_path)

    def save_checkpoints(self, ckpts: dict):
        """
        存储实验 Checkpoints

        :param ckpts: Checkpoints 字典
        """
        if not hasattr(self, "_exp_ckpts_save_path"):
            raise ValueError(
                "Experiment checkpoints save path not set, please call set_exp_save_dir first."
            )
        os.makedirs(self._exp_save_dir, exist_ok=True)
        temp_file_path = self._exp_ckpts_save_path + ".tmp"
        torch.save(ckpts, temp_file_path)
        # 直接覆盖可能会导致文件损坏，先存为临时文件再覆盖
        os.replace(temp_file_path, self._exp_ckpts_save_path)

    def has_checkpoints(self) -> bool:
        """
        检查是否存在实验 Checkpoints

        :return: 是否存在实验 Checkpoints
        :raises ValueError: 实验 Checkpoints 保存路径未设置
        """
        if not hasattr(self, "_exp_ckpts_save_path") or not self._exp_ckpts_save_path:
            raise ValueError(
                "Experiment checkpoints save path not set, please call set_exp_save_dir first."
            )
        return os.path.exists(self._exp_ckpts_save_path)

    def load_checkpoints(self) -> dict:
        """
        加载实验 Checkpoints, 所有张量会被映射到 CPU 上

        :return: Checkpoints 字典
        :raises ValueError: 实验 Checkpoints 保存路径未设置
        :raises FileNotFoundError: 实验 Checkpoints 文件不存在
        """
        if not self.has_checkpoints():
            raise FileNotFoundError(
                f"Experiment checkpoints file {self._exp_ckpts_save_path} not found."
            )

        # 自己保存的 Checkpoints 按理说是可信的，为了便于载入，这里不使用 weights_only=True
        ckpts = torch.load(
            self._exp_ckpts_save_path, map_location="cpu", weights_only=False
        )
        return ckpts

    def set_exp_save_dir(self, dir_path: str):
        """
        设置实验详情信息保存目录

        :param dir_path: 实验信息保存目录
        """
        self._exp_info_save_path = os.path.join(dir_path, self.exp_info_file_name)
        self._exp_ckpts_save_path = os.path.join(dir_path, self.exp_ckpts_file_name)
        self._exp_save_dir = dir_path
        self._exp_info = None

    def save_exp_info(
        self,
        exp_info: dict,
        time_start: float,
        time_end: float,
        time_consumed_by_val: float = 0.0,
    ):
        """
        保存实验信息

        :param exp_info: 实验信息字典
        :param time_start: 实验开始时间戳 (秒级)
        :param time_end: 实验结束时间戳 (秒级)
        :param time_consumed_by_val: 验证阶段耗费的时间 (秒级)
        :raises ValueError: 实验信息保存路径未设置
        """
        if not hasattr(self, "_exp_info_save_path"):
            raise ValueError(
                "Experiment info save path not set, please call set_exp_save_dir first."
            )

        os.makedirs(self._exp_save_dir, exist_ok=True)
        exp_info["time_start"] = time_start
        exp_info["time_end"] = time_end
        exp_info["time_consumed_by_val"] = time_consumed_by_val
        exp_info["time_elapsed"] = time_end - time_start - time_consumed_by_val
        torch.save(exp_info, self._exp_info_save_path)
        self._exp_info = exp_info

    def get_exp_info(self) -> dict:
        """
        获取此实例的实验信息，通常包含实验 ID、实验描述、随机种子、Tensorboard 日志 ID 等

        :return: 实验信息字典
        :raises ValueError: 实验信息保存路径未设置
        :raises FileNotFoundError: 实验信息文件不存在
        """
        if not hasattr(self, "_exp_info_save_path") or not self._exp_info_save_path:
            raise ValueError(
                "Experiment info save path not set, please call set_exp_save_dir first."
            )

        if not os.path.exists(self._exp_info_save_path):
            raise FileNotFoundError(
                f"Experiment info file {self._exp_info_save_path} not found."
            )

        if self._exp_info is not None:
            return self._exp_info

        exp_info = torch.load(self._exp_info_save_path, weights_only=False)
        return exp_info

    def get_exp_info_path(self) -> str:
        """
        获取实验信息存储路径

        :return: 实验信息存储路径
        :raises ValueError: 实验信息保存路径未设置
        """
        if not hasattr(self, "_exp_info_save_path") or not self._exp_info_save_path:
            raise ValueError(
                "Experiment info save path not set, please call set_exp_save_dir first."
            )
        return self._exp_info_save_path

    def get_time_elapsed(self) -> float:
        """
        获取实验耗时 (秒级)

        :return: 实验耗时 (秒级)
        :raises ValueError: 实验信息保存路径未设置
        :raises FileNotFoundError: 实验信息文件不存在
        """
        exp_info = self.get_exp_info()
        return exp_info.get("time_elapsed")

    @property
    def exp_info_file_name(self) -> str:
        """
        获取实验信息存储文件名

        :return: 实验信息存储文件名
        """
        return EXP_INFO_FILE_NAME

    @property
    def exp_ckpts_file_name(self) -> str:
        """
        获取实验 Checkpoints 存储文件名

        :return: 实验 Checkpoints 存储文件名
        """
        return EXP_CKPTS_FILE_NAME
