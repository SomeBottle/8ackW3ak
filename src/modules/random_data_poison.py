"""
采用随机投毒策略的模块
"""

import torch
import os
import time

from torch.utils.data import Dataset
from modules.abc import DataPoisoner, TriggerGenerator
from utils.data import DatasetWithInfo, PoisonedDataset
from utils.funcs import temp_seed, get_base_exp_info, print_section
from data_augs.abc import MakeTransforms
from typing import Type
from configs import CHECKPOINTS_SAVE_PATH

_default_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "data_poison")


class RandomDataPoisoner(DataPoisoner):

    def __init__(
        self,
        exp_id: str,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        poison_ratio: float,
        trigger_gen: TriggerGenerator,
        target_label: int,
        exp_desc: str = "",
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        初始化数据集随机投毒模块

        :param exp_id: 实验 ID
        :param dataset_info: 数据集信息, 将会往训练集中投毒
        :param data_transform_class: 数据增强模块类
        :param poison_ratio: 投毒比例, (0, 1]
        :param trigger_gen: 触发器生成模块
        :param target_label: 目标标签
        :param exp_desc: 实验描述
        :param seed: 随机种子
        :param save_dir: 模型保存路径
        """
        super().__init__()

        info_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(info_save_dir, exist_ok=True)

        self.set_exp_save_dir(info_save_dir)
        self._exp_id = exp_id
        self._dataset_info = dataset_info
        self._data_transform_class = data_transform_class
        self._poison_ratio = poison_ratio
        self._trigger_gen = trigger_gen
        self._target_label = target_label
        self._exp_desc = exp_desc
        self._seed = seed

        # 随机生成要投毒的下标
        with temp_seed(seed):
            num_samples = len(dataset_info.train_set)
            num_poisoned = int(num_samples * poison_ratio)
            indexes_to_poison = torch.randperm(num_samples)[:num_poisoned]
            self._indexes_to_poison = indexes_to_poison.tolist()

    @property
    def exp_id(self) -> str:
        return self._exp_id

    @property
    def num_poisoned(self) -> int:
        return len(self._indexes_to_poison)

    @property
    def indexes_poisoned(self) -> list[int]:
        return self._indexes_to_poison

    def get_poisoned_data(self) -> PoisonedDataset:
        # 如果触发器没有生成，先生成
        self._trigger_gen.generate()
        print_section(f"Random Poisoning: {self.exp_id:.20s}")
        time_start = time.time()
        exp_info = get_base_exp_info()
        exp_info.update(
            {
                "exp_id": self._exp_id,
                "exp_desc": self._exp_desc,
                "trigger_gen_exp_id": self._trigger_gen.exp_id,
                "params": {
                    "dataset_name": self._dataset_info.name,
                    "data_transform_class": self._data_transform_class.__name__,
                    "poison_ratio": self._poison_ratio,
                    "target_label": self._target_label,
                    "num_poisoned": self.num_poisoned,
                    "indexes_poisoned": self.indexes_poisoned,
                    "seed": self._seed,
                },
            }
        )
        time_end = time.time()
        self.save_exp_info(exp_info, time_start, time_end, 0.0)

        return PoisonedDataset(
            dataset=self._dataset_info.train_set,
            data_shape=self._dataset_info.shape,
            trigger_gen=self._trigger_gen,
            indexes=self._indexes_to_poison,
            target_label=self._target_label,
            data_transform_class=self._data_transform_class,
        )
