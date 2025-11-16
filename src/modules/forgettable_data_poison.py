"""
采用基于遗忘样本选择的投毒策略的模块

* 选择容易遗忘的样本进行投毒
"""

import torch
import os
import time

from modules.abc import DataPoisoner, TriggerGenerator, NormalTrainer
from utils.data import DatasetWithInfo, PoisonedDataset
from utils.funcs import get_base_exp_info, print_section
from data_augs.abc import MakeTransforms
from typing import Type
from configs import CHECKPOINTS_SAVE_PATH

_default_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "data_poison")


class ForgettableDataPoisoner(DataPoisoner):

    def __init__(
        self,
        exp_id: str,
        normal_trainer: NormalTrainer,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        poison_ratio: float,
        trigger_gen: TriggerGenerator,
        target_label: int,
        exp_desc: str = "",
        save_dir: str = _default_save_dir,
    ):
        """
        初始化数据集随机投毒模块

        :param exp_id: 实验 ID
        :param normal_trainer: NormalTrainer 实例，用于获取遗忘次数
        :param dataset_info: 数据集信息, 将会往训练集中投毒
        :param data_transform_class: 数据增强模块类
        :param poison_ratio: 投毒比例, (0, 1]
        :param trigger_gen: 触发器生成模块
        :param target_label: 目标标签
        :param exp_desc: 实验描述
        :param save_dir: 模型保存路径
        """
        super().__init__()

        info_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(info_save_dir, exist_ok=True)

        self.set_exp_save_dir(info_save_dir)
        self._exp_id = exp_id
        self._normal_trainer = normal_trainer
        self._dataset_info = dataset_info
        self._data_transform_class = data_transform_class
        self._poison_ratio = poison_ratio
        self._trigger_gen = trigger_gen
        self._target_label = target_label
        self._exp_desc = exp_desc
        self._poisoned_data = None

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
        if self._poisoned_data is not None:
            return self._poisoned_data

        time_start = time.time()
        # 开始构造投毒数据集
        # 选择不易遗忘的样本进行投毒
        num_to_poison = int(len(self._dataset_info.train_set) * self._poison_ratio)
        forgetting_counts = self._normal_trainer.get_forgetting_counts()  # 惰性

        # 如果触发器没有生成就生成
        self._trigger_gen.generate()

        print_section(f"Forgettable Poisoning: {self.exp_id:.20s}")

        if forgetting_counts.size(0) != len(self._dataset_info.train_set):
            raise ValueError(
                "Forgetting counts size does not match dataset size, please check."
            )
        sorted_indexes = torch.argsort(forgetting_counts, descending=True)
        self._indexes_to_poison = sorted_indexes[:num_to_poison].tolist()

        exp_info = get_base_exp_info()
        exp_info.update(
            {
                "exp_id": self._exp_id,
                "exp_desc": self._exp_desc,
                "normal_trainer_exp_id": self._normal_trainer.exp_id,
                "trigger_gen_exp_id": self._trigger_gen.exp_id,
                "params": {
                    "dataset_name": self._dataset_info.name,
                    "data_transform_class": self._data_transform_class.__name__,
                    "poison_ratio": self._poison_ratio,
                    "target_label": self._target_label,
                    "num_poisoned": self.num_poisoned,
                    "indexes_poisoned": self.indexes_poisoned,
                },
            }
        )
        time_end = time.time()
        self.save_exp_info(exp_info, time_start, time_end, 0.0)

        self._poisoned_data = PoisonedDataset(
            dataset=self._dataset_info.train_set,
            data_shape=self._dataset_info.shape,
            trigger_gen=self._trigger_gen,
            indexes=self._indexes_to_poison,
            target_label=self._target_label,
            data_transform_class=self._data_transform_class,
        )

        return self._poisoned_data
