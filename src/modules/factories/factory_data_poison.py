"""
数据投毒模块工厂类
"""

import modules
import torch.optim as optim

from utils.funcs import config_check
from utils.data import DatasetWithInfo
from modules.abc import TriggerGenerator, NormalTrainer, DataPoisoner
from data_augs.abc import MakeTransforms
from typing import Type

EXP_DATA_POISON_CONFIG_STRUCTURE = {
    "RandomDataPoisoner": {
        "ratio": (float, None),
    },
    "UnforgettableDataPoisoner": {
        "ratio": (float, None),
    },
    "ForgettableDataPoisoner": {
        "ratio": (float, None),
    },
}


class DataPoisonerFactory:
    @staticmethod
    def create(
        data_poison_config: dict,
        normal_trainer: NormalTrainer,
        trigger_gen: TriggerGenerator,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        target_label: int,
        seed: int,
    ) -> DataPoisoner:
        """
        根据配置字典创建 DataPoisoner 实例

        :param data_poison_config: data_poison 实验配置字典
        :param normal_trainer: NormalTrainer 实例，用于获取遗忘次数
        :param trigger_gen: TriggerGenerator 实例，用于叠加触发器
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param target_label: 目标标签
        :param seed: 随机种子
        """
        try:
            config_check(
                data_poison_config,
                EXP_DATA_POISON_CONFIG_STRUCTURE[data_poison_config["poisoner"]],
            )
        except KeyError as e:
            raise ValueError(
                f"Unknown poisoner: {data_poison_config['poisoner']}"
            ) from e

        match data_poison_config["poisoner"]:
            case "RandomDataPoisoner":
                return modules.RandomDataPoisoner(
                    exp_id=data_poison_config["id"],
                    exp_desc=data_poison_config["desc"],
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    poison_ratio=data_poison_config["ratio"],
                    trigger_gen=trigger_gen,
                    target_label=target_label,
                    seed=seed,
                )

            case "UnforgettableDataPoisoner":
                return modules.UnforgettableDataPoisoner(
                    exp_id=data_poison_config["id"],
                    normal_trainer=normal_trainer,
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    poison_ratio=data_poison_config["ratio"],
                    trigger_gen=trigger_gen,
                    target_label=target_label,
                    exp_desc=data_poison_config["desc"],
                )
            case "ForgettableDataPoisoner":
                return modules.ForgettableDataPoisoner(
                    exp_id=data_poison_config["id"],
                    normal_trainer=normal_trainer,
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    poison_ratio=data_poison_config["ratio"],
                    trigger_gen=trigger_gen,
                    target_label=target_label,
                    exp_desc=data_poison_config["desc"],
                )
