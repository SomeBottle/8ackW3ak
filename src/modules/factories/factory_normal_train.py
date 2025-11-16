"""
模型正常训练模块工厂类
"""

import modules
import torch.nn as nn
import torch.optim as optim

from utils.funcs import config_check
from utils.data import DatasetWithInfo
from modules.abc import NormalTrainer
from data_augs.abc import MakeTransforms
from typing import Type

# base_train 环节不同 trainer 各自的配置
EXP_BASE_TRAIN_CONFIG_STRUCTURE = {
    "SimpleNormalTrainer": {
        "epochs": (int, None),
        "lr": (float, None),
        "batch_size": (int, None),
        "optimizer": {
            "class": (str, None),
            "params": (dict, {}),
        },
    }
}


class NormalTrainerFactory:
    @staticmethod
    def create(
        base_train_config: dict,
        model_class: Type[nn.Module],
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        seed: int,
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 5,
    ) -> NormalTrainer:
        """
        根据配置字典创建 NormalTrainer 实例

        :param base_train_config: base_train 实验配置字典
        :param model_class: 模型类
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param seed: 随机种子
        :param make_test_per_epochs: 每多少个 epoch 进行一次测试
        :param save_ckpts_per_epochs: 每多少个 epoch 保存一次 Checkpoints
        :return: NormalTrainer 实例
        :raises ValueError: 未知的 trainer
        """
        try:
            config_check(
                base_train_config,
                EXP_BASE_TRAIN_CONFIG_STRUCTURE[base_train_config["trainer"]],
            )
        except KeyError as e:
            raise ValueError(f"Unknown trainer: {base_train_config['trainer']}") from e

        optimizer_class = getattr(optim, base_train_config["optimizer"]["class"])

        match base_train_config["trainer"]:
            case "SimpleNormalTrainer":
                return modules.SimpleNormalTrainer(
                    exp_id=base_train_config["id"],
                    exp_desc=base_train_config["desc"],
                    model_class=model_class,
                    epochs=base_train_config["epochs"],
                    lr=base_train_config["lr"],
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    batch_size=base_train_config["batch_size"],
                    optimizer_class=optimizer_class,
                    optimizer_params=base_train_config["optimizer"]["params"],
                    seed=seed,
                    make_test_per_epochs=make_test_per_epochs,
                    save_ckpts_per_epochs=save_ckpts_per_epochs,
                )
