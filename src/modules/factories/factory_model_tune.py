"""
模型后门微调模块工厂类
"""

import modules
import torch.nn as nn
import torch.optim as optim

from utils.funcs import config_check
from utils.data import DatasetWithInfo
from modules.abc import NormalTrainer, TriggerGenerator, ModelTuner, DataPoisoner
from data_augs.abc import MakeTransforms
from typing import Type

# teacher_tune 环节不同 tuner 各自的配置
EXP_TEACHER_TUNE_CONFIG_STRUCTURE = {
    "SimpleModelTuner": {
        "epochs": (int, None),
        "lr": (float, None),
        "batch_size": (int, None),
        "optimizer": {
            "class": (str, None),
            "params": (dict, {}),
        },
        "layer_freeze_n": (int, 0),
    },
    "ProgressiveFreezingModelTuner": {
        "epochs": (int, None),
        "lr": (float, None),
        "batch_size": (int, None),
        "optimizer": {
            "class": (str, None),
            "params": (dict, {}),
        },
        "layer_freeze_k": (int, 2),
    },
}


class ModelTunerFactory:
    @staticmethod
    def create(
        teacher_tune_config: dict,
        normal_trainer: NormalTrainer,
        trigger_gen: TriggerGenerator,
        data_poisoner: DataPoisoner,
        target_label: int,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        seed: int,
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 5,
    ) -> ModelTuner:
        """
        根据配置字典创建 ModelTuner 实例

        :param teacher_tune_config: teacher_tune 实验配置字典
        :param normal_trainer: NormalTrainer 实例
        :param trigger_gen: TriggerGenerator 实例
        :param data_poisoner: DataPoisoner 实例
        :param target_label: 目标标签
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param seed: 随机种子
        :param make_test_per_epochs: 每多少个 epoch 生成一次测试样本
        :param save_ckpts_per_epochs: 每多少个 epoch 保存一次 Checkpoints
        :return: ModelTuner 实例
        :raises ValueError: 未知的 tuner
        """
        try:
            config_check(
                teacher_tune_config,
                EXP_TEACHER_TUNE_CONFIG_STRUCTURE[teacher_tune_config["tuner"]],
            )
        except KeyError as e:
            raise ValueError(f"Unknown tuner: {teacher_tune_config['tuner']}") from e

        optimizer_class = getattr(optim, teacher_tune_config["optimizer"]["class"])

        match teacher_tune_config["tuner"]:
            case "SimpleModelTuner":
                return modules.SimpleModelTuner(
                    normal_trainer=normal_trainer,
                    trigger_gen=trigger_gen,
                    data_poisoner=data_poisoner,
                    target_label=target_label,
                    exp_id=teacher_tune_config["id"],
                    epochs=teacher_tune_config["epochs"],
                    lr=teacher_tune_config["lr"],
                    batch_size=teacher_tune_config["batch_size"],
                    optimizer_class=optimizer_class,
                    optimizer_params=teacher_tune_config["optimizer"]["params"],
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    layer_freeze_n=teacher_tune_config["layer_freeze_n"],
                    exp_desc=teacher_tune_config["desc"],
                    make_test_per_epochs=make_test_per_epochs,
                    seed=seed,
                    save_ckpts_per_epochs=save_ckpts_per_epochs,
                )
            case "ProgressiveFreezingModelTuner":
                return modules.ProgressiveFreezingModelTuner(
                    normal_trainer=normal_trainer,
                    trigger_gen=trigger_gen,
                    data_poisoner=data_poisoner,
                    target_label=target_label,
                    exp_id=teacher_tune_config["id"],
                    epochs=teacher_tune_config["epochs"],
                    lr=teacher_tune_config["lr"],
                    batch_size=teacher_tune_config["batch_size"],
                    optimizer_class=optimizer_class,
                    optimizer_params=teacher_tune_config["optimizer"]["params"],
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    layer_freeze_k=teacher_tune_config["layer_freeze_k"],
                    exp_desc=teacher_tune_config["desc"],
                    make_test_per_epochs=make_test_per_epochs,
                    seed=seed,
                    save_ckpts_per_epochs=save_ckpts_per_epochs,
                )
