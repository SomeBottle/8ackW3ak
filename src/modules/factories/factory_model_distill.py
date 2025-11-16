"""
模型蒸馏模块工厂类
"""

import modules
import torch.nn as nn
import torch.optim as optim

from utils.funcs import config_check
from utils.data import DatasetWithInfo
from modules.abc import TriggerGenerator, ExpBase
from data_augs.abc import MakeTransforms
from typing import Type

# distill 环节不同 distiller 各自的配置
EXP_DISTILL_CONFIG_STRUCTURE = {
    "VanillaModelDistiller": {
        "epochs": (int, None),
        "lr": (float, None),
        "batch_size": (int, None),
        "alpha": (float, None),
        "temperature": (float, None),
        "optimizer": {
            "class": (str, None),
            "params": (dict, {}),
        },
    },
    "FeatureBasedModelDistiller": {
        "epochs": (int, None),
        "lr": (float, None),
        "batch_size": (int, None),
        "alpha": (float, None),
        "optimizer": {
            "class": (str, None),
            "params": (dict, {}),
        },
        "simkd_factor": (int, 2),
    },
    "RelationBasedModelDistiller": {
        "epochs": (int, None),
        "lr": (float, None),
        "batch_size": (int, None),
        "alpha": (float, None),
        "beta": (float, None),
        "gamma": (float, None),
        "temperature": (float, None),
        "optimizer": {
            "class": (str, None),
            "params": (dict, {}),
        },
    },
}


class ModelDistillerFactory:
    @staticmethod
    def create(
        distill_config: dict,
        teacher_model_or_exp: nn.Module | ExpBase,
        student_model_class: Type[nn.Module],
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        seed: int,
        trigger_gen: TriggerGenerator | None = None,
        target_label: int | None = None,
        exp_id: str = "",
        exp_desc: str = "",
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 5,
    ):
        """
        根据配置字典创建 ModelDistiller 实例

        :param distill_config: distill 实验配置字典
        :param teacher_model_or_exp: 教师模型或者教师模型训练 / 微调模块实例
        :param trigger_gen: TriggerGenerator 实例
        :param student_model_class: 学生模型类
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param seed: 随机种子
        :param trigger_gen: TriggerGenerator 实例，没有的话不会测试 ASR
        :param target_label: 目标标签 (没有的话不会进行 ASR 测试)
        :param exp_id: 实验 ID (如果没有会从 distill_config 中找)
        :param exp_desc: 实验描述
        :param make_test_per_epochs: 每多少个 epoch 生成一次测试样本
        :param save_ckpts_per_epochs: 每多少个 epoch 保存一次 Checkpoints
        :return: ModelDistiller 实例
        :raises ValueError: exp_id 和 distill_config['id'] 都没有提供实验 ID
        """
        try:
            config_check(
                distill_config,
                EXP_DISTILL_CONFIG_STRUCTURE[distill_config["distiller"]],
            )
        except KeyError as e:
            raise ValueError(f"Unknown distiller: {distill_config['distiller']}") from e

        if exp_id == "":
            if "id" in distill_config:
                exp_id = distill_config["id"]
            else:
                raise ValueError("exp_id is required if not in distill_config")

        if exp_desc == "" and "desc" in distill_config:
            exp_desc = distill_config["desc"]

        optimizer_class = getattr(optim, distill_config["optimizer"]["class"])

        match distill_config["distiller"]:
            case "VanillaModelDistiller":
                return modules.VanillaModelDistiller(
                    exp_id=exp_id,
                    teacher_model_or_exp=teacher_model_or_exp,
                    trigger_gen=trigger_gen,
                    student_model_class=student_model_class,
                    epochs=distill_config["epochs"],
                    lr=distill_config["lr"],
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    batch_size=distill_config["batch_size"],
                    alpha=distill_config["alpha"],
                    temperature=distill_config["temperature"],
                    optimizer_class=optimizer_class,
                    optimizer_params=distill_config["optimizer"]["params"],
                    target_label=target_label,
                    exp_desc=exp_desc,
                    seed=seed,
                    make_test_per_epochs=make_test_per_epochs,
                    save_ckpts_per_epochs=save_ckpts_per_epochs,
                )
            case "FeatureBasedModelDistiller":
                return modules.FeatureBasedModelDistiller(
                    exp_id=exp_id,
                    teacher_model_or_exp=teacher_model_or_exp,
                    student_model_class=student_model_class,
                    epochs=distill_config["epochs"],
                    lr=distill_config["lr"],
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    batch_size=distill_config["batch_size"],
                    alpha=distill_config["alpha"],
                    optimizer_class=optimizer_class,
                    optimizer_params=distill_config["optimizer"]["params"],
                    target_label=target_label,
                    trigger_gen=trigger_gen,
                    simkd_factor=distill_config["simkd_factor"],
                    exp_desc=exp_desc,
                    seed=seed,
                    make_test_per_epochs=make_test_per_epochs,
                    save_ckpts_per_epochs=save_ckpts_per_epochs,
                )
            case "RelationBasedModelDistiller":
                return modules.RelationBasedModelDistiller(
                    exp_id=exp_id,
                    teacher_model_or_exp=teacher_model_or_exp,
                    student_model_class=student_model_class,
                    epochs=distill_config["epochs"],
                    lr=distill_config["lr"],
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    batch_size=distill_config["batch_size"],
                    alpha=distill_config["alpha"],
                    beta=distill_config["beta"],
                    gamma=distill_config["gamma"],
                    temperature=distill_config["temperature"],
                    optimizer_class=optimizer_class,
                    optimizer_params=distill_config["optimizer"]["params"],
                    target_label=target_label,
                    trigger_gen=trigger_gen,
                    exp_desc=exp_desc,
                    seed=seed,
                    make_test_per_epochs=make_test_per_epochs,
                    save_ckpts_per_epochs=save_ckpts_per_epochs,
                )
