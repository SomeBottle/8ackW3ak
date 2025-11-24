"""
触发器生成模块工厂类
"""

import modules
import torch.optim as optim

from utils.funcs import config_check
from utils.data import DatasetWithInfo
from modules.abc import TriggerGenerator, NormalTrainer
from data_augs.abc import MakeTransforms
from typing import Type

# trigger_gen 环节不同 generator 各自的配置
# NOTE: 以 ? 结尾的字段是非必要的。
EXP_TRIGGER_GEN_CONFIG_STRUCTURE = {
    "WeakUAPGenerator": {
        "l_inf_r_over_255": (int, None),
        "lambda_margin": (float, None),
        "budget_asr": (float, None),
        "mu_margin": (float, None),
        "initialization?": (str, "zero"),  # "zero" or "random"
        "lr": (float, None),
        "epochs": (int, None),
        "batch_size": (int, None),
        "optimizer": {
            "class": (str, None),
            "params": (dict, {}),
        },
    },
    "BadNetsTriggerGenerator": {
        "patch_size": (int, 5),
    },
}


class TriggerGeneratorFactory:
    @staticmethod
    def create(
        trigger_gen_config: dict,
        normal_trainer: NormalTrainer,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        target_label: int,
        seed: int,
    ) -> TriggerGenerator:
        """
        根据配置字典创建 TriggerGenerator 实例

        :param trigger_gen_config: trigger_gen 实验配置字典
        :param normal_trainer: NormalTrainer 实例
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param target_label: 目标标签
        :param seed: 随机种子
        :return: TriggerGenerator 实例
        :raises ValueError: 未知的 generator
        """
        try:
            config_check(
                trigger_gen_config,
                EXP_TRIGGER_GEN_CONFIG_STRUCTURE[trigger_gen_config["generator"]],
            )
        except KeyError as e:
            raise ValueError(
                f"Unknown trigger generator: {trigger_gen_config['generator']}"
            ) from e

        match trigger_gen_config["generator"]:
            case "WeakUAPGenerator":
                l_inf_r = trigger_gen_config["l_inf_r_over_255"] / 255.0
                optimizer_class = getattr(
                    optim, trigger_gen_config["optimizer"]["class"]
                )
                trigger_initialization = trigger_gen_config.get(
                    "initialization", "zero"
                )
                return modules.WeakUAPGenerator(
                    normal_trainer=normal_trainer,
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    exp_id=trigger_gen_config["id"],
                    exp_desc=trigger_gen_config["desc"],
                    l_inf_r=l_inf_r,
                    budget_asr=trigger_gen_config["budget_asr"],
                    lambda_margin=trigger_gen_config["lambda_margin"],
                    mu_margin=trigger_gen_config["mu_margin"],
                    lr=trigger_gen_config["lr"],
                    epochs=trigger_gen_config["epochs"],
                    batch_size=trigger_gen_config["batch_size"],
                    initialization=trigger_initialization,
                    target_label=target_label,
                    optimizer_class=optimizer_class,
                    optimizer_params=trigger_gen_config["optimizer"]["params"],
                    seed=seed,
                )
            case "BadNetsTriggerGenerator":
                return modules.BadNetsTriggerGenerator(
                    exp_id=trigger_gen_config["id"],
                    exp_desc=trigger_gen_config["desc"],
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    patch_size=trigger_gen_config["patch_size"],
                )
