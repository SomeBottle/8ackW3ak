"""
SCAR 模块工厂类
"""

import importlib
import torch.nn as nn

from reprod_modules.scar import SCAR
from reprod_modules.oscar import OSCAR
from reprod_modules.abc import SCARBase
from utils.funcs import config_check
from utils.data import DatasetWithInfo
from modules.abc import TriggerGenerator
from data_augs.abc import MakeTransforms
from typing import Type

# SCAR 和 OSCAR 的配置
SCAR_CONFIG_STRUCTURE = {
    "SCAR": {
        "outer_epochs": (int, None),
        "inner_updates": (int, None),
        "fixed_point_iters": (int, None),
        "outer_grad_batches": (int, None),
        "teacher_lr": (float, None),
        "surrogate_lr": (float, None),
        "alpha": (float, None),
        "beta": (float, None),
        "gamma": (float, None),
        "delta": (float, None),
        "temperature": (float, None),
        "batch_size": (int, None),
        "surrogate": {
            "model": (str, None),
        },
    },
    "OSCAR": {
        "epochs": (int, None),
        "lr": (float, None),
        "alpha": (float, None),
        "batch_size": (int, None),
    },
}


class ReprodSCARFactory:
    @staticmethod
    def create(
        scar_config: dict,
        trigger_generator: TriggerGenerator,
        teacher_model_class: Type[nn.Module],
        teacher_data_transform_class: Type[MakeTransforms],
        target_label: int,
        dataset_info: DatasetWithInfo,
        seed: int,
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 2,
    ) -> SCARBase:
        """
        根据配置字典创建 SCAR 实例

        :param scar_config: SCAR 实验配置字典
        :param trigger_generator: 触发器生成模块
        :param teacher_model_class: 教师模型类
        :param teacher_data_transform_class: 教师数据增强模块类
        :param target_label: 目标标签
        :param dataset_info: 数据集信息
        :param seed: 随机种子
        :param make_test_per_epochs: 每多少个 epoch 进行一次测试
        :param save_ckpts_per_epochs: 每多少个 epoch 保存一次 checkpoint
        :return: SCAR 实例
        :raises ValueError: 未知的 ExpBase
        """
        try:
            config_check(scar_config, SCAR_CONFIG_STRUCTURE[scar_config["solution"]])
        except KeyError as e:
            raise ValueError(f"Unknown SCAR solution: {scar_config['solution']}") from e

        img_h = dataset_info.shape[1]
        model_modules = importlib.import_module(f"models.size_{img_h}")

        match scar_config["solution"]:
            case "SCAR":
                # SCAR 还需要代理学生模型
                try:
                    surrogate_model_class = getattr(
                        model_modules, scar_config["surrogate"]["model"]
                    )
                except AttributeError as e:
                    raise ValueError(
                        f"Unknown surrogate model: {scar_config['surrogate']['model']}"
                    ) from e

                return SCAR(
                    trigger_preoptimizer=trigger_generator,
                    teacher_model_class=teacher_model_class,
                    surrogate_model_class=surrogate_model_class,
                    outer_epochs=scar_config["outer_epochs"],
                    inner_updates=scar_config["inner_updates"],
                    fixed_point_iters=scar_config["fixed_point_iters"],
                    outer_grad_batches=scar_config["outer_grad_batches"],
                    teacher_lr=scar_config["teacher_lr"],
                    surrogate_lr=scar_config["surrogate_lr"],
                    alpha=scar_config["alpha"],
                    beta=scar_config["beta"],
                    gamma=scar_config["gamma"],
                    delta=scar_config["delta"],
                    temperature=scar_config["temperature"],
                    target_label=target_label,
                    dataset_info=dataset_info,
                    data_transform_class=teacher_data_transform_class,
                    batch_size=scar_config["batch_size"],
                    exp_id=scar_config["id"],
                    exp_desc=scar_config["desc"],
                    make_test_per_epochs=make_test_per_epochs,
                    save_ckpts_per_epochs=save_ckpts_per_epochs,
                    seed=seed,
                )
            case "OSCAR":
                return OSCAR(
                    trigger_preoptimizer=trigger_generator,
                    teacher_model_class=teacher_model_class,
                    epochs=scar_config["epochs"],
                    lr=scar_config["lr"],
                    alpha=scar_config["alpha"],
                    target_label=target_label,
                    dataset_info=dataset_info,
                    data_transform_class=teacher_data_transform_class,
                    batch_size=scar_config["batch_size"],
                    exp_id=scar_config["id"],
                    exp_desc=scar_config["desc"],
                    make_test_per_epochs=make_test_per_epochs,
                    save_ckpts_per_epochs=save_ckpts_per_epochs,
                    seed=seed,
                )
