"""
(复现部分) 触发器生成模块工厂类
"""
from utils.funcs import config_check
from utils.data import DatasetWithInfo
from modules.abc import TriggerGenerator, NormalTrainer, ModelDistiller
from data_augs.abc import MakeTransforms
from typing import Type
from modules.factories import TriggerGeneratorFactory
from reprod_modules.scar import SCARTriggerPreoptimizer

# trigger_gen 环节不同 generator 各自的配置
EXP_TRIGGER_GEN_CONFIG_STRUCTURE = {
    "SCARTriggerPreoptimizer": {
        "l_inf_r_over_255": (int, None),
        "lr": (float, None),
        "epochs": (int, None),
        "batch_size": (int, None),
    }
}


class ReprodTriggerGeneratorFactory:
    @staticmethod
    def create(
        trigger_gen_config: dict,
        normal_trainer: NormalTrainer,
        model_distiller: ModelDistiller,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        target_label: int,
        seed: int,
    ) -> TriggerGenerator:
        """
        根据配置字典创建 TriggerGenerator 实例

        :param trigger_gen_config: trigger_gen 实验配置字典
        :param normal_trainer: NormalTrainer 实例 (Benign Model)
        :param model_distiller: ModelDistiller 实例 (Student distilled from the Benign Model)
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param target_label: 目标标签
        :param seed: 随机种子
        """
        try:
            config_check(
                trigger_gen_config,
                EXP_TRIGGER_GEN_CONFIG_STRUCTURE[trigger_gen_config["generator"]],
            )
        except KeyError as e:
            # 这个模块没有，就去 BackWeak 主模块
            print(
                f"'{trigger_gen_config['generator']}' not found in ReprodTriggerGeneratorFactory, try TriggerGeneratorFactory"
            )
            return TriggerGeneratorFactory.create(
                trigger_gen_config=trigger_gen_config,
                normal_trainer=normal_trainer,
                dataset_info=dataset_info,
                data_transform_class=data_transform_class,
                target_label=target_label,
                seed=seed,
            )

        epsilon = trigger_gen_config["l_inf_r_over_255"] / 255.0  # L_inf 范数约束

        match trigger_gen_config["generator"]:
            case "SCARTriggerPreoptimizer":
                return SCARTriggerPreoptimizer(
                    exp_id=trigger_gen_config["id"],
                    exp_desc=trigger_gen_config["desc"],
                    teacher_trainer=normal_trainer,
                    student_distiller=model_distiller,
                    dataset_info=dataset_info,
                    data_transform_class=data_transform_class,
                    target_label=target_label,
                    epsilon=epsilon,
                    epochs=trigger_gen_config["epochs"],
                    lr=trigger_gen_config["lr"],
                    batch_size=trigger_gen_config["batch_size"],
                    seed=seed,
                )
