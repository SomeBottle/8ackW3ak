"""
运行单次 ADBA 实验流程的脚本

* Ref: Ge Y, Wang Q, Zheng B, et al. Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation[C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 826-834.
"""

from utils.arg_parser import ExperimentArgParser

if __name__ == "__main__":
    args = ExperimentArgParser("ADBA").parse()

import json
import torch.nn as nn
import os
import importlib
import data_augs
import data_augs.abc as da_abc

# tomllib 在 Python 3.11+ 中可用，但是较低版本只能用第三方的 tomli 了
try:
    import tomllib  # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib

from utils.data import DatasetWithInfo
from utils.funcs import (
    get_timestamp,
    config_check,
    print_section,
    fix_seed,
)
from utils.hasher import ConfigHasher
from configs import ADBA_EXP_CONFIG_STRUCTURE, set_selected_device
from typing import Type
from modules.factories import (
    NormalTrainerFactory,
    ModelDistillerFactory,
)
from modules import ASRTester, BATester, TITGTester, TriggerTester
from modules.abc import ExpBase
from reprod_modules.adba import ADBA

from base import BackdoorExperimentBase


class ADBAExperiment(BackdoorExperimentBase):
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        force_run: bool = False,
        device: str = "auto",
    ):
        self._config_path = config_path
        self._output_dir = output_dir
        self._force_run = force_run
        self._device = device

        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            raise RuntimeError(f"Unable to load config {config_path}") from e
        # 先检查全局配置
        config_check(config, ADBA_EXP_CONFIG_STRUCTURE)
        # 配置设备
        set_selected_device(device)

        self._config = config
        # 哈希链
        self._config_hasher = ConfigHasher()
        # 实验名
        self._exp_result_name = os.path.splitext(os.path.basename(self._config_path))[0]
        # 实验结果保存目录
        self._exp_result_save_dir = os.path.join(
            self._output_dir, self._exp_result_name
        )
        os.makedirs(self._exp_result_save_dir, exist_ok=True)

    def _hash_and_replace_conf_default_id(self, conf: dict, test_stage: bool = False):
        """
        如果配置的 id 字段是 "default"，则自动计算哈希值进行替换

        :param conf: 阶段配置字典
        :param test_stage: 是否为测试阶段配置，测试阶段不更新哈希链
        """
        if conf["id"] == "default":
            conf["id"] = self._config_hasher.chain_hash(conf, inplace=not test_stage)
        else:
            # 否则采用指定的 id，但是哈希链还是得更新
            self._config_hasher.chain_hash(conf, inplace=not test_stage)

    def run(self):
        """
        运行单次 ADBA 实验流程
        """
        # 扫描目录检查是否已经有结果存在
        if not self._force_run:
            for fname in os.listdir(self._exp_result_save_dir):
                if fname.startswith("result_") and fname.endswith(".json"):
                    # 结果文件已存在，跳过实验
                    print(
                        f"Experiment [{self._exp_result_name}] result already exists at {self._exp_result_save_dir}, skip running."
                    )
                    return

        config = self._config
        # 配置哈希链，前面阶段发生变化，后面阶段全都要重来
        self._config_hasher.chain_hash(config["basic"])

        # 实验结果收集
        collected_results = {"info": {}, "results": {}}

        # 记录总共耗时
        total_time_elapsed = 0.0

        # 实验结果存储路径
        exp_result_save_path = os.path.join(
            self._exp_result_save_dir, f"result_{get_timestamp()}.json"
        )

        # ------------------- 全局使用的配置
        global_seed = config["basic"]["seed"]
        make_test_per_epochs = config["validate"]["make_test_per_epochs"]
        save_ckpts_per_epochs = config["validate"]["save_ckpts_per_epochs"]

        # ------------------- 获取数据集并进行划分

        # 注：根据原文 3.1.2，ADBA 假设教师模型训练和学生蒸馏用的是相同的数据集
        # 这里我和 BackWeak 保持一致，对数据集进行划分，训练用 part_1，蒸馏用 part_2

        dataset_info = DatasetWithInfo.from_name(config["basic"]["dataset_name"])

        dataset_info_part_1, dataset_info_part_2 = DatasetWithInfo.split_into_two(
            dataset_info=dataset_info,
            seed=global_seed,
        )

        collected_results["info"]["dataset"] = {
            "name": dataset_info.name,
            "part_1": {
                "train": len(dataset_info_part_1.train_set),
                "val": len(dataset_info_part_1.val_set),
                "test": len(dataset_info_part_1.test_set),
            },
            "part_2": {
                "train": len(dataset_info_part_2.train_set),
                "val": len(dataset_info_part_2.val_set),
                "test": len(dataset_info_part_2.test_set),
            },
        }

        # 根据数据集形状导入对应的模型模块
        img_h = dataset_info.shape[1]
        model_modules = importlib.import_module(f"models.size_{img_h}")
        # ------------------- 全局设置一次随机种子
        fix_seed(global_seed)

        # ------------------- 取出教师模型类以及相应数据增强

        try:
            teacher_model_cls: Type[nn.Module] = getattr(
                model_modules, config["basic"]["teacher"]["model"]
            )
        except AttributeError as e:
            raise ValueError("Model class not found") from e

        try:
            teacher_data_transform_cls: Type[da_abc.MakeTransforms] = getattr(
                data_augs, config["basic"]["teacher"]["data_transform"]
            )
        except AttributeError as e:
            raise ValueError("Data transform class not found") from e

        # ------------------- 初始化 benign_train 阶段模块
        self._hash_and_replace_conf_default_id(config["benign_train"])

        benign_trainer = NormalTrainerFactory.create(
            config["benign_train"],
            model_class=teacher_model_cls,
            dataset_info=dataset_info_part_1,
            data_transform_class=teacher_data_transform_cls,
            seed=global_seed,
            make_test_per_epochs=make_test_per_epochs,
            save_ckpts_per_epochs=save_ckpts_per_epochs,
        )

        # ------------------- 后门配置
        target_label = config["backdoor"]["target_label"]
        # 计入哈希链
        self._config_hasher.chain_hash(config["backdoor"])

        # ------------------- 初始化 ADBA 模块
        adba_config = config["adba"]
        self._hash_and_replace_conf_default_id(adba_config)

        # 影子学生模型类
        try:
            adba_shadow_model_cls: Type[nn.Module] = getattr(
                model_modules, adba_config["shadow"]["model"]
            )
        except AttributeError as e:
            raise ValueError("Model class not found") from e

        adba_exp = ADBA(
            teacher_model_class=teacher_model_cls,
            shadow_model_class=adba_shadow_model_cls,
            epochs=adba_config["epochs"],
            alpha=adba_config["alpha"],
            temperature=adba_config["temperature"],
            beta=adba_config["beta"],
            mu=adba_config["mu"],
            p=adba_config["p"],
            c=adba_config["c"],
            target_label=target_label,
            batch_size=adba_config["batch_size"],
            teacher_lr=adba_config["teacher_lr"],
            trigger_lr=adba_config["trigger_lr"],
            dataset_info=dataset_info_part_1,
            data_transform_class=teacher_data_transform_cls,
            exp_id=adba_config["id"],
            exp_desc=adba_config["desc"],
            make_test_per_epochs=make_test_per_epochs,
            save_ckpts_per_epochs=save_ckpts_per_epochs,
            seed=global_seed,
        )

        # ------------------- 实验开始
        print_section(f"ADBA Experiment [{self._exp_result_name}] Start")

        # 触发整个实验流程
        benign_trainer.get_model()  # 先训练良性教师模型
        adba_exp.get_model()  # 再训练 ADBA 模型

        # ------------------- 实验完成后记录信息
        stage_obj_pairs: list[tuple[str, ExpBase]] = [
            ("benign_train", benign_trainer),
            ("adba", adba_exp),
        ]
        for stage_name, stage_obj in stage_obj_pairs:
            time_elapsed = stage_obj.get_time_elapsed()
            total_time_elapsed += time_elapsed
            collected_results["info"][stage_name] = {
                "id": stage_obj.exp_id,
                "time_elapsed": time_elapsed,
                "info_path": stage_obj.get_exp_info_path(),
            }

        # 总耗时
        collected_results["info"]["total_time_elapsed"] = total_time_elapsed

        # ------------------- 准备测试器
        asr_tester = ASRTester(
            dataset_info=dataset_info_part_1,
            data_transform_class=teacher_data_transform_cls,
            trigger_gen=adba_exp.get_trigger_generator(),
            target_label=target_label,
        )
        ba_tester = BATester(
            dataset_info=dataset_info_part_1,
            data_transform_class=teacher_data_transform_cls,
        )
        titg_tester = TITGTester(
            dataset_info=dataset_info_part_1,
            data_transform_class=teacher_data_transform_cls,
            trigger_gen=adba_exp.get_trigger_generator(),
            target_label=target_label,
        )

        # ------------------- 模型蒸馏测试

        collected_results["info"]["test"] = {}

        distill_config = config["test_distill"]
        self._hash_and_replace_conf_default_id(distill_config, test_stage=True)

        # 初始化学生的数据增强和模型类
        try:
            data_transform_test_student_cls: Type[da_abc.MakeTransforms] = getattr(
                data_augs, distill_config["student"]["data_transform"]
            )
        except AttributeError as e:
            raise ValueError("Data transform class not found") from e
        try:
            test_student_model_cls: Type[nn.Module] = getattr(
                model_modules, distill_config["student"]["model"]
            )
        except AttributeError as e:
            raise ValueError("Model class not found") from e

        # 获得良性教师和 ADBA 教师
        benign_teacher = benign_trainer.get_model()
        adba_teacher = adba_exp.get_model()

        # 从良性教师蒸馏
        model_distiller_benign = ModelDistillerFactory.create(
            exp_id=f'benign_{distill_config["id"]}',  # 加个 benign_ 前缀以区分
            exp_desc=distill_config["desc"],
            distill_config=distill_config,
            teacher_model_or_exp=benign_teacher,
            trigger_gen=adba_exp.get_trigger_generator(),
            student_model_class=test_student_model_cls,
            dataset_info=dataset_info_part_2,
            data_transform_class=data_transform_test_student_cls,
            target_label=target_label,
            seed=global_seed,
            make_test_per_epochs=make_test_per_epochs,
            save_ckpts_per_epochs=save_ckpts_per_epochs,
        )

        # 从 ADBA 教师蒸馏
        model_distiller_adba = ModelDistillerFactory.create(
            exp_id=f'adba_{distill_config["id"]}',  # 加个 adba_ 前缀以区分
            exp_desc=distill_config["desc"],
            distill_config=distill_config,
            teacher_model_or_exp=adba_teacher,
            trigger_gen=adba_exp.get_trigger_generator(),
            student_model_class=test_student_model_cls,
            dataset_info=dataset_info_part_2,
            data_transform_class=data_transform_test_student_cls,
            target_label=target_label,
            seed=global_seed,
            make_test_per_epochs=make_test_per_epochs,
            save_ckpts_per_epochs=save_ckpts_per_epochs,
        )

        # 触发模型蒸馏
        benign_student = model_distiller_benign.get_model()
        adba_student = model_distiller_adba.get_model()

        collected_results["info"]["test"]["clean"] = {
            "id": model_distiller_benign.exp_id,
            "time_elapsed": model_distiller_benign.get_time_elapsed(),
            "info_path": model_distiller_benign.get_exp_info_path(),
        }

        collected_results["info"]["test"]["poisoned"] = {
            "id": model_distiller_adba.exp_id,
            "time_elapsed": model_distiller_adba.get_time_elapsed(),
            "info_path": model_distiller_adba.get_exp_info_path(),
        }

        # 对 ASR, BA 进行测试
        asr_benign_teacher = asr_tester.test(benign_teacher)
        asr_benign_student = asr_tester.test(benign_student)
        asr_adba_teacher = asr_tester.test(adba_teacher)
        asr_adba_student = asr_tester.test(adba_student)
        ba_benign_teacher = ba_tester.test(benign_teacher)
        ba_benign_student = ba_tester.test(benign_student)
        ba_adba_teacher = ba_tester.test(adba_teacher)
        ba_adba_student = ba_tester.test(adba_student)
        titg_benign_teacher = titg_tester.test(benign_teacher)
        titg_benign_student = titg_tester.test(benign_student)
        titg_adba_teacher = titg_tester.test(adba_teacher)
        titg_adba_student = titg_tester.test(adba_student)

        collected_results["results"] = {
            "benign_teacher": {
                "asrc": asr_benign_teacher,
                "ba": ba_benign_teacher,
                "titg": titg_benign_teacher,
            },
            "benign_student": {
                "asrc": asr_benign_student,
                "ba": ba_benign_student,
                "titg": titg_benign_student,
            },
            "adba_teacher": {
                "asr": asr_adba_teacher,
                "ba": ba_adba_teacher,
                "titg": titg_adba_teacher,
            },
            "adba_student": {
                "asr": asr_adba_student,
                "ba": ba_adba_student,
                "titg": titg_adba_student,
            },
        }

        # -------------------------- 看看需不需要测试触发器的可见性指标
        trigger_test_config: dict = config.get(
            "test_trigger",
            {
                "perform": False,
            },
        )

        if trigger_test_config["perform"]:
            trigger_vis_save_dir = os.path.join(
                self._exp_result_save_dir, "trigger_vis"
            )
            trigger_tester = TriggerTester(
                dataset_info=dataset_info_part_1,
                data_transform_class=teacher_data_transform_cls,
                trigger_gen=adba_exp.get_trigger_generator(),
                vis_save_dir=trigger_vis_save_dir,
                num_samples=trigger_test_config.get("num_samples", 5),
                seed=global_seed,
            )
            trigger_lpips_results = trigger_tester.test()
            collected_results["results"]["trigger_visibility"] = trigger_lpips_results
            collected_results["info"]["test"]["trigger_visibility"] = {
                "save_dir": trigger_vis_save_dir
            }

        print_section(f"ADBA Experiment [{self._exp_result_name}] Completed")
        print(collected_results["results"])

        # 存储实验结果
        try:
            with open(exp_result_save_path, "w", encoding="utf-8") as f:
                json.dump(collected_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(
                f"Unable to save experiment result to {exp_result_save_path}"
            ) from e


if __name__ == "__main__":
    ADBAExperiment(
        config_path=args.config_path,
        output_dir=args.output_dir,
        force_run=True,
        device=args.device,
    ).run()
