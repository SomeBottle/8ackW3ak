"""
运行单次 BackWeak 实验流程的脚本
"""

from utils.arg_parser import ExperimentArgParser

if __name__ == "__main__":
    args = ExperimentArgParser("BackWeak").parse()

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
from configs import BACKWEAK_EXP_CONFIG_STRUCTURE, set_selected_device
from typing import Type
from modules.factories import (
    NormalTrainerFactory,
    TriggerGeneratorFactory,
    DataPoisonerFactory,
    ModelTunerFactory,
    ModelDistillerFactory,
)
from modules import ASRTester, BATester, TITGTester, TriggerTester
from modules.abc import ExpBase, ModelDistiller
from defense_modules.abc import DefenseModule

from base import BackdoorExperimentBase


class BackWeakExperiment(BackdoorExperimentBase):
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
        config_check(config, BACKWEAK_EXP_CONFIG_STRUCTURE)
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
        运行单次 BackWeak 实验流程
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
        # -------------------------- 先注入后门
        self.inject_backdoor()

        exp_result_save_path = os.path.join(
            self._exp_result_save_dir, f"result_{get_timestamp()}.json"
        )
        config = self._config

        # ------------------------- 对防御部分的配置进行处理
        defense_config = config["defense"]
        defender_module_name = defense_config["defender"]
        defender_for = defense_config["for"]  # 标记是防御教师还是学生模型
        defender_params = defense_config.get("params", {})
        defender_class: Type[DefenseModule] | None = None

        if defender_for not in ["teacher", "student"]:
            raise ValueError("Defender 'for' field must be 'teacher' or 'student'")

        if defender_module_name != "none":
            # 指定了要进行防御
            defense_modules = importlib.import_module("defense_modules")
            try:
                defender_class = getattr(defense_modules, defender_module_name)
            except AttributeError as e:
                raise ValueError("Defender class not found") from e

        # ------------------------- 对教师进行防御

        clean_teacher = self._normal_trainer.get_trained_model()
        poisoned_teacher = self._teacher_tuner.get_tuned_model()

        self._collected_results["defense"] = {}

        if defender_class is not None and defender_for == "teacher":
            id_defender_poisoned = f"defend_t_poisoned_{self._config_hasher.current}"
            defender_poisoned_teacher = defender_class(
                test_id=id_defender_poisoned,
                model=poisoned_teacher,
                dataset_info=self._dataset_info_part_2,  # 防御方只有第二部分数据
                trigger_generator=self._trigger_generator,
                seed=self._global_seed,
                **defender_params,
            )

            if defender_class.is_mitigation():
                # 如果是要缓解教师模型的后门，那么要更新哈希
                # 因为缓解教师后门会影响后续蒸馏的结果
                self._config_hasher.chain_hash(defense_config)

                # 缓解式防御
                poisoned_teacher = defender_poisoned_teacher.mitigate()

                # 缓解后要测试 metrics
                metrics_mitigated_poisoned = self.test_metrics(poisoned_teacher)

                self._collected_results["defense"]["teacher"] = {
                    "id": {
                        "poisoned": id_defender_poisoned,
                    },
                    "metrics": {
                        "mitigated_poisoned": metrics_mitigated_poisoned,
                    },
                }
            else:
                # 检测式防御
                detect_result_poisoned = defender_poisoned_teacher.detect()

                self._collected_results["defense"]["teacher"] = {
                    "id": {
                        "poisoned": id_defender_poisoned,
                    },
                    "detection": {
                        "poisoned": detect_result_poisoned,
                    },
                }

        # ------------------------- 对蒸馏配置进行处理
        distill_config = config["distill"]
        distill_dataset_name = distill_config["dataset_name"]
        # 如果蒸馏数据集名称是 "auto"，则采用 basic.dataset_name 分割的数据集
        if distill_dataset_name == "auto":
            distill_dataset_info = self._dataset_info_part_2
            # 这里加的比较晚了，实验都跑了很多了，为了屎山兼容，这里临时移除掉 dataset_name 字段再哈希，看到这里的朋友可以把这行移除掉哦 (。_。)
            del distill_config["dataset_name"]
        else:
            # 否则采用指定的蒸馏集
            distill_dataset_info = DatasetWithInfo.from_name(distill_dataset_name)
        # 蒸馏配置也要哈希，蒸馏配置变了后面测试也要重跑
        self._config_hasher.chain_hash(distill_config)
        # -------------------------- 测试配置
        test_config = config["test"]

        self._hash_and_replace_conf_default_id(test_config["clean"], test_stage=True)
        self._hash_and_replace_conf_default_id(test_config["poisoned"], test_stage=True)

        self._distill_config = distill_config
        self._distill_dataset_info = distill_dataset_info

        # -------------------------- 开始测试蒸馏
        self._collected_results["info"]["test"] = {}

        distilled_student_clean = None
        distilled_student_poisoned = None

        if test_config["clean"]["test"]:
            # 从干净模型蒸馏并测试
            model_distiller_clean = self.test_distillation(
                exp_id=f"clean_{test_config['clean']['id']}",  # 加个 clean_ 前缀以区分
                exp_desc=test_config["clean"]["desc"],
                teacher_model=clean_teacher,
            )
            distilled_student_clean = model_distiller_clean.get_distilled_student()
            # 评估时采用蒸馏所用的数据集
            self._collected_results["results"]["clean_student"] = self.test_metrics(
                distilled_student_clean, dataset_info=distill_dataset_info
            )
            self._collected_results["info"]["test"]["clean"] = {
                "id": model_distiller_clean.exp_id,
                "time_elapsed": model_distiller_clean.get_time_elapsed(),
                "info_path": model_distiller_clean.get_exp_info_path(),
            }

        if test_config["poisoned"]["test"]:
            # 从污染模型蒸馏并测试
            model_distiller_poisoned = self.test_distillation(
                exp_id=f"poisoned_{test_config['poisoned']['id']}",  # 加个 poisoned_ 前缀以区分
                exp_desc=test_config["poisoned"]["desc"],
                teacher_model=poisoned_teacher,
            )
            distilled_student_poisoned = (
                model_distiller_poisoned.get_distilled_student()
            )
            self._collected_results["results"]["poisoned_student"] = self.test_metrics(
                distilled_student_poisoned, dataset_info=distill_dataset_info
            )
            self._collected_results["info"]["test"]["poisoned"] = {
                "id": model_distiller_poisoned.exp_id,
                "time_elapsed": model_distiller_poisoned.get_time_elapsed(),
                "info_path": model_distiller_poisoned.get_exp_info_path(),
            }

        # -------------------------- 测试教师的各项指标
        metrics_clean_teacher = self.test_metrics(
            clean_teacher, dataset_info=distill_dataset_info
        )
        metrics_poisoned_teacher = self.test_metrics(
            poisoned_teacher, dataset_info=distill_dataset_info
        )
        self._collected_results["results"]["clean_teacher"] = metrics_clean_teacher
        self._collected_results["results"][
            "poisoned_teacher"
        ] = metrics_poisoned_teacher

        # -------------------------- 对学生执行防御
        if defender_class is not None and defender_for == "student":
            id_defender_poisoned = f"defend_s_poisoned_{self._config_hasher.current}"
            if distilled_student_poisoned is not None:
                defender_poisoned_student = defender_class(
                    test_id=id_defender_poisoned,
                    model=distilled_student_poisoned,
                    dataset_info=self._dataset_info_part_2,  # 防御方只有第二部分数据
                    trigger_generator=self._trigger_generator,
                    seed=self._global_seed,
                    **defender_params,
                )
                if defender_class.is_mitigation():
                    # 缓解式防御
                    distilled_student_poisoned = defender_poisoned_student.mitigate()

                    metrics_mitigated_poisoned = self.test_metrics(
                        distilled_student_poisoned
                    )

                    self._collected_results["defense"]["student"] = {
                        "id": {
                            "poisoned": id_defender_poisoned,
                        },
                        "metrics": {
                            "mitigated_poisoned": metrics_mitigated_poisoned,
                        },
                    }

                else:
                    # 检测式防御
                    detect_result_poisoned = defender_poisoned_student.detect()

                    self._collected_results["defense"]["student"] = {
                        "id": {
                            "poisoned": id_defender_poisoned,
                        },
                        "detection": {
                            "poisoned": detect_result_poisoned,
                        },
                    }

        # -------------------------- 看看需不需要测试触发器的可见性指标
        trigger_test_config = config["test_trigger"]

        if trigger_test_config["perform"]:
            trigger_vis_save_dir = os.path.join(
                self._exp_result_save_dir, "trigger_vis"
            )
            trigger_tester = TriggerTester(
                dataset_info=self._dataset_info_part_1,
                data_transform_class=self._data_transform_teacher_cls,
                trigger_gen=self._trigger_generator,
                vis_save_dir=trigger_vis_save_dir,
                num_samples=trigger_test_config.get("num_samples", 5),
                seed=self._global_seed,
            )
            trigger_lpips_results = trigger_tester.test()
            self._collected_results["results"][
                "trigger_visibility"
            ] = trigger_lpips_results
            self._collected_results["info"]["test"]["trigger_visibility"] = {
                "save_dir": trigger_vis_save_dir
            }

        print_section(f"BackWeak Experiment [{self._exp_result_name}] Completed")
        print(self._collected_results["results"])

        # 存储实验结果
        try:
            with open(exp_result_save_path, "w", encoding="utf-8") as f:
                json.dump(self._collected_results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(
                f"Unable to save experiment result to {exp_result_save_path}"
            ) from e

    def inject_backdoor(self):
        """
        执行后门教师模型训练流程
        """
        config = self._config
        # 配置哈希链，前面阶段发生变化，后面阶段全都要重来
        self._config_hasher.chain_hash(config["basic"])

        # 实验结果收集
        collected_results = {"info": {}, "results": {}}  # 实验追溯信息, 实验结果

        # 记录总共耗时
        total_time_elapsed = 0.0

        # ------------------- 全局使用的配置
        global_seed = config["basic"]["seed"]
        make_test_per_epochs = config["validate"]["make_test_per_epochs"]
        save_ckpts_per_epochs = config["validate"]["save_ckpts_per_epochs"]

        # ------------------- 获取数据集并进行划分
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
        # ------------------- 全局设置一次随机种子
        fix_seed(global_seed)
        # ------------------- 初始化教师的数据增强
        try:
            data_transform_teacher_cls: Type[da_abc.MakeTransforms] = getattr(
                data_augs, config["basic"]["teacher"]["data_transform"]
            )
        except AttributeError as e:
            raise ValueError("Data transform class not found") from e

        # ------------------- 获得教师和学生的模型类
        img_h = dataset_info.shape[1]
        model_modules = importlib.import_module(f"models.size_{img_h}")

        try:
            teacher_model_cls: Type[nn.Module] = getattr(
                model_modules, config["basic"]["teacher"]["model"]
            )
        except AttributeError as e:
            raise ValueError("Model class not found") from e

        # ------------------- 初始化 base_train 阶段模块
        self._hash_and_replace_conf_default_id(config["base_train"])

        normal_trainer = NormalTrainerFactory.create(
            config["base_train"],
            model_class=teacher_model_cls,
            dataset_info=dataset_info_part_1,
            data_transform_class=data_transform_teacher_cls,
            seed=global_seed,
            make_test_per_epochs=make_test_per_epochs,
            save_ckpts_per_epochs=save_ckpts_per_epochs,
        )

        # ------------------- 获取后门目标标签
        target_label = config["backdoor"]["target_label"]
        # 后门的配置影响到后面实验，也要加入哈希链
        self._config_hasher.chain_hash(config["backdoor"])

        # ------------------- 初始化 trigger_gen 阶段模块
        self._hash_and_replace_conf_default_id(config["trigger_gen"])

        trigger_generator = TriggerGeneratorFactory.create(
            config["trigger_gen"],
            normal_trainer=normal_trainer,
            dataset_info=dataset_info_part_1,
            data_transform_class=data_transform_teacher_cls,
            target_label=target_label,
            seed=global_seed,
        )

        # ------------------- 初始化 data_poison 阶段模块
        self._hash_and_replace_conf_default_id(config["data_poison"])

        data_poisoner = DataPoisonerFactory.create(
            config["data_poison"],
            normal_trainer=normal_trainer,
            trigger_gen=trigger_generator,
            dataset_info=dataset_info_part_1,
            data_transform_class=data_transform_teacher_cls,
            target_label=target_label,
            seed=global_seed,
        )

        # ------------------- 初始化 teacher_tune 阶段模块
        self._hash_and_replace_conf_default_id(config["teacher_tune"])

        teacher_tuner = ModelTunerFactory.create(
            config["teacher_tune"],
            normal_trainer=normal_trainer,
            trigger_gen=trigger_generator,
            data_poisoner=data_poisoner,
            target_label=target_label,
            dataset_info=dataset_info_part_1,
            data_transform_class=data_transform_teacher_cls,
            seed=global_seed,
            make_test_per_epochs=make_test_per_epochs,
            save_ckpts_per_epochs=save_ckpts_per_epochs,
        )

        print_section(f"BackWeak Experiment [{self._exp_result_name}] Start")

        # 触发整个实验流程
        teacher_tuner.get_tuned_model()

        # 实验完成后记录信息
        stage_obj_pairs: list[tuple[str, ExpBase]] = [
            ("base_train", normal_trainer),
            ("trigger_gen", trigger_generator),
            ("data_poison", data_poisoner),
            ("teacher_tune", teacher_tuner),
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

        self._target_label = target_label
        self._dataset_info_part_1 = dataset_info_part_1
        self._dataset_info_part_2 = dataset_info_part_2
        self._data_transform_teacher_cls = data_transform_teacher_cls
        self._normal_trainer = normal_trainer
        self._trigger_generator = trigger_generator
        self._teacher_tuner = teacher_tuner
        self._model_modules = model_modules
        self._global_seed = global_seed
        self._make_test_per_epochs = make_test_per_epochs
        self._save_ckpts_per_epochs = save_ckpts_per_epochs
        self._collected_results = collected_results

    def test_metrics(
        self, model: nn.Module, dataset_info: DatasetWithInfo = None
    ) -> dict:
        """
        执行后门模型测试流程，计算各个指标

        :param model: 待测试的模型
        :param dataset_info: 待测试的数据集信息，如果为 None 则使用第一部分数据集
        :return: 测试结果字典 {"asr": float, "ba": float, "titg": float}
        """
        if dataset_info is None:
            dataset_info = self._dataset_info_part_1
        asr_tester = ASRTester(
            dataset_info=dataset_info,
            data_transform_class=self._data_transform_teacher_cls,
            trigger_gen=self._trigger_generator,
            target_label=self._target_label,
        )
        ba_tester = BATester(
            dataset_info=dataset_info,
            data_transform_class=self._data_transform_teacher_cls,
        )
        titg_tester = TITGTester(
            dataset_info=dataset_info,
            data_transform_class=self._data_transform_teacher_cls,
            trigger_gen=self._trigger_generator,
            target_label=self._target_label,
        )

        asr_results = asr_tester.test(model)
        ba_results = ba_tester.test(model)
        titg_results = titg_tester.test(model)

        return {
            "asr": asr_results["asr"],
            "ba": ba_results["ba"],
            "titg": titg_results["titg"],
        }

    def test_distillation(
        self, exp_id: str, exp_desc: str, teacher_model: nn.Module
    ) -> ModelDistiller:
        """
        执行模型蒸馏测试流程

        :param exp_id: 实验 ID
        :param exp_desc: 实验描述
        :param teacher_model: 教师模型
        """
        # 初始化学生的数据增强和模型类
        try:
            data_transform_student_cls: Type[da_abc.MakeTransforms] = getattr(
                data_augs, self._distill_config["student"]["data_transform"]
            )
        except AttributeError as e:
            raise ValueError("Data transform class not found") from e
        try:
            student_model_cls: Type[nn.Module] = getattr(
                self._model_modules, self._distill_config["student"]["model"]
            )
        except AttributeError as e:
            raise ValueError("Model class not found") from e

        model_distiller_clean = ModelDistillerFactory.create(
            exp_id=exp_id,
            exp_desc=exp_desc,
            distill_config=self._distill_config,
            teacher_model_or_exp=teacher_model,
            trigger_gen=self._trigger_generator,
            student_model_class=student_model_cls,
            dataset_info=self._distill_dataset_info,  # 用第二部分数据集蒸馏学生模型
            data_transform_class=data_transform_student_cls,
            target_label=self._target_label,
            seed=self._global_seed,
            make_test_per_epochs=self._make_test_per_epochs,
            save_ckpts_per_epochs=self._save_ckpts_per_epochs,
        )

        # 触发模型蒸馏流程
        model_distiller_clean.get_distilled_student()

        return model_distiller_clean


if __name__ == "__main__":
    BackWeakExperiment(
        config_path=args.config_path,
        output_dir=args.output_dir,
        force_run=True,
        device=args.device,
    ).run()
