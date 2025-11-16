"""
执行一批实验的脚本
"""

import os

from utils.arg_parser import BatchExperimentArgParser
from utils.logger import Logger

from base import BackdoorExperimentBase
from typing import Type

from run_single_adba import ADBAExperiment
from run_single_backweak import BackWeakExperiment
from run_single_scar import SCARExperiment


# 路径中包含这些目录名的将被忽略
IGNORED_DIRS = [".ipynb_checkpoints", "__pycache__", ".git"]

if __name__ == "__main__":
    args = BatchExperimentArgParser().parse()
    config_dir = args.config_dir
    experiment_type = args.experiment_type
    output_dir = args.output_dir
    log_dir = args.log_dir
    force_run = args.run_all
    device = args.device

    class_map: dict[str, Type[BackdoorExperimentBase]] = {
        "backweak": BackWeakExperiment,
        "scar": SCARExperiment,
        "adba": ADBAExperiment,
    }

    try:
        experiment_class = class_map[experiment_type]
    except KeyError:
        raise ValueError(f"Experiment type '{experiment_type}' is not supported.")

    logger = Logger(name=experiment_type, log_save_dir=log_dir).logger

    experiment_tomls: list[str] = []  # 存储目录中所有实验配置文件路径

    for root, _, files in os.walk(config_dir):
        if any(ignored in root for ignored in IGNORED_DIRS):
            continue  # 忽略指定目录
        for file in files:
            if file.endswith(".toml"):
                full_path = os.path.join(root, file)
                experiment_tomls.append(full_path)

    # 执行实验，保留目录结构
    for toml_path in experiment_tomls:
        # 计算相对于 config_dir 的子路径
        relative_path = os.path.relpath(toml_path, config_dir)
        # 构建输出目录，保留子目录结构
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_subdir, exist_ok=True)

        try:
            logger.info(f"Running experiment with config: {toml_path}")
            experiment_class(
                config_path=toml_path,
                output_dir=output_subdir,
                force_run=force_run,
                device=device,
            ).run()
        except Exception as e:
            # 防止实验崩溃导致后续实验无法进行
            logger.error(f"Experiment with config {toml_path} failed with error: {e}")
