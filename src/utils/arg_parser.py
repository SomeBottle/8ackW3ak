"""
实验命令行参数处理模块
"""

import argparse


class ExperimentArgParser:

    def __init__(self, experiment_name: str):
        self._parser = argparse.ArgumentParser(
            description=f"Run a single {experiment_name} experiment"
        )
        # 使用的数据集名字
        self._parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="Configuration file (.toml) path for the experiment",
            dest="config_path",
        )
        self._parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="auto",
            help="Device used for experiment, default to 'auto'",
            dest="device",
        )
        self._parser.add_argument(
            "--od",
            "--output-dir",
            type=str,
            default="./outputs",
            help="Directory to save outputs, default to ./outputs",
            dest="output_dir",
        )

    def parse(self) -> argparse.Namespace:
        return self._parser.parse_args()


class BatchExperimentArgParser:

    def __init__(self):
        self._parser = argparse.ArgumentParser(description=f"Run multiple experiments")
        # 使用的数据集名字
        self._parser.add_argument(
            "--cd",
            "--config-dir",
            type=str,
            required=True,
            help="Directory containing configuration files (.toml) for the experiments, it can have a nested subdirectory structure.",
            dest="config_dir",
        )
        self._parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="auto",
            help="Device used for experiment, default to 'auto'",
            dest="device",
        )
        self._parser.add_argument(
            "--ra",
            "--run-all",
            action="store_true",
            help="Run the experiment even if the results already exists",
            dest="run_all",
        )
        self._parser.add_argument(
            "--type",
            type=str,
            choices=["adba", "backweak", "scar"],
            required=True,
            help="Type of experiments to run: adba, backweak, or scar.",
            dest="experiment_type",
        )
        self._parser.add_argument(
            "--od",
            "--output-dir",
            type=str,
            default="./outputs_batch",
            help="Directory to save outputs, default to ./outputs_batch, the directory structure will be preserved in the output directory.",
            dest="output_dir",
        )
        self._parser.add_argument(
            "--log-dir",
            type=str,
            default="./logs_multiple_exps",
            help="Directory to save logs, default to ./logs_multiple_exps",
            dest="log_dir",
        )

    def parse(self) -> argparse.Namespace:
        return self._parser.parse_args()
