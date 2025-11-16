"""
SCALE-UP 检测方法实现 (Figure.2)

* Repo: https://github.com/JunfengGo/SCALE-UP
* Ref: Figure 2 of [Guo J, Li Y, Chen X, et al. Scale-up: An efficient black-box input-level backdoor detection via analyzing scaled prediction consistency[J]. arXiv preprint arXiv:2302.03251, 2023].
"""

import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from utils.data import DatasetWithInfo, TransformedDataset
from data_augs import MakeSimpleTransforms
from utils.funcs import auto_select_device, temp_seed, get_timestamp, print_section

from modules.abc import TriggerGenerator
from defense_modules.abc import DefenseModule
from configs import TENSORBOARD_LOGS_PATH, CHECKPOINTS_SAVE_PATH

from utils.visualization import visualize_records

_ckpt_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "scale_up")


class ScaleUp(DefenseModule):
    def __init__(
        self,
        test_id: str,
        model: nn.Module,
        dataset_info: DatasetWithInfo,
        trigger_generator: TriggerGenerator,
        *args,
        scaling_factors: list[float] = [3.0, 5.0, 7.0, 9.0, 11.0],
        **kwargs,
    ):
        """
        初始化 SCALE-UP 检测模块

        :param test_id: 用来标记本次 SCALE-UP 运行的测试 ID
        :param model: 待检测模型
        :param dataset_info: 数据集信息
        :param trigger_generator: 触发器生成器对象
        :param scaling_factors: SCALE-UP 中使用的缩放因子列表
        """
        self._test_id = test_id
        self._model = copy.deepcopy(model)  # 不影响原模型
        self._transforms_maker = MakeSimpleTransforms(input_shape=dataset_info.shape)
        self._dataset_info = dataset_info
        self._trigger_generator = trigger_generator
        self._scaling_factors = scaling_factors
        self._save_dir = os.path.join(_ckpt_save_dir, test_id)
        self._data_loader = DataLoader(
            dataset=TransformedDataset(
                dataset_info.test_set,
                transform=self._transforms_maker.normalize_standardize,
            ),
            batch_size=128,
            shuffle=True,
            num_workers=4,
        )
        os.makedirs(self._save_dir, exist_ok=True)

    @classmethod
    def is_mitigation(cls) -> bool:
        return False

    def _detect_batch(self, images: torch.Tensor):
        """
        对一个批次的图像使用 SCALE-UP 方法进行检测

        :param images: 待检测的一批图像张量，形状为 (n_batch, C, H, W)
        :return: (模型对这批图像原本的预测概率, 每个缩放因子对应的预测概率)，
                 形状分别为 (n_batch, 1) 和 (n_batch, num_factors)
        """
        device = images.device
        n_batch = images.size(0)

        self._model.to(device)
        self._model.eval()
        self._model.requires_grad_(False)

        num_factors = len(self._scaling_factors)

        # 获得没有放大的图像的预测结果
        with torch.no_grad():
            original_outputs = self._model(images)
            original_softmaxes = F.softmax(original_outputs, dim=-1)
            # shape (n_batch, )
            original_probs, original_preds = torch.max(original_softmaxes, dim=-1)

        original_probs = original_probs.unsqueeze(1)  # shape (n_batch, 1)
        original_preds = original_preds.unsqueeze(1)  # shape (n_batch, 1)

        # 存储每个缩放因子的预测结果
        scaled_probs = torch.zeros((n_batch, num_factors), device=device)

        for i, factor in enumerate(self._scaling_factors):
            # 因为我们的图像像素值在 [-1, 1] 内，为了符合原文实现
            # 先转换为 [0, 1]，再进行缩放，然后再转换回 [-1, 1]
            destandardized_images = self._transforms_maker.destandardize(images)
            scaled_images = destandardized_images * factor
            scaled_images = torch.clip(scaled_images, 0.0, 1.0)
            scaled_images = self._transforms_maker.standardize(scaled_images)

            # 看看模型对放大后的图像的预测结果
            with torch.no_grad():
                scaled_outputs = self._model(scaled_images)
                scaled_softmaxes = F.softmax(scaled_outputs, dim=-1)

            # 根据 Figure.2 进行实现，提取 original_preds 对应的概率
            probs_on_original_labels = torch.gather(
                scaled_softmaxes, dim=-1, index=original_preds
            )  # shape (n_batch, 1)

            scaled_probs[:, i] = probs_on_original_labels.squeeze(1)

        return original_probs, scaled_probs

    def detect(self):
        """
        使用 SCALE-UP 方法检测输入

        :return: 检测结果字典, { "normal_avg_probs": list, "triggered_avg_probs": list }
        """
        tensorboard_log_id = f"scale_up_{self._test_id}"
        tensorboard_log_dir = os.path.join(TENSORBOARD_LOGS_PATH, tensorboard_log_id)
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        device = auto_select_device()

        print_section(f"SCALE-UP Defense: {self._test_id}")

        normal_all_results_list = []
        triggered_all_results_list = []

        for images, _ in tqdm(self._data_loader, desc="SCALE-UP Detection"):
            images: torch.Tensor = images.to(device)
            triggered_images = self._trigger_generator.apply_trigger(images)

            # 先在正常图像上检测
            original_probs, scaled_probs = self._detect_batch(images)
            # 拼接起来
            normal_scale_probs = torch.cat(
                [original_probs, scaled_probs], dim=1
            )  # shape (n_batch, num_factors + 1)

            # 再在带有触发器的图像上检测
            triggered_original_probs, triggered_scaled_probs = self._detect_batch(
                triggered_images
            )
            triggered_scale_probs = torch.cat(
                [triggered_original_probs, triggered_scaled_probs], dim=1
            )  # shape (n_batch, num_factors + 1)

            normal_all_results_list.append(normal_scale_probs)
            triggered_all_results_list.append(triggered_scale_probs)

        # 计算最终的平均结果
        normal_all_results = torch.cat(
            normal_all_results_list, dim=0
        )  # shape (n_samples, num_factors + 1)
        triggered_all_results = torch.cat(
            triggered_all_results_list, dim=0
        )  # shape (n_samples, num_factors + 1)

        normal_all_avg_probs = (
            torch.mean(normal_all_results, dim=0).cpu().tolist()
        )  # shape (num_factors + 1, )
        triggered_all_avg_probs = (
            torch.mean(triggered_all_results, dim=0).cpu().tolist()
        )  # shape (num_factors + 1, )

        x = [1] + self._scaling_factors  # 横轴

        normal_vis_img = visualize_records(
            records={"Average Confidence": normal_all_avg_probs},
            records_x={"Factor": x},
        )

        triggered_vis_img = visualize_records(
            records={"Average Confidence": triggered_all_avg_probs},
            records_x={"Factor": x},
        )

        tb_writer.add_image(
            "Normal Images Average Confidence",
            normal_vis_img,
            dataformats="HWC",
        )
        tb_writer.add_image(
            "Triggered Images Average Confidence",
            triggered_vis_img,
            dataformats="HWC",
        )

        # 把结果写入 TensorBoard
        for i, factor in enumerate(x):
            tb_writer.add_scalar(
                "Normal_Images/Average_Confidence", normal_all_avg_probs[i], factor
            )
            tb_writer.add_scalar(
                "Triggered_Images/Average_Confidence",
                triggered_all_avg_probs[i],
                factor,
            )

        tb_writer.close()

        # 保存结果
        result_save_path = os.path.join(
            self._save_dir, f"detection_results_{get_timestamp()}.json"
        )

        result = {
            "normal_avg_probs": normal_all_avg_probs,
            "triggered_avg_probs": triggered_all_avg_probs,
        }

        with open(result_save_path, "w") as f:
            json.dump(result, f, indent=4)

        return result
