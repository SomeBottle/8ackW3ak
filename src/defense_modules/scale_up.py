"""
SCALE-UP 检测方法实现

* Repo: https://github.com/JunfengGo/SCALE-UP, https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/SCALE_UP.py
* 和 SCALE_UP.py 的实现不同，更遵循原文，为每个类别独立计算均值和标准差。
* Ref: Guo J, Li Y, Chen X, et al. Scale-up: An efficient black-box input-level backdoor detection via analyzing scaled prediction consistency[J]. arXiv preprint arXiv:2302.03251, 2023.
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
from sklearn import metrics

from utils.data import DatasetWithInfo, TransformedDataset
from utils.data_funcs import balanced_split_into_two
from data_augs import MakeSimpleTransforms
from utils.funcs import (
    auto_select_device,
    get_timestamp,
    print_section,
    auto_num_workers,
    temp_seed,
)

from modules.abc import TriggerGenerator
from defense_modules.abc import DefenseModule
from configs import TENSORBOARD_LOGS_PATH, CHECKPOINTS_SAVE_PATH

from utils.visualization import visualize_records, visualize_roc_curve

_ckpt_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "scale_up")


class ScaleUp(DefenseModule):
    def __init__(
        self,
        test_id: str,
        model: nn.Module,
        dataset_info: DatasetWithInfo,
        trigger_generator: TriggerGenerator,
        target_label: int,
        *args,
        data_portion: float = 0.05,
        T: float = 0.5,
        scaling_factors: list[float] = [3.0, 5.0, 7.0, 9.0, 11.0],
        seed: int = 42,
        **kwargs,
    ):
        """
        初始化 SCALE-UP 检测模块

        :param test_id: 用来标记本次 SCALE-UP 运行的测试 ID
        :param model: 待检测模型
        :param dataset_info: 数据集信息
        :param trigger_generator: 触发器生成器对象
        :param target_label: 攻击目标标签
        :param data_portion: 在 Data-limited 情况下使用的验证数据比例 (即假设防御方有少许 benign 样本的情况)，如果为 0 则为 Data-free (假设防御方没有 benign 样本)
        :param T: 阈值，SPC 值超出这个阈值则认为是触发器样本。在 Data-limited 情况下，这个阈值表示偏离 benign 样本 SPC 均值几个标准差。
        :param scaling_factors: SCALE-UP 中使用的缩放因子列表
        :param seed: 随机种子
        """
        self._test_id = test_id
        self._model = copy.deepcopy(model)  # 不影响原模型
        self._transforms_maker = MakeSimpleTransforms(input_shape=dataset_info.shape)
        self._dataset_info = dataset_info
        self._target_label = target_label
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
            num_workers=auto_num_workers(),
        )
        self._data_limited = False
        self._T = T
        self._seed = seed
        os.makedirs(self._save_dir, exist_ok=True)

        # 准备良性样本
        if abs(data_portion) > 1e-10:
            # Data-limited 情况
            self._data_limited = True
            # 计算良性样本的 SPC 均值和标准差
            # 用于后方 Z-Score 计算
            _, benign_dataset = balanced_split_into_two(
                dataset=dataset_info.val_set,
                latter_size_or_ratio=data_portion,
                random_state=seed,
            )
            benign_dataloader = DataLoader(
                dataset=TransformedDataset(
                    benign_dataset,
                    transform=self._transforms_maker.normalize_standardize,
                ),
                batch_size=128,
                shuffle=False,
                num_workers=auto_num_workers(),
            )
            device = auto_select_device()
            self._model.to(device)
            self._model.eval()
            all_batch_spc = []
            all_labels = []
            for images, labels in tqdm(
                benign_dataloader, desc="Calculating Benign SPC Stats"
            ):
                images: torch.Tensor = images.to(device)
                labels: torch.Tensor = labels.to(device)

                # (和 BackdoorBox 的实现不同) 这里和后面保持一致，过滤掉原本分类就不正确的
                with torch.no_grad():
                    outputs = self._model(images)
                    model_preds = torch.argmax(outputs, dim=-1)
                    correct_mask = model_preds == labels

                # 计算这一批每个样本的 SPC 值
                batch_spc = torch.zeros_like(labels, dtype=torch.float32)

                for factor in self._scaling_factors:
                    scaled_images = self._scale_image(images, factor)
                    with torch.no_grad():
                        outputs = self._model(scaled_images)
                        model_preds = torch.argmax(outputs, dim=-1)
                        batch_spc += (model_preds == labels).float()

                # 除以因子数量，得到 SPC
                batch_spc /= len(self._scaling_factors)  # shape (n_batch, )
                all_batch_spc.append(batch_spc[correct_mask])
                all_labels.append(labels[correct_mask])

            # 得到 benign 数据中所有样本的 SPC 值
            all_spc_tensor = torch.cat(all_batch_spc, dim=0)  # shape (n_samples, )
            all_labels_tensor = torch.cat(all_labels, dim=0)  # shape (n_samples, )

            # 有的类别可能没有样本，因此还是计算一下这些样本总共的 SPC 均值和标准差
            benign_spc_mean = torch.mean(all_spc_tensor).item()
            benign_spc_std = torch.std(all_spc_tensor).item()
            self._fallback_spc_stats = {
                "mean": benign_spc_mean,
                "std": benign_spc_std,
            }

            # 按类别计算均值和标准差
            # 所有的标签
            class_labels = torch.unique(all_labels_tensor)
            classwise_spc_stats = {}
            for c in class_labels:
                class_c_mask = all_labels_tensor == c
                class_c_spc = all_spc_tensor[class_c_mask]
                if len(class_c_spc) > 1:
                    class_c_mean = torch.mean(class_c_spc).item()
                    class_c_std = torch.std(class_c_spc).item()
                else:
                    class_c_mean = benign_spc_mean
                    # 注意，只有一个样本的时候 std 会返回 NaN
                    class_c_std = benign_spc_std
                classwise_spc_stats[c.item()] = {
                    "mean": class_c_mean,
                    "std": class_c_std,
                }
            self._classwise_spc_stats = classwise_spc_stats

    @classmethod
    def is_mitigation(cls) -> bool:
        return False

    def _scale_image(self, images: torch.Tensor, factor: float) -> torch.Tensor:
        """
        按照给定的缩放因子对图像进行缩放

        :param images: 待缩放的图像张量，形状为 (n_batch, C, H, W)
        :param factor: 缩放因子
        :return: 缩放后的图像张量，形状为 (n_batch, C, H, W)
        """
        # 因为我们的图像像素值在 [-1, 1] 内，为了符合原文实现
        # 先转换为 [0, 1]，再进行缩放，然后再转换回 [-1, 1]
        destandardized_images = self._transforms_maker.destandardize(images)
        scaled_images = destandardized_images * factor
        scaled_images = torch.clip(scaled_images, 0.0, 1.0)
        scaled_images = self._transforms_maker.standardize(scaled_images)
        return scaled_images

    def _detect_batch(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对一个批次的图像使用 SCALE-UP 方法进行检测

        :param images: 待检测的一批图像张量，形状为 (n_batch, C, H, W)
        :labels: 这一批图像的真实标签，形状为 (n_batch, )
        :return: (模型对这批图像原本的预测概率, 每个缩放因子对应的预测概率, SPC 值)，
                 形状分别为 (B', 1), (B', num_factors) 和 (B', );
                 返回向量的长度并不一定是 n_batch, 会舍弃掉模型原本就分类错的样本
        """
        device = images.device
        labels = labels.to(device)
        n_batch = images.size(0)

        self._model.to(device)
        self._model.eval()

        num_factors = len(self._scaling_factors)

        # 获得没有放大的图像的预测结果
        with torch.no_grad():
            original_outputs = self._model(images)
            original_softmaxes = F.softmax(original_outputs, dim=-1)
            # shape (n_batch, )
            original_probs, original_preds = torch.max(original_softmaxes, dim=-1)
            correct_mask = original_preds == labels

        original_probs = original_probs.unsqueeze(1)  # shape (n_batch, 1)
        original_preds = original_preds.unsqueeze(1)  # shape (n_batch, 1)

        # 存储每个缩放因子的预测结果
        scaled_probs = torch.zeros((n_batch, num_factors), device=device)
        # 计算这一批每个样本的 SPC 值
        batch_spc = torch.zeros((n_batch,), dtype=torch.float32, device=device)

        for i, factor in enumerate(self._scaling_factors):
            # 因为我们的图像像素值在 [-1, 1] 内，为了符合原文实现
            # 先转换为 [0, 1]，再进行缩放，然后再转换回 [-1, 1]
            scaled_images = self._scale_image(images, factor)

            # 看看模型对放大后的图像的预测结果
            with torch.no_grad():
                scaled_outputs = self._model(scaled_images)
                scaled_softmaxes = F.softmax(scaled_outputs, dim=-1)
                scaled_preds = torch.argmax(
                    scaled_softmaxes, dim=-1
                )  # shape (n_batch, )

                batch_spc += (scaled_preds == original_preds.squeeze(1)).float()

            # 根据 Figure.2 进行实现，提取 original_preds 对应的概率
            probs_on_original_labels = torch.gather(
                scaled_softmaxes, dim=-1, index=original_preds
            )  # shape (n_batch, 1)

            scaled_probs[:, i] = probs_on_original_labels.squeeze(1)

        # 除以因子数量，得到 SPC
        batch_spc /= len(self._scaling_factors)  # shape (n_batch, )

        # 如果是 Data-limited 情况，则进行 Z-Score 标准化
        if self._data_limited:
            # 先根据预测标签获取对应的均值和标准差
            spc_means = []
            spc_stds = []
            for pred_label in original_preds.squeeze(1):
                pred_label = pred_label.item()
                stats = self._classwise_spc_stats.get(
                    pred_label, self._fallback_spc_stats
                )
                spc_means.append(stats["mean"])
                spc_stds.append(stats["std"])

            spc_means_tensor = torch.tensor(
                spc_means, dtype=torch.float32, device=device
            )
            spc_stds_tensor = torch.tensor(spc_stds, dtype=torch.float32, device=device)
            batch_spc = (batch_spc - spc_means_tensor) / (spc_stds_tensor + 1e-12)

        # 对于 Benign 样本，筛掉模型本来就分类错误的 (去除噪声)
        # 对于 Triggered 样本，筛掉模型本来就没分类成目标类的 (仅考虑成功攻击目标的样本)
        return (
            original_probs[correct_mask],
            scaled_probs[correct_mask],
            batch_spc[correct_mask],
        )

    def detect(self) -> dict:
        """
        使用 SCALE-UP 方法检测输入

        :return: 检测结果字典
        """
        tensorboard_log_id = f"scale_up_{self._test_id}"
        tensorboard_log_dir = os.path.join(TENSORBOARD_LOGS_PATH, tensorboard_log_id)
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        device = auto_select_device()

        print_section(f"SCALE-UP Defense: {self._test_id}")

        normal_all_results_list = []
        triggered_all_results_list = []

        normal_all_spc_list = []
        triggered_all_spc_list = []

        # 计算混淆矩阵条目
        tp = 0.0  # 预测为触发图像，实际为触发图像
        tn = 0.0  # 预测为正常图像，实际为正常图像
        fp = 0.0  # 预测为触发图像，实际为正常图像
        fn = 0.0  # 预测为正常图像，实际为触发图像

        with temp_seed(self._seed):
            for images, labels in tqdm(self._data_loader, desc="SCALE-UP Detection"):
                images: torch.Tensor = images.to(device)
                labels: torch.Tensor = labels.to(device)
                triggered_images = self._trigger_generator.apply_trigger(images)

                # 先在正常图像上检测
                original_probs, scaled_probs, original_spc = self._detect_batch(
                    images, labels
                )
                # 拼接起来
                normal_scale_probs = torch.cat(
                    [original_probs, scaled_probs], dim=1
                )  # shape (B?, num_factors + 1)

                # 预测为触发图像的样本
                original_spc_pos_preds = (original_spc >= self._T).sum().item()
                # 预测为正常图像的样本
                original_spc_neg_preds = (original_spc < self._T).sum().item()
                tn += original_spc_neg_preds
                fp += original_spc_pos_preds

                # 再在带有触发器的图像上检测
                target_labels = torch.full_like(labels, self._target_label)
                triggered_original_probs, triggered_scaled_probs, triggered_spc = (
                    self._detect_batch(triggered_images, target_labels)
                )
                triggered_scale_probs = torch.cat(
                    [triggered_original_probs, triggered_scaled_probs], dim=1
                )  # shape (B?, num_factors + 1)

                triggered_spc_pos_preds = (triggered_spc >= self._T).sum().item()
                triggered_spc_neg_preds = (triggered_spc < self._T).sum().item()

                tp += triggered_spc_pos_preds
                fn += triggered_spc_neg_preds

                normal_all_results_list.append(normal_scale_probs)
                triggered_all_results_list.append(triggered_scale_probs)
                normal_all_spc_list.append(original_spc)
                triggered_all_spc_list.append(triggered_spc)

        # 计算 TPR, FPR
        tpr_spc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_spc = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        normal_all_spc = torch.cat(normal_all_spc_list, dim=0)
        triggered_all_spc = torch.cat(triggered_all_spc_list, dim=0)

        # shape (B'', ), B'' 为所有正确分类的良性样本 + 所有成功攻击目标类的数量
        all_spc = torch.cat([normal_all_spc, triggered_all_spc], dim=0)
        # 这些样本中 SCALEUP 预测为触发器样本的预测结果
        triggered_preds = (all_spc >= self._T).float()
        # 这些样本中实际上是触发器样本的
        triggered_true = torch.cat(
            [
                torch.zeros_like(normal_all_spc),
                torch.ones_like(triggered_all_spc),
            ],
            dim=0,
        )
        # 获得 ROC 曲线和 AUC、F1-Score 分数
        fpr_arr, tpr_arr, _ = metrics.roc_curve(
            triggered_true.cpu().numpy(), all_spc.cpu().numpy()
        )
        auc = metrics.auc(fpr_arr, tpr_arr)
        f1_score = metrics.f1_score(
            triggered_true.cpu().numpy(), triggered_preds.cpu().numpy()
        )

        roc_vis_img = visualize_roc_curve(
            fpr=fpr_arr,
            tpr=tpr_arr,
            title=f"SCALE-UP ROC Curve (AUC={auc:.4f})",
        )

        tb_writer.add_image(
            "SCALE-UP ROC Curve",
            roc_vis_img,
            dataformats="HWC",
        )

        # 计算最终的平均结果
        normal_all_results = torch.cat(
            normal_all_results_list, dim=0
        )  # shape (N', num_factors + 1)
        triggered_all_results = torch.cat(
            triggered_all_results_list, dim=0
        )  # shape (M', num_factors + 1)

        normal_all_avg_probs = (
            torch.mean(normal_all_results, dim=0).cpu().tolist()
        )  # shape (num_factors + 1, )
        triggered_all_avg_probs = (
            torch.mean(triggered_all_results, dim=0).cpu().tolist()
        )  # shape (num_factors + 1, )

        x = [1] + self._scaling_factors  # 横轴

        vis_img = visualize_records(
            records={
                "Benign Samples": normal_all_avg_probs,
                "Triggered Samples": triggered_all_avg_probs,
            },
            records_x={"Benign Samples": x, "Triggered Samples": x},
            x_label="Scaling Factor",
            y_label="Average Prediction Confidence",
        )

        tb_writer.add_image(
            "SCALE-UP Detection Results on Probs",
            vis_img,
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

        # 保存结果
        result_save_path = os.path.join(
            self._save_dir, f"detection_results_{get_timestamp()}.json"
        )

        result = {
            "normal_avg_probs": normal_all_avg_probs,
            "triggered_avg_probs": triggered_all_avg_probs,
            "FPR": fpr_spc,
            "TPR": tpr_spc,
            "AUC": auc,
            "F1_Score": f1_score,
        }

        with open(result_save_path, "w") as f:
            json.dump(result, f, indent=4)

        tb_writer.add_text(
            "SCALE-UP Detection Results",
            json.dumps(result, indent=4),
        )

        tb_writer.close()

        return result
