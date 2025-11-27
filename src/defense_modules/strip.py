"""
STRIP 检测方式实现

检测实验中严格来说有三部分互不相交的数据:

1. 攻击者用来构造恶意输入的数据 D_A，待检测
2. 防御者用来评估良性样本的熵分布的数据 D_V (防御者掌握)
3. 用于叠加在输入上造成扰动的样本集 D_P (防御者掌握)

* 验证时 D_A 和 D_V 的数量通常相同

在此实现中，D_A 从训练集中划分 n_test 个样本，D_V 从验证集中划分 n_test 个样本，D_P 则使用验证集的剩余样本。
> @SomeBottle 2025.11.25

* Official Repo: https://github.com/garrisongys/STRIP
* Ref: Gao Y, Xu C, Wang D, et al. Strip: A defence against trojan attacks on deep neural networks[C]//Proceedings of the 35th annual computer security applications conference. 2019: 113-125.
"""

import os
import copy
import json
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from sklearn import metrics

from utils.data import DatasetWithInfo, TransformedDataset, DataLoaderDataIter
from utils.data_funcs import balanced_split_into_two
from data_augs import MakeSimpleTransforms
from utils.funcs import (
    auto_select_device,
    temp_seed,
    get_timestamp,
    print_section,
    shannon_entropy,
    auto_num_workers,
)

from modules.abc import TriggerGenerator
from defense_modules.abc import DefenseModule
from configs import TENSORBOARD_LOGS_PATH, CHECKPOINTS_SAVE_PATH
from typing import Iterator
from utils.visualization import visualize_roc_curve, visualize_entropy_dist_histogram

_ckpt_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "strip")


class STRIP(DefenseModule):

    def __init__(
        self,
        test_id: str,
        model: nn.Module,
        dataset_info: DatasetWithInfo,
        trigger_generator: TriggerGenerator,
        *args,
        n_test: int = 2000,
        perturb_samples_per_input: int = 20,
        desired_fpr: float = 0.01,
        seed: int = 42,
        **kwargs,
    ):
        """
        初始化 STRIP 防御模块

        :param test_id: 测试 ID
        :param model: 待检测的模型
        :param dataset_info: 数据集信息
        :param trigger_generator: 触发器生成器
        :param n_test: 用来作为输入的测试样本数量 (同时会从训练集和验证集中进行抽取)
        :param perturb_samples_per_input: 用来叠加扰动的样本数量
        :param desired_fpr: 用来设定熵阈值的假阳性率参数，默认 1%
        :param seed: 随机种子
        """
        self._test_id = test_id
        self._model = copy.deepcopy(model)  # 不影响原模型
        self._transforms_maker = MakeSimpleTransforms(input_shape=dataset_info.shape)
        self._dataset_info = dataset_info
        self._trigger_generator = trigger_generator
        self._n_test = n_test
        self._perturb_samples_per_input = perturb_samples_per_input
        self._seed = seed
        self._desired_fpr = desired_fpr
        self._save_dir = os.path.join(_ckpt_save_dir, test_id)
        # 从训练集中抽取 D_A
        _, attacker_dataset = balanced_split_into_two(
            dataset=dataset_info.train_set,
            latter_size_or_ratio=n_test,
            random_state=seed,
        )
        # 从验证集中抽取 D_V 和 D_P
        perturb_dataset, defender_dataset = balanced_split_into_two(
            dataset=dataset_info.val_set,
            latter_size_or_ratio=n_test,
            random_state=seed,
        )

        self._attacker_loader = DataLoader(
            dataset=TransformedDataset(
                attacker_dataset,
                transform=self._transforms_maker.normalize_standardize,
            ),
            batch_size=32,
            shuffle=False,
            num_workers=auto_num_workers(),
        )
        self._defender_loader = DataLoader(
            dataset=TransformedDataset(
                defender_dataset,
                transform=self._transforms_maker.normalize_standardize,
            ),
            batch_size=32,
            shuffle=False,
            num_workers=auto_num_workers(),
        )
        # 用于叠加的样本
        self._perturb_loader = DataLoader(
            dataset=TransformedDataset(
                perturb_dataset,
                transform=self._transforms_maker.normalize_standardize,
            ),
            batch_size=perturb_samples_per_input,  # 每次取出用于叠加的样本数量
            shuffle=True,
            num_workers=auto_num_workers(),
        )

        os.makedirs(self._save_dir, exist_ok=True)

    @classmethod
    def is_mitigation(cls) -> bool:
        return False

    def _calc_batch_entropies(
        self, images: torch.Tensor, perturb_data_iter: Iterator
    ) -> torch.Tensor:
        """
        对一个批次的图像叠加扰动并计算模型对每个图像 (加了扰动) 的预测平均熵值

        :param images: 输入图像张量，形状为 (B, C, H, W)
        :param perturb_data_iter: 用于叠加扰动的样本数据迭代器
        :return: 每个输入图像在扰动下的平均熵值张量，形状为 (B,)
        """
        device = images.device
        n_batch = images.size(0)

        self._model.to(device)
        self._model.eval()

        # 这一批复制 perturb_samples_per_input 次，
        # 即每个样本创建 perturb_samples_per_input 个副本用于叠加扰动
        # 假设前两个紧挨的样本是 A, B，复制 3 次后应该是 A, A, A, B, B, B 这样排列
        image_copies = images.repeat_interleave(
            self._perturb_samples_per_input, dim=0
        )  # shape: (B * P, C, H, W)

        # 相同样本的副本为一组, A, A, A, B, B, B -> (A, A, A), (B, B, B)
        # 每一组的 shape 为 (P, C, H, W)
        image_groups = list(image_copies.split(self._perturb_samples_per_input, dim=0))

        # 对每个样本的多个副本进行扰动叠加
        for i, group in enumerate(image_groups):
            # 获取用于叠加的扰动样本
            perturb_images, _ = next(perturb_data_iter)
            perturb_images: torch.Tensor = perturb_images.to(device)
            # 此处遵循原仓库实现，在 [0, 1] 图像值范围下直接叠加两个图像

            # 先转换回 [0, 1] 范围
            destd_group = self._transforms_maker.destandardize(group)
            destd_perturb = self._transforms_maker.destandardize(perturb_images)

            perturbed_group = destd_group + destd_perturb
            # 转换回 [-1, 1] 范围
            perturbed_group: torch.Tensor = self._transforms_maker.standardize(
                perturbed_group
            )

            image_groups[i] = perturbed_group

        # 将扰动后的各组样本重新拼接回一个批次
        perturbed_inputs = torch.cat(image_groups, dim=0)  # shape: (B * P, C, H, W)

        with torch.no_grad():
            outputs = self._model(perturbed_inputs)  # shape: (B * P, num_classes)
            probs = F.softmax(outputs, dim=-1)
            # 计算模型为每个样本的预测的熵
            entropies = shannon_entropy(probs)  # shape: (B * P,)

        # 分组求平均
        grouped_entropies = entropies.view(
            n_batch, self._perturb_samples_per_input
        )  # shape: (B, P)
        avg_entropies = grouped_entropies.mean(dim=1)  # shape: (B,)

        return avg_entropies

    def detect(self) -> dict:
        """
        使用 STRIP 方法进行检测

        :return: 检测结果字典
        """
        tensorboard_log_id = f"strip_{self._test_id}"
        tensorboard_log_dir = os.path.join(TENSORBOARD_LOGS_PATH, tensorboard_log_id)
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        device = auto_select_device()

        print_section(f"STRIP Defense: {self._test_id}")

        with temp_seed(self._seed):
            # 无尽随机迭代器，在这里初始化，受种子控制
            perturb_data_iter = DataLoaderDataIter(self._perturb_loader)
            # -------------- 测试攻击者样本(良性) --------------
            benign_entropies = []
            for images, _ in tqdm(self._defender_loader, desc="Testing benign samples"):
                images: torch.Tensor = images.to(device)
                # 原文这里是串行逐个处理样本的，速度会很慢，这里改为并行处理
                batch_entropies = self._calc_batch_entropies(
                    images, perturb_data_iter
                )  # shape: (B,)
                benign_entropies.append(batch_entropies)
            benign_entropies = torch.cat(benign_entropies, dim=0)  # shape: (n_test, )
            # -------------- 测试攻击者样本(加了触发器的) --------------
            triggered_entropies = []
            for images, _ in tqdm(
                self._attacker_loader, desc="Testing attacker samples"
            ):
                images: torch.Tensor = images.to(device)
                triggered_images = self._trigger_generator.apply_trigger(images)
                triggered_batch_entropies = self._calc_batch_entropies(
                    triggered_images, perturb_data_iter
                )  # shape: (B,)
                triggered_entropies.append(triggered_batch_entropies)
            triggered_entropies = torch.cat(
                triggered_entropies, dim=0
            )  # shape: (n_test, )

            # -------------- 计算阈值 --------------
            benign_mean = torch.mean(benign_entropies)
            benign_std = torch.std(benign_entropies)

            # 根据官方实现，假设良性样本熵值服从高斯分布
            # 找到熵值最低的 1% 的良性样本对应的分位点熵值作为阈值 (左尾)
            # 熵值低于该阈值的样本被判定为恶意样本
            benign_threshold = dist.Normal(benign_mean, benign_std).icdf(
                torch.tensor(self._desired_fpr, device=device)
            )

            # 良性和触发样本的检测结果
            benign_preds: torch.Tensor = (benign_entropies <= benign_threshold).long()
            triggered_preds: torch.Tensor = (
                triggered_entropies <= benign_threshold
            ).long()

            # 检测真值
            benign_truths = torch.zeros_like(benign_preds)
            triggered_truths = torch.ones_like(triggered_preds)

            # -------------- 检测结果 --------------

            # 对 sklearn 来说分数越高越 positive，因此取负熵值
            benign_scores = -benign_entropies
            triggered_scores = -triggered_entropies

            all_scores = torch.cat([benign_scores, triggered_scores], dim=0)
            all_truths = torch.cat([benign_truths, triggered_truths], dim=0)
            all_preds = torch.cat([benign_preds, triggered_preds], dim=0)

            # ROC & AUC
            fpr_arr, tpr_arr, _ = metrics.roc_curve(
                all_truths.cpu().numpy(), all_scores.cpu().numpy()
            )
            auc = metrics.auc(fpr_arr, tpr_arr)

            roc_vis_img = visualize_roc_curve(
                fpr=fpr_arr,
                tpr=tpr_arr,
                title=f"STRIP ROC Curve (AUC={auc:.4f})",
            )

            tb_writer.add_image(
                "STRIP ROC Curve",
                roc_vis_img,
                dataformats="HWC",
            )

            # TPR & FPR
            tn, fp, fn, tp = metrics.confusion_matrix(
                all_truths.cpu().numpy(), all_preds.cpu().numpy()
            ).ravel()

            tpr_scalar = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr_scalar = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            # Accuracy & F1-Score

            accuracy = (tp + tn) / (tp + tn + fp + fn)

            f1_score = metrics.f1_score(
                all_truths.cpu().numpy(), all_preds.cpu().numpy()
            )

            # 熵分布直方图
            entropy_hist_img = visualize_entropy_dist_histogram(
                [
                    (benign_entropies, "Benign Samples"),
                    (triggered_entropies, "Triggered Samples"),
                ],
                title="STRIP Entropy Distribution Histogram",
                x_label="Entropy",
                y_label="Probability",
            )
            tb_writer.add_image(
                "STRIP Entropy Distribution Histogram",
                entropy_hist_img,
                dataformats="HWC",
            )

        result_save_path = os.path.join(
            self._save_dir, f"strip_detection_result_{get_timestamp()}.json"
        )

        result = {
            "TPR": tpr_scalar,
            "FPR": fpr_scalar,
            "Accuracy": accuracy,
            "AUC": auc,
            "F1-Score": f1_score,
        }

        with open(result_save_path, "w") as f:
            json.dump(result, f, indent=4)

        tb_writer.add_text(
            "STRIP Detection Results",
            json.dumps(result, indent=4),
        )

        tb_writer.close()

        return result
