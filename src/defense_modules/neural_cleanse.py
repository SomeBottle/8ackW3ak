"""
Neural Cleanse 检测方法实现

* Repo: https://github.com/lijiachun123/TrojAi
* Ref: Wang B, Yao Y, Shan S, et al. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks[C]//2019 IEEE symposium on security and privacy (SP). IEEE, 2019: 707-723.
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

from utils.data import DatasetWithInfo, DataLoaderDataIter, TransformedDataset
from data_augs import MakeSimpleTransforms
from utils.funcs import auto_select_device, temp_seed, get_timestamp,print_section

from defense_modules.abc import DefenseModule
from configs import TENSORBOARD_LOGS_PATH, CHECKPOINTS_SAVE_PATH

_ckpt_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "neural_cleanse")


class NeuralCleanse(DefenseModule):
    def __init__(
        self,
        test_id: str,
        model: nn.Module,
        dataset_info: DatasetWithInfo,
        *args,
        steps=1000,
        lr=0.1,
        init_cost=1e-3,
        num_mini_batches=31,
        batch_size=32,
        asr_threshold=0.99,
        patience=5,
        epsilon=1e-7,
        cost_factor=1.5,
        seed=42,
        **kwargs,
    ):
        """
        初始化 Neural Cleanse 检测模块

        :param test_id: 用来标记本次 NC 运行的测试 ID
        :param model: 待检测模型
        :param dataset_info: 数据集信息
        :param steps: 优化总轮数
        :param lr: 优化学习率
        :param init_cost: 触发器正则项的初始权重 (对应原文 Eq.3 中的 λ)
        :param num_mini_batches: 每轮中优化进行的批次数
        :param batch_size: 每个 mini-batch 的样本数量
        :param asr_threshold: 攻击成功率阈值
        :param patience: 动态调整 cost 时的耐心轮数，防止抖动
        :param epsilon: 用于数值稳定性的极小值
        :param cost_factor: 每次调整 cost 时的乘数因子
        """
        self._test_id = test_id
        self._model = copy.deepcopy(model)  # 不影响原模型
        self._transforms_maker = MakeSimpleTransforms(input_shape=dataset_info.shape)
        self._dataset_info = dataset_info
        self._steps = steps
        self._lr = lr
        self._init_cost = init_cost
        self._num_mini_batches = num_mini_batches
        self._batch_size = batch_size
        self._asr_threshold = asr_threshold
        self._patience = patience
        self._epsilon = epsilon
        self._cost_factor = cost_factor
        self._seed = seed
        self._save_dir = os.path.join(_ckpt_save_dir, test_id)
        self._data_loader = DataLoader(
            dataset=TransformedDataset(
                dataset_info.train_set,
                transform=self._transforms_maker.normalize_standardize,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        os.makedirs(self._save_dir, exist_ok=True)

    @classmethod
    def is_mitigation(cls) -> bool:
        return False

    def _reverse_trigger(
        self, target_label: int, tb_writer: SummaryWriter
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        为 target_label 逆向生成触发器

        :param target_label: 目标标签
        :param tb_writer: TensorBoard 写入器，用于记录
        :return: 最佳触发器 mask, 最佳触发器 pattern, 最佳触发器的 L1 范数, 最佳触发器的攻击成功率
        """
        device = auto_select_device()  # torch.device
        img_c, img_h, img_w = self._dataset_info.shape

        # 触发器 mask
        mask_tanh_tensor = torch.zeros(
            1, 1, img_h, img_w, device=device, requires_grad=True
        )
        # 触发器图案
        pattern_tanh_tensor = torch.zeros(
            1, img_c, img_h, img_w, device=device, requires_grad=True
        )

        optimizer = torch.optim.Adam(
            [mask_tanh_tensor, pattern_tanh_tensor], lr=self._lr
        )

        # lambda
        cost = self._init_cost
        # 调整 cost 前的轮数计数器，和 patience 一起用
        counter_cost_reset = 0
        counter_cost_increase = 0
        counter_cost_decrease = 0
        # cost 在一个调整周期中的调整标志，一个调整周期从 cost 被重置的时候开始算
        cost_increase_flag = False
        cost_decrease_flag = False

        # 上调和下调 cost 的因子
        cost_increase_factor = self._cost_factor
        cost_decrease_factor = self._cost_factor**1.5

        # 记录最佳的 mask 和 pattern
        best_mask: torch.Tensor | None = None
        best_pattern: torch.Tensor | None = None
        best_l1_norm = float("inf")
        best_asr = 0.0

        self._model.eval()
        self._model.requires_grad_(False)
        self._model.to(device)

        data_iter = DataLoaderDataIter(self._data_loader)

        with tqdm(
            total=self._steps, desc=f"Reverse Trigger for target {target_label}"
        ) as pbar:
            for step in range(self._steps):
                total_loss_ce = 0.0
                total_loss_reg = 0.0
                total_loss = 0.0
                correct = 0
                total = 0

                for _ in range(self._num_mini_batches):
                    images: torch.Tensor
                    images, _ = next(data_iter)
                    images = images.to(device)

                    # 从 tanh 空间获得 mask 和 pattern
                    mask = (
                        torch.tanh(mask_tanh_tensor) / (2.0 - self._epsilon) + 0.5
                    )  # [0,1]
                    mask = torch.clip(mask, 0.0, 1.0)
                    # 咱们图像本来就是 [-1, 1] 范围的，触发器扰动也在这个范围内
                    pattern = torch.tanh(pattern_tanh_tensor)  # [-1,1]

                    adv_images = images * (1 - mask) + pattern * mask
                    adv_images = torch.clip(adv_images, -1.0, 1.0)

                    outputs: torch.Tensor = self._model(adv_images)
                    target_labels = torch.full(
                        (images.size(0),),
                        target_label,
                        dtype=torch.long,
                        device=device,
                    )

                    loss_ce = F.cross_entropy(outputs, target_labels)

                    # L1 正则项
                    loss_res = torch.sum(torch.abs(mask))

                    loss = loss_ce + cost * loss_res

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss_ce += loss_ce.item()
                    total_loss_reg += loss_res.item()
                    total_loss += loss.item()

                    preds = outputs.argmax(dim=-1)
                    correct += (preds == target_labels).sum().item()
                    total += images.size(0)

                avg_loss_ce = total_loss_ce / self._num_mini_batches
                avg_loss_reg = total_loss_reg / self._num_mini_batches
                avg_loss = total_loss / self._num_mini_batches
                asr = correct / total

                pbar.set_postfix(
                    {
                        "avg_ce": avg_loss_ce,
                        "avg_l1": avg_loss_reg,
                        "asr": asr,
                        "cost": cost,
                    }
                )
                tb_writer.add_scalar(
                    f"Target_{target_label}/Avg_CE_Loss", avg_loss_ce, step + 1
                )
                tb_writer.add_scalar(
                    f"Target_{target_label}/Avg_L1_Loss", avg_loss_reg, step + 1
                )
                tb_writer.add_scalar(
                    f"Target_{target_label}/Avg_Total_Loss", avg_loss, step + 1
                )
                tb_writer.add_scalar(f"Target_{target_label}/ASR", asr, step + 1)
                tb_writer.add_scalar(f"Target_{target_label}/Cost", cost, step + 1)
                pbar.update(1)

                # 动态调整 cost
                if cost == 0 and asr >= self._asr_threshold:
                    # 正则项权重太小，ASR 高了，要准备重置
                    counter_cost_reset += 1
                    if counter_cost_reset >= self._patience:
                        cost = self._init_cost
                        counter_cost_reset = 0
                        # 重置 cost，调整周期重新开始
                        cost_increase_flag = False
                        cost_decrease_flag = False
                        counter_cost_increase = 0
                        counter_cost_decrease = 0
                else:
                    counter_cost_reset = 0

                if asr >= self._asr_threshold:
                    # ASR 达标，尝试提升 cost，增强正则
                    counter_cost_increase += 1
                    counter_cost_decrease = 0
                else:
                    # ASR 未达标，尝试降低 cost，减弱正则
                    counter_cost_decrease += 1
                    counter_cost_increase = 0

                # 超出耐心轮数，进行调整
                if counter_cost_increase >= self._patience:
                    counter_cost_increase = 0
                    cost *= cost_increase_factor
                    cost_increase_flag = True  # 标记这个调整周期中有过提升
                elif counter_cost_decrease >= self._patience:
                    counter_cost_decrease = 0
                    cost /= cost_decrease_factor
                    cost_decrease_flag = True  # 标记这个调整周期中有过降低

                # 记录最佳结果
                if asr >= self._asr_threshold and avg_loss_reg < best_l1_norm:
                    best_l1_norm = avg_loss_reg
                    best_mask = mask.detach().cpu()
                    best_pattern = pattern.detach().cpu()
                    best_asr = asr

                if (step + 1) % 100 == 0 or step == self._steps - 1:
                    pbar.write(f"Step {step+1}: Best L1 Norm={best_l1_norm:.4f}")
                    tb_writer.add_text(
                        f"Target_{target_label}/Best_L1_Norm",
                        f"Step {step+1}: Best L1 Norm={best_l1_norm:.4f}",
                        step + 1,
                    )

                # 早停
                if (
                    cost_increase_flag
                    and cost_decrease_flag
                    and asr >= self._asr_threshold
                ):
                    pbar.write("Early stopping triggered.")
                    tb_writer.add_text(
                        f"Target_{target_label}/Early_Stopping",
                        f"Early stopping at step {step+1}",
                        step + 1,
                    )
                    break

        if best_mask is None:
            # 没有达到阈值，直接用最后的 mask 和 pattern
            best_mask = mask.detach().cpu()
            best_pattern = pattern.detach().cpu()
            best_l1_norm = avg_loss_reg
            best_asr = asr

        return best_mask, best_pattern, best_l1_norm, best_asr

    def _mad_outlier_detection(
        self, l1_norms: list[float]
    ) -> tuple[torch.Tensor, int, float]:
        """
        Mean Absolute Deviation 异常值检测

        :param l1_norms: 各类别触发器的 L1 范数列表
        :return: 异常指数张量, 最小 L1 范数对应的标签, 最小 L1 范数对应的异常指数
        """
        l1_norms = torch.tensor(l1_norms, dtype=torch.float32)

        CONSISTENCY_CONSTANT = 1.4826  # 假设正态分布

        median = torch.median(l1_norms)

        abs_deviations = torch.abs(l1_norms - median)
        mad = torch.median(abs_deviations) * CONSISTENCY_CONSTANT

        # 计算每个点的异常指数
        mad = 1e-9 if mad == 0 else mad
        anomaly_indices = abs_deviations / mad

        # 最小 L1 NORM 下标
        min_l1_label = torch.argmin(l1_norms).item()
        anomaly_score_of_min_l1 = anomaly_indices[min_l1_label].item()

        return anomaly_indices, min_l1_label, anomaly_score_of_min_l1

    def detect(self):
        """
        执行 Neural Cleanse 检测

        :return: 包含检测结果的字典，{"anomaly_indices": List[float], "detected_label": int, "detected_anomaly_index": float}
        """
        tensorboard_log_id = f"neural_cleanse_{self._test_id}"
        tensorboard_log_dir = os.path.join(TENSORBOARD_LOGS_PATH, tensorboard_log_id)
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # 保存路径
        result_save_path = os.path.join(
            self._save_dir, f"nc_result_{get_timestamp()}.json"
        )
        ckpt_save_path = os.path.join(self._save_dir, f"nc_ckpt_{get_timestamp()}.pth")

        print_section(f"NC Defense: {self._test_id}")

        # 存储 target_label -> (mask, pattern, l1_norm, asr) 的映射
        label_to_info = {}

        if os.path.exists(ckpt_save_path):
            print(f"Found existing checkpoint at {ckpt_save_path}, loading it...")
            loaded_info = torch.load(ckpt_save_path, weights_only=False)
            label_to_info.update(loaded_info)
        else:
            with temp_seed(self._seed):
                # 对于每个标签都执行一次 NC
                for target_label in range(self._dataset_info.num_classes):
                    print(
                        f"Starting reverse trigger for target label {target_label}..."
                    )

                    best_mask, best_pattern, best_l1_norm, best_asr = (
                        self._reverse_trigger(target_label, tb_writer)
                    )

                    # 将结果存储到字典中
                    label_to_info[target_label] = (
                        best_mask,
                        best_pattern,
                        best_l1_norm,
                        best_asr,
                    )

            # 保存结果
            torch.save(label_to_info, ckpt_save_path)

        # 组装所有的 L1 范数，进行异常值检测
        l1_norms = [label_to_info[i][2] for i in range(self._dataset_info.num_classes)]

        anomaly_indices, detected_label, detected_anomaly_index = (
            self._mad_outlier_detection(l1_norms)
        )

        result = {
            "anomaly_indices": anomaly_indices.tolist(),
            "detected_label": detected_label,
            "detected_anomaly_index": detected_anomaly_index,
        }

        tb_writer.add_text(
            "Detection_Result",
            f"Detected Label: {detected_label}, and its Anomaly Index: {detected_anomaly_index:.4f}\nAnomaly Indices: {anomaly_indices.tolist()}",
        )

        # 保存检测结果
        with open(
            result_save_path,
            "w",
        ) as f:
            json.dump(result, f, indent=4)

        tb_writer.close()

        return result
