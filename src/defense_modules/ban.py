"""
BAN 检测方法实现

老实说官方实现有些地方真令人匪夷所思，这个实现中进行了一定的修复。

* Repo: https://github.com/xiaoyunxxy/ban
* Ref: Xu X, Liu Z, Koffas S, et al. BAN: detecting backdoors activated by adversarial neuron noise[J]. Advances in Neural Information Processing Systems, 2024, 37: 114348-114373.
"""

import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from defense_modules.ban_components.perb_batchnorm import NoisyBatchNorm2d

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

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

from models.abc import ModelBase
from modules.abc import TriggerGenerator
from defense_modules.abc import DefenseModule
from configs import TENSORBOARD_LOGS_PATH, CHECKPOINTS_SAVE_PATH

from utils.records import AverageLossRecorder

_ckpt_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "ban")


def _replace_bn_with_noisybn(module: nn.Module) -> nn.Module:
    """
    **原地**将 module 中的 BatchNorm2d 替换为 NoisyBatchNorm2d

    :param module: 待替换的模块
    :return: 替换后的模块，但注意其实模块已经原地发生改变
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            noisy_bn = NoisyBatchNorm2d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            # 复制权重和统计量状态
            if child.affine:
                with torch.no_grad():
                    noisy_bn.weight.copy_(child.weight)
                    noisy_bn.bias.copy_(child.bias)

            if child.track_running_stats:
                noisy_bn.running_mean.copy_(child.running_mean)
                noisy_bn.running_var.copy_(child.running_var)
                noisy_bn.num_batches_tracked.copy_(child.num_batches_tracked)

            setattr(module, name, noisy_bn)
        else:
            _replace_bn_with_noisybn(child)

    return module


def _reset_noisybn(module: nn.Module, rand_init: bool, eps: float) -> nn.Module:
    """
    **原地**重置 module 中的 NoisyBatchNorm2d 的噪声参数

    :param module: 待重置的模块
    :param rand_init: 是否随机初始化噪声参数
    :param eps: 噪声的 L_\infty 范数球半径
    :return: 重置后的模块，但注意其实模块已经原地发生改变
    """
    for child in module.children():
        if isinstance(child, NoisyBatchNorm2d):
            child.reset(rand_init=rand_init, eps=eps)
        else:
            _reset_noisybn(child, rand_init, eps)
    return module


class BAN(DefenseModule):
    def __init__(
        self,
        test_id: str,
        model: ModelBase,
        dataset_info: DatasetWithInfo,
        trigger_generator: TriggerGenerator,
        *args,
        data_portion: float = 0.05,
        eps: float = 0.3,
        steps: int = 1,
        mask_lambda: float = 0.25,
        mask_lr: float = 0.01,
        mask_steps: int = 20,
        suspicious_rate_threshold: float = 0.9,
        batch_size: int = 128,
        seed: int = 42,
        **kwargs,
    ):
        """
        初始化 BAN 检测模块

        :param test_id: 本次测试的唯一标识符
        :param model: 待检测的模型
        :param dataset_info: 数据集信息
        :param trigger_generator: 触发器生成器对象
        :param data_portion: 用于检测的验证数据比例，此处从验证集中划分。原文从训练集中划分，但也提到这部分训练数据没有用于模型训练。
        :param eps: 噪声扰动范围, L_\infty 范数球半径
        :param steps: 生成对抗噪声的迭代步数 (PGD 步数)
        :param mask_lambda: 掩码正则化参数
        :param mask_lr: 掩码优化学习率
        :param mask_steps: 掩码优化迭代步数
        :param suspicious_rate_threshold: 可疑目标类别比例阈值，高于该值则判断模型为后门目标
        :param batch_size: 检测时的数据批大小
        :param seed: 随机种子
        """
        self._test_id = test_id
        self._model = copy.deepcopy(model)  # 不影响原模型
        self._transforms_maker = MakeSimpleTransforms(input_shape=dataset_info.shape)
        self._dataset_info = dataset_info
        self._trigger_generator = trigger_generator
        self._data_portion = data_portion
        self._eps = eps
        self._steps = steps
        self._mask_lambda = mask_lambda
        self._mask_lr = mask_lr
        self._mask_steps = mask_steps
        self._suspicious_rate_threshold = suspicious_rate_threshold
        self._seed = seed
        self._save_dir = os.path.join(_ckpt_save_dir, test_id)

        os.makedirs(self._save_dir, exist_ok=True)

        # 划分出指定比例的数据用于检测
        remain_val_dataset, partial_val_dataset = balanced_split_into_two(
            dataset=dataset_info.val_set,
            latter_size_or_ratio=data_portion,
            random_state=seed,
        )
        # 第一部分数据，用于检测过程中特征掩码训练和对抗噪声生成
        self._partial_val_dataloader = DataLoader(
            dataset=TransformedDataset(
                partial_val_dataset,
                # 需要有数据增强
                transform=self._transforms_maker.train_transforms,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=auto_num_workers(),
            pin_memory=True,
        )
        # 第二部分数据，用于最终的损失和准确率评估
        self._remain_val_dataloader = DataLoader(
            dataset=TransformedDataset(
                remain_val_dataset,
                transform=self._transforms_maker.normalize_standardize,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=auto_num_workers(),
            pin_memory=True,
        )

    @classmethod
    def is_mitigation(cls) -> bool:
        return False

    def detect(self) -> dict:
        """
        使用 BAN 方法检测后门

        :return: 检测结果字典
        :raises RuntimeError: 如果模型中不包含 BatchNorm2d 层
        """

        tensorboard_log_id = f"ban_{self._test_id}"
        tensorboard_log_dir = os.path.join(TENSORBOARD_LOGS_PATH, tensorboard_log_id)
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        device = auto_select_device()

        print_section(f"BAN Defense: {self._test_id}")

        # 不影响原模型
        suspicious_model = copy.deepcopy(self._model)

        # 将模型中的 BatchNorm 替换为 NoisyBatchNorm
        _replace_bn_with_noisybn(suspicious_model)

        # 顺带把 NoisyBN 的参数也移动到 device
        suspicious_model.to(device)

        # 模型转训练模式，关闭计算图构建
        suspicious_model.train()  # 这里很重要，要持续更新 BN 层状态量，不然检测结果会有很大偏差
        suspicious_model.requires_grad_(False)

        # -------------------- STAGE 1: 特征解耦，训练掩码

        with temp_seed(self._seed):
            # 初始化特征掩码
            dummy_input = torch.rand(2, *self._dataset_info.shape, device=device)
            dummy_features, _ = suspicious_model(dummy_input, feat=True)
            dummy_feature = dummy_features[0]  # 取池化前特征

            mask = torch.empty_like(dummy_feature[0]).uniform_(0, 1)
            mask.requires_grad = True

            mask_optimizer = torch.optim.Adam([mask], lr=self._mask_lr)

            with tqdm(range(self._mask_steps), desc="Optimizing Mask") as pbar:
                for step in range(self._mask_steps):
                    recorder_loss_positive = AverageLossRecorder()
                    recorder_loss_negative = AverageLossRecorder()
                    for images, labels in self._partial_val_dataloader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        features = suspicious_model(images, feat=True)[0]
                        features_l_2 = features[0]  # 池化前特征

                        pred_positive = suspicious_model.feature_to_output(
                            features_l_2 * mask, feat_level=2
                        )
                        pred_negative = suspicious_model.feature_to_output(
                            features_l_2 * (1 - mask), feat_level=2
                        )

                        loss_positive = F.cross_entropy(pred_positive, labels)
                        loss_negative = F.cross_entropy(pred_negative, labels)
                        # 注意，正则项这里官方代码实现有点怪，这里进行了修复
                        # https://github.com/xiaoyunxxy/ban/blob/8a0f91890ae0e4afe604b347f79f450500d4d09d/ban_detection.py#L205
                        loss_mask_reg = torch.mean(torch.abs(mask))

                        # 希望掩码能最大化正负特征的区分度 (剥离良性特征和恶意特征)，同时保持掩码稀疏
                        loss = (
                            loss_positive
                            - loss_negative
                            + self._mask_lambda * loss_mask_reg
                        )

                        mask_optimizer.zero_grad()
                        loss.backward()
                        mask_optimizer.step()

                        with torch.no_grad():
                            mask.clip_(0, 1)

                        recorder_loss_positive.batch_update(
                            loss_positive, images.size(0)
                        )
                        recorder_loss_negative.batch_update(
                            loss_negative, images.size(0)
                        )

                    pbar.set_postfix(
                        {
                            "loss_pos": f"{recorder_loss_positive.avg_loss:.4f}",
                            "loss_neg": f"{recorder_loss_negative.avg_loss:.4f}",
                            "l1_norm": f"{loss_mask_reg.item():.4f}",
                        }
                    )
                    tb_writer.add_scalar(
                        "mask_optimization/loss_positive",
                        recorder_loss_positive.avg_loss,
                        step,
                    )
                    tb_writer.add_scalar(
                        "mask_optimization/loss_negative",
                        recorder_loss_negative.avg_loss,
                        step,
                    )
                    tb_writer.add_scalar(
                        "mask_optimization/l1_norm",
                        loss_mask_reg.item(),
                        step,
                    )
                    pbar.update(1)

            mask = mask.detach()

        # -------------------- STAGE 2: 生成对抗噪声

        with temp_seed(self._seed):
            # 先重置 NoisyBN 的噪声参数
            _reset_noisybn(suspicious_model, rand_init=True, eps=self._eps)

            # 启动噪声层的计算图构建
            noise_params = []
            for module in suspicious_model.modules():
                if isinstance(module, NoisyBatchNorm2d):
                    # 启动噪声
                    module.include_noise()
                    module.neuron_noise.requires_grad = True
                    module.neuron_noise_bias.requires_grad = True
                    noise_params.extend([module.neuron_noise, module.neuron_noise_bias])

            if len(noise_params) == 0:
                raise RuntimeError(
                    "No NoisyBatchNorm2d layers found in the model! This may be due to the model not containing any BatchNorm2d layers originally."
                )

            noise_lr = self._eps / self._steps  # PGD 学习率设定
            noise_optimizer = torch.optim.SGD(noise_params, lr=noise_lr)
            with tqdm(range(self._steps), desc="Generating Adversarial Noise") as pbar:
                for step in range(self._steps):
                    recorder_loss = AverageLossRecorder()
                    for images, labels in self._partial_val_dataloader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        outputs = suspicious_model(images)
                        # 最大化分类损失，希望噪声能尽量干扰模型预测
                        loss = -F.cross_entropy(outputs, labels)

                        noise_optimizer.zero_grad()
                        loss.backward()

                        noise_params = [
                            param
                            for name, param in suspicious_model.named_parameters()
                            if "neuron_noise" in name
                        ]
                        # 取梯度符号
                        for param in noise_params:
                            param.grad.sign_()

                        noise_optimizer.step()

                        # 噪声投影到 L_\infty 范数球，遵循原文实现 (Eq.4)
                        with torch.no_grad():
                            for param in noise_params:
                                param.clip_(-self._eps, self._eps)

                        recorder_loss.batch_update(loss, images.size(0))

                    pbar.set_postfix(
                        {
                            "loss": f"{recorder_loss.avg_loss:.4f}",
                        }
                    )
                    pbar.update(1)
                    tb_writer.add_scalar(
                        "noise_generation/loss",
                        recorder_loss.avg_loss,
                        step,
                    )

        # Stage 3: 在噪声 + 掩码的情况下评估模型
        # 噪声在上面已经启用
        suspicious_model.eval()
        suspicious_model.requires_grad_(False)

        # 负掩码，根据上面的优化，负掩码对应可能的恶意特征
        negative_mask = 1 - mask

        # 分类为每个类别的样本数
        class_sample_counts = torch.zeros(
            self._dataset_info.num_classes, dtype=torch.long, device=device
        )
        num_samples = 0

        for images, _ in tqdm(
            self._remain_val_dataloader, desc="Evaluating with Noise + Mask"
        ):
            images: torch.Tensor = images.to(device)

            features = suspicious_model(images, feat=True)[0]
            features_l_2 = features[0]  # 池化前特征

            masked_features = features_l_2 * negative_mask

            outputs = suspicious_model.feature_to_output(masked_features, feat_level=2)
            preds = torch.argmax(outputs, dim=-1)  # shape: (B,)
            counts = torch.bincount(preds, minlength=self._dataset_info.num_classes)
            class_sample_counts += counts
            num_samples += images.size(0)

        # 在噪声 + 负掩码下，良性模型表现会很混乱，但是后门模型会集中预测到后门目标类别
        # BAN 依据这个直觉来找后门目标类别
        suspicious_target = torch.argmax(class_sample_counts).item()
        # 看看有多少比例的样本被分类为该类别
        suspicious_rate = class_sample_counts[suspicious_target].item() / num_samples
        is_backdoored = suspicious_rate >= self._suspicious_rate_threshold

        result = {
            "suspicious_target": suspicious_target,
            "suspicious_rate": suspicious_rate,
            "is_backdoored": is_backdoored,
            "class_sample_counts": class_sample_counts.cpu().tolist(),
        }

        tb_writer.add_text(
            "Detection_Result",
            json.dumps(result, indent=4),
        )

        # 检测结果保存路径
        result_save_path = os.path.join(
            self._save_dir, f"ban_detection_result_{get_timestamp()}.json"
        )

        # 保存检测结果
        with open(result_save_path, "w") as f:
            json.dump(result, f, indent=4)

        tb_writer.close()

        return result
