"""
复现论文 Taught Well Learned Ill 的代码，这个实现有爆显存问题。

- 主模块

* Ref: Chen Y, Li B, Yuan Y, et al. Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack[J]. arXiv preprint arXiv:2509.23871, 2025.

孩子们，这有点不好复现了。SCAR 代码是 Coming soon 的，发邮件是不回的，what can I say!
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from reprod_modules.abc import SCARBase
from itertools import islice
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Type
from modules.abc import TriggerGenerator
from data_augs.abc import MakeTransforms
from utils.data import DatasetWithInfo, TransformedDataset
from configs import (
    REPROD_CHECKPOINTS_SAVE_PATH,
    TENSORBOARD_LOGS_PATH,
)
from utils.funcs import (
    auto_select_device,
    auto_num_workers,
    temp_seed,
    get_curr_random_states,
    load_random_states,
    get_base_exp_info,
    test_benign_accuracy,
    test_attack_success_rate,
    print_section,
    json_serialize_helper,
    get_grad_norm,
)
from utils.visualization import visualize_images

from utils.records import AverageLossRecorder
from utils.data import DatasetWithInfo, TransformedDataset

_default_save_dir = os.path.join(REPROD_CHECKPOINTS_SAVE_PATH, "scar_main")
_default_num_workers = auto_num_workers()


class SCAR(SCARBase):
    """
    复现论文 Taught Well Learned Ill - SCAR (Algorithm 1) 的代码主模块
    """

    def __init__(
        self,
        trigger_preoptimizer: TriggerGenerator,
        teacher_model_class: Type[nn.Module],
        surrogate_model_class: Type[nn.Module],
        outer_epochs: int,
        inner_updates: int,
        fixed_point_iters: int,
        outer_grad_batches: int,
        teacher_lr: float,
        surrogate_lr: float,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        temperature: float,
        target_label: int,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        batch_size: int,
        exp_id: str,
        exp_desc: str = "",
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 1,
        num_workers: int = _default_num_workers,
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        初始化 SCAR 主模块

        :param trigger_preoptimizer: 触发器预优化模块
        :param teacher_model_class: 教师模型类
        :param surrogate_model_class: 影子学生模型类
        :param outer_epochs: 外层优化轮数
        :param inner_updates: 内层优化步数 (T)
        :param fixed_point_iters: 固定点迭代次数 (K)
        :param outer_grad_batches: 估计一次外层梯度需要的批次数 (M)
        :param teacher_lr: 教师模型学习率
        :param surrogate_lr: 影子学生模型学习率
        :param alpha: 教师忽略触发器的损失的权重
        :param beta: 影子学生从正常样本学习的损失的权重
        :param gamma: 影子学生学习后门的损失的权重
        :param delta: 模拟蒸馏学生时的蒸馏损失的权重
        :param temperature: 蒸馏温度
        :param target_label: 目标标签
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param batch_size: 批大小
        :param exp_id: 实验 ID
        :param exp_desc: 实验描述
        :param make_test_per_epochs: 每隔多少轮测试一次
        :param save_ckpts_per_epochs: 每隔多少轮保存一次 Checkpoints
        :param num_workers: Dataloader 的 num_workers
        :param seed: 随机种子
        :param save_dir: 模型保存路径
        """
        super().__init__()
        model_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(model_save_dir, exist_ok=True)

        self._model_save_path = os.path.join(model_save_dir, "scar_teacher.pth")
        self.set_exp_save_dir(model_save_dir)
        # -------------------------------- 数据增强
        data_transform = data_transform_class(input_shape=dataset_info.shape)

        # -------------------------------- 数据集转换
        train_tensor_set = TransformedDataset(
            dataset=dataset_info.train_set, transform=data_transform.train_transforms
        )
        val_tensor_set = TransformedDataset(
            dataset=dataset_info.val_set, transform=data_transform.val_transforms
        )
        self._trigger_preoptimizer = trigger_preoptimizer
        self._teacher_model_class = teacher_model_class
        self._surrogate_model_class = surrogate_model_class
        self._outer_epochs = outer_epochs
        self._inner_updates = inner_updates
        self._fixed_point_iters = fixed_point_iters
        self._outer_grad_batches = outer_grad_batches
        self._teacher_lr = teacher_lr
        self._surrogate_lr = surrogate_lr
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._temperature = temperature
        self._target_label = target_label
        self._dataset_info = dataset_info
        self._data_transform_class = data_transform_class
        self._train_tensor_set = train_tensor_set
        self._val_tensor_set = val_tensor_set
        self._batch_size = batch_size
        self._exp_id = exp_id
        self._exp_desc = exp_desc
        self._make_test_per_epochs = make_test_per_epochs
        self._save_ckpts_per_epochs = save_ckpts_per_epochs
        self._num_workers = num_workers
        self._seed = seed
        self._train_loader = DataLoader(
            dataset=train_tensor_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self._val_loader = DataLoader(
            dataset=val_tensor_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self._scar_teacher = None

    @property
    def exp_id(self) -> str:
        return self._exp_id

    def get_scar_teacher(self) -> nn.Module:
        """
        获取训练好的 SCAR 教师模型

        :return: SCAR 教师模型
        """

        if self._scar_teacher is not None:
            return self._scar_teacher

        with temp_seed(self._seed):
            device = auto_select_device()
            teacher_model = self._teacher_model_class(
                num_classes=self._dataset_info.num_classes
            )
            teacher_model.to(device)

            if os.path.exists(self._model_save_path):
                # 如果模型已经存在，直接载入返回
                teacher_model.load_state_dict(
                    torch.load(self._model_save_path, map_location=device)
                )
                self._scar_teacher = teacher_model
                return self._scar_teacher

            # 先触发触发器生成
            self._trigger_preoptimizer.generate()

            print_section(f"SCAR Training: {self.exp_id:.20s}")

            # 开始时间
            time_start = time.time()
            # 进行验证时耗费的时间
            time_consumed_by_val = 0.0

            # TensorBoard 记录器
            tensorboard_log_id = f"scar_main_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH, tensorboard_log_id
            )
            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # 模型训练
            teacher_model.requires_grad_(True)

            teacher_optimizer = torch.optim.Adam(
                teacher_model.parameters(), lr=self._teacher_lr
            )
            teacher_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                teacher_optimizer, T_max=self._outer_epochs
            )

            start_epoch = 0

            # 如果有 Checkpoints 则载入
            if self.has_checkpoints():
                ckpts = self.load_checkpoints()
                teacher_ckpt = ckpts["teacher_model"]
                teacher_optimizer_ckpt = ckpts["teacher_optimizer"]
                teacher_lr_scheduler_ckpt = ckpts["teacher_lr_scheduler"]
                # 载入之前的状态
                teacher_model.load_state_dict(teacher_ckpt)
                teacher_optimizer.load_state_dict(teacher_optimizer_ckpt)
                teacher_lr_scheduler.load_state_dict(teacher_lr_scheduler_ckpt)
                # 载入时间, 外部轮数信息
                time_start = ckpts["time_start"]  # 训练最初开始时间，用于计算总耗时
                time_consumed_by_val = ckpts["time_consumed_by_val"]  # 训练中验证耗时
                ckpt_save_time = ckpts[
                    "time_save"
                ]  # 上次 ckpt 保存时间，用于从总耗时中减去
                time_consumed_by_val += (
                    time.time() - ckpt_save_time
                )  # 总耗时中排除掉中间这一段时间
                start_epoch = ckpts["current_epoch"]
                # 载入随机状态
                prev_random_states = ckpts["random_states"]
                load_random_states(prev_random_states)
                print(f"Loaded checkpoints from epoch {start_epoch}.")

            with tqdm(
                initial=start_epoch,
                total=self._outer_epochs,
                desc=f"SCAR ({self._exp_id})",
            ) as pbar:
                for outer_epoch in range(start_epoch, self._outer_epochs):
                    # Algorithm 1, line 2, 重初始化 ω_0 (代理学生模型权重)
                    surrogate_model = self._surrogate_model_class(
                        num_classes=self._dataset_info.num_classes
                    )
                    surrogate_model.to(device)
                    # 原文未指定代理模型优化器
                    # 因为后续推导基于标准梯度下降，这里直接采用 SGD (0 动量)
                    surrogate_optimizer = torch.optim.SGD(
                        surrogate_model.parameters(),
                        lr=self._surrogate_lr,
                        momentum=0.0,
                    )

                    # --------------------- Algorithm 1, line 3, 内层循环
                    # 先把学生优化 T 步，接近不动点 ω*
                    teacher_model.eval()
                    for inner_step in range(self._inner_updates):
                        surrogate_model.train()
                        recorder_loss_inner = AverageLossRecorder()
                        recorder_loss_outer_teacher_benign = AverageLossRecorder()
                        recorder_loss_outer_teacher_stealthy = AverageLossRecorder()
                        recorder_loss_surrogate_benign = AverageLossRecorder()
                        recorder_loss_surrogate_attack = AverageLossRecorder()
                        for images, labels in self._train_loader:
                            images: torch.Tensor = images.to(device)
                            labels: torch.Tensor = labels.to(device)

                            student_logits: torch.Tensor = surrogate_model(images)
                            with torch.no_grad():
                                teacher_logits: torch.Tensor = teacher_model(images)
                                soft_teacher_probs = F.softmax(
                                    teacher_logits / self._temperature, dim=-1
                                )

                            loss_ce = F.cross_entropy(student_logits, labels)
                            loss_kd = F.kl_div(
                                F.log_softmax(
                                    student_logits / self._temperature, dim=-1
                                ),
                                soft_teacher_probs,
                                reduction="batchmean",
                            ) * (self._temperature**2)

                            # Algorithm 1, line 4, Eq.(2) 的 L_in
                            loss_inner = loss_ce + self._delta * loss_kd

                            surrogate_optimizer.zero_grad()
                            loss_inner.backward()
                            surrogate_optimizer.step()
                            recorder_loss_inner.batch_update(loss_inner, images.size(0))

                        pbar.set_postfix(
                            {
                                "Inner Step": f"{inner_step+1}/{self._inner_updates}",
                                "Loss_in": f"{recorder_loss_inner.avg_loss:.4f}",
                            }
                        )
                        tb_writer.add_scalar(
                            f"Outer_{outer_epoch+1:04d}/Inner_Loss",
                            recorder_loss_inner.avg_loss,
                            inner_step + 1,
                        )

                    # Algorithm 1, line 6, 完成内层循环后，准备数据子集用于外层梯度估计
                    subset_data = list(
                        islice(iter(self._train_loader), self._outer_grad_batches)
                    )

                    # --------------------- 开始估计外层梯度
                    # 即估计:  ∇_λ L_out(ω*(λ), λ)
                    #             = (∂ω*/∂λ)^T g_ω + g_λ     (Eq. 5)
                    teacher_model.train()  # 教师转换为训练模式，保证前向传播准确
                    surrogate_model.train()  # 代理模型继续模拟一个待被更新的对象
                    teacher_optimizer.zero_grad()

                    # Algorithm 1, line 7, 第二项, 计算对于教师的直接梯度 g_λ
                    # g_λ = L_CE(F_t(x;λ), y) + α * L_CE(F_t(G(x);λ), y)
                    mean_loss_direct = torch.tensor(
                        0.0, device=device
                    )  # 记录直接梯度损失
                    for images, labels in subset_data:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        poisoned_images = self._trigger_preoptimizer.apply_trigger(
                            images
                        )

                        # 良性部分 L_CE(F_t(x;λ), y)
                        teacher_benign_logits = teacher_model(images)
                        loss_teacher_benign = F.cross_entropy(
                            teacher_benign_logits, labels
                        )
                        recorder_loss_outer_teacher_benign.batch_update(
                            loss_teacher_benign, images.size(0)
                        )

                        # 后门抑制部分 L_CE(F_t(G(x);λ), y)
                        teacher_poisoned_logits = teacher_model(poisoned_images)
                        loss_teacher_poisoned = F.cross_entropy(
                            teacher_poisoned_logits, labels
                        )
                        recorder_loss_outer_teacher_stealthy.batch_update(
                            loss_teacher_poisoned, images.size(0)
                        )

                        loss_direct = (
                            loss_teacher_benign + self._alpha * loss_teacher_poisoned
                        )
                        # 累积梯度 g_λ ← ∇_λ L_direct
                        (loss_direct / self._outer_grad_batches).backward()
                        mean_loss_direct += (
                            loss_direct.item() / self._outer_grad_batches
                        )

                    # 记录 g_λ 范数
                    g_lambda_norm, time_consumed = get_grad_norm(
                        teacher_model.parameters()
                    )
                    time_consumed_by_val += time_consumed

                    # Algorithm 1, line 7, 第一项, 计算对于学生的直接梯度 g_ω，对于教师有间接梯度
                    # g_ω = β * L_CE(F_s(x;ω(λ)), y) + γ * L_CE(F_s(G(x);ω(λ)), y_t)
                    mean_loss_surrogate = torch.tensor(
                        0.0, device=device
                    )  # 记录代理模型损失
                    # g_omega 梯度累加
                    g_omega_sum = [
                        torch.zeros_like(p) for p in surrogate_model.parameters()
                    ]
                    for images, labels in subset_data:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        poisoned_images = self._trigger_preoptimizer.apply_trigger(
                            images
                        )
                        target_labels = torch.full_like(labels, self._target_label)

                        # 良性部分 L_CE(F_s(x;ω(λ)), y)
                        surrogate_benign_logits = surrogate_model(images)
                        loss_surrogate_benign = F.cross_entropy(
                            surrogate_benign_logits, labels
                        )
                        recorder_loss_surrogate_benign.batch_update(
                            loss_surrogate_benign, images.size(0)
                        )

                        # 后门注入部分 L_CE(F_s(G(x);ω(λ)), y_t)
                        surrogate_poisoned_logits = surrogate_model(poisoned_images)
                        loss_surrogate_poisoned = F.cross_entropy(
                            surrogate_poisoned_logits, target_labels
                        )
                        recorder_loss_surrogate_attack.batch_update(
                            loss_surrogate_poisoned, images.size(0)
                        )

                        loss_surrogate_batch = (
                            self._beta * loss_surrogate_benign
                            + self._gamma * loss_surrogate_poisoned
                        )
                        # 计算梯度并进行累加，累加梯度而不是损失，累加损失会显存爆炸的，bro！
                        g_omega_batch = torch.autograd.grad(
                            loss_surrogate_batch,
                            surrogate_model.parameters(),
                        )
                        for i, g in enumerate(g_omega_batch):
                            g_omega_sum[i] += g / self._outer_grad_batches

                        # 记录平均损失
                        mean_loss_surrogate += loss_surrogate_batch.item()

                    # 每批的平均损失
                    mean_loss_surrogate /= self._outer_grad_batches
                    # g_ω
                    g_omega = [g.detach() for g in g_omega_sum]

                    g_omega_norm, time_consumed = get_grad_norm(g_omega)
                    time_consumed_by_val += time_consumed

                    # ------------- 中场休息 --------------
                    # 这里回顾一下我们要估计的
                    # ∇_λ L_out(ω*(λ), λ) = (∂ω*/∂λ)^T g_ω + g_λ
                    # 此时 g_w 和 g_λ 都已经计算完毕
                    # 还差 (∂ω*/∂λ) 这个硬骨头

                    # 内层优化的更新步 Φ(ω, λ) = ω - ε * ∇ω L_in (ε 是 surrogate_lr)
                    # 设 ω* 是内层 T 步梯度下降的不动点解, 即 ω* = Φ(ω*, λ) 即 ω* = ω* - ε * ∇ω L_in(ω*, λ)
                    # 即在 ω* 处 ∇ω L_in(ω*, λ) = 0，达到一个极小值点
                    # - TIP: 不动点即是一个使函数的输入等于输出的解点

                    # J_{Φ,ω} := ∂Φ/∂ω
                    # J_{Φ,λ} := ∂Φ/∂λ

                    # 根据原文 Eq.(9) 下方的说明，对定点 ω* = Φ(ω*, λ) 两边对 λ 求导：
                    # 得到 (I − J_{Φ,ω}) · (∂ω*/∂λ) = J_{Φ,λ}
                    # 即 ∂ω*/∂λ = (I − J_{Φ,ω})^{-1} J_{Φ,λ}
                    # 则 ∂ω*/∂λ 的转置为 (∂ω*/∂λ)^T = J_{Φ,λ}^T (I − J_{Φ,ω})^{-T}
                    # 要求的第一项就是 (∂ω*/∂λ)^T g_ω = J_{Φ,λ}^T (I − J_{Φ,ω})^{-T} g_ω

                    # 不显式求逆，而是用线性系统的思路：
                    # - 令 v = (I − J_{Φ,ω})^{-1} g_ω = (I − J_{Φ,ω})^{-T} g_ω，
                    # - 则有线性系统 (I − J_{Φ,ω}) v = g_ω (v 是这个线性系统的解)
                    # - 则有 (∂ω*/∂λ)^T g_ω = J_{Φ,λ}^T v
                    # - 则有 ∇_λ L_out = (∂ω*/∂λ)^T g_ω + g_λ = J_{Φ,λ}^T v + g_λ

                    # 固定点迭代的公式：
                    # 上面已经提到 v = (I − J_{Φ,ω})^{-1} g_ω
                    # 当谱半径 ρ(J_{Φ,ω}) < 1，解 v 的经典方法是 Neumann 级数展开:
                    # (I − J_{Φ,ω})^{-1} = Σ_{i=0}^{∞} (J_{Φ,ω})^i
                    # 则有 v = Σ_{i=0}^{∞} (J_{Φ,ω})^i g_ω
                    # 用递推的方式计算 v:
                    # v_0 = 0
                    # v_{n+1} = J_{Φ,ω} v_n + g_ω
                    # 当 n → ∞ 时，v_n → Σ_{i=0}^{∞} (J_{Φ,ω})^i g_ω

                    # 而 v_{n+1} = J_{Φ,ω} v_n + g_ω = v_n - ε * H_{ωω}v_n + g_ω
                    # 固定点迭代 K 次对应取 Neumann 级数的前 K 项，K 越大，越接近真实解

                    # ----------------------------- 接下来开始不动点迭代计算
                    # 最终要得到 K 步后的近似解 v_K

                    # 和前面内层循环一样把教师转换为评估模式
                    teacher_model.eval()

                    # 先计算 ∇ω L_in (内层损失对代理模型参数的梯度) 供后面使用
                    # 这里的 ω 是上面 surrogate model 已经更新了 T 步得到的 ω_T
                    inner_grad_sum = [
                        torch.zeros_like(p) for p in surrogate_model.parameters()
                    ]
                    # 1. ∇ω L_in 是在哪部分数据上计算的呢？
                    #    -  为了节省计算量和显存，和 g_ω 保持一致，且根据附录 F 推测，这里仍然是用子集进行估计
                    # 2. 附录 F 中提到 "40 random batches are selected from the training set in each fixed-point iteration"，那我就有问题了，'each fixed-point iteration' 是每次更新 v_{n+1} 都随机采样 40 个批次重新计算梯度？？
                    #    - 上一句话写到 "each fixed-point iteration runs for 100 steps"，即 Algorithm 中 K=100, 看来这个 'each' 指的是整个 K 步迭代过程，应该是每次外部循环随机采样一次作为 D_s.
                    for images, labels in subset_data:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        student_logits: torch.Tensor = surrogate_model(images)
                        # 注意这里同时要保留关于教师的计算图，后面会用到
                        teacher_logits: torch.Tensor = teacher_model(images)
                        soft_teacher_probs = F.softmax(
                            teacher_logits / self._temperature, dim=-1
                        )

                        loss_ce = F.cross_entropy(student_logits, labels)
                        loss_kd = F.kl_div(
                            F.log_softmax(student_logits / self._temperature, dim=-1),
                            soft_teacher_probs,
                            reduction="batchmean",
                        ) * (self._temperature**2)

                        loss_inner = loss_ce + self._delta * loss_kd

                        # ----------------------------------------------------------------------
                        #               喜报！RuntimeError: CUDA out of memory.  Orz
                        # ----------------------------------------------------------------------

                        grad_inner = torch.autograd.grad(
                            loss_inner, surrogate_model.parameters(), create_graph=True
                        )  # create_graph 准备做二阶导数
                        for i, g in enumerate(grad_inner):
                            inner_grad_sum[i] += g / len(subset_data)

                    # 零初始化 v_0
                    v = [torch.zeros_like(p) for p in surrogate_model.parameters()]
                    # v_{n+1} = J_{Φ,ω} v_n + g_ω
                    # 其中 J_{Φ,ω} v_n 是一个 VJP (Vector-Jacobian product)
                    for i in range(self._fixed_point_iters):
                        # 内层优化的更新步 Φ(ω, λ) = ω - ε * ∇ω L_in (ε 是 surrogate_lr)
                        # J_{Φ,ω}v_n 简化得 = v_n - ε * H_{ωω}v_n
                        # 其中的 Hessian-向量积 (Hessian-vector product):
                        # >>> H_{ωω}v_n = ∇ω( (∇ω L_in) · v_n )
                        # >>> H_{ωω} 即 L_in 对 ω 的 Hessian 矩阵 (二阶导数矩阵)
                        hessian_vector_product = torch.autograd.grad(
                            inner_grad_sum,  # 一阶导 ∇ω L_in
                            surrogate_model.parameters(),
                            grad_outputs=v,  # 乘以向量 v_n
                            retain_graph=True,  # 始终保留计算图，后面还要用到 inner_grad_sum
                        )

                        # 更新 v
                        # v_{n+1} = J_{Φ,ω} v_n + g_ω = v_n - ε * H_{ωω}v_n + g_ω
                        with torch.no_grad():
                            # 推导 v 时用不着记录计算图
                            v = [
                                v_p - self._surrogate_lr * hvp_p + g_omega_p
                                for v_p, hvp_p, g_omega_p in zip(
                                    v, hessian_vector_product, g_omega
                                )
                            ]

                    # K 步后得到近似解
                    v_K = [v_p for v_p in v]

                    # ----------------------------- 接下来开始估算教师梯度
                    # Algorithm 1, line 11, 计算 ∇λ L_out
                    # ∇λ L_out ≈ g_λ (上面计算后在教师原地累积) + J_{Φ,λ}^T v_K
                    # J_{Φ,λ} = -ε * ∇λ(∇ω L_in)
                    # J_{Φ,λ}^T v_K = -ε * ∇λ( ∇ω L_in · v_K )

                    # 计算隐式梯度 ∇λ( ∇ω L_in · v_K )
                    implicit_grads = torch.autograd.grad(
                        inner_grad_sum,  # 一阶导 ∇ω L_in
                        teacher_model.parameters(),  # 对 λ 求导
                        grad_outputs=v_K,  # 乘以 v_K
                    )

                    implicit_grad_norm, time_consumed = get_grad_norm(implicit_grads)
                    time_consumed_by_val += time_consumed

                    # ∇λ L_out ≈ g_λ (上面计算后在教师原地累积) + J_{Φ,λ}^T v_K
                    for p, implicit_grad in zip(
                        teacher_model.parameters(), implicit_grads
                    ):
                        p.grad.add_(-self._surrogate_lr * implicit_grad)

                    final_teacher_grad_norm, time_consumed = get_grad_norm(
                        teacher_model.parameters()
                    )
                    time_consumed_by_val += time_consumed

                    # 更新教师模型
                    teacher_optimizer.step()

                    # 学习率调度器前进一步
                    teacher_lr_scheduler.step()

                    time_val_start = time.perf_counter()

                    tb_writer.add_scalar(
                        f"Outer_Loss/line7_g_lambda_loss_total",
                        mean_loss_direct,
                        outer_epoch + 1,
                    )
                    tb_writer.add_scalar(
                        f"Outer_Loss/line7_g_lambda_loss_benign",
                        recorder_loss_outer_teacher_benign.avg_loss,
                        outer_epoch + 1,
                    )
                    tb_writer.add_scalar(
                        f"Outer_Loss/line7_g_lambda_loss_stealthy",
                        recorder_loss_outer_teacher_stealthy.avg_loss,
                        outer_epoch + 1,
                    )
                    tb_writer.add_scalar(
                        f"Outer_Loss/line7_g_omega_loss_total",
                        mean_loss_surrogate,
                        outer_epoch + 1,
                    )
                    tb_writer.add_scalar(
                        f"Outer_Loss/line7_g_omega_loss_benign",
                        recorder_loss_surrogate_benign.avg_loss,
                        outer_epoch + 1,
                    )
                    tb_writer.add_scalar(
                        f"Outer_Loss/line7_g_omega_loss_attack",
                        recorder_loss_surrogate_attack.avg_loss,
                        outer_epoch + 1,
                    )
                    tb_writer.add_scalar(
                        f"Outer_Grad_Norm/g_lambda_norm", g_lambda_norm, outer_epoch + 1
                    )
                    tb_writer.add_scalar(
                        f"Outer_Grad_Norm/g_omega_norm", g_omega_norm, outer_epoch + 1
                    )
                    tb_writer.add_scalar(
                        f"Outer_Grad_Norm/implicit_grad_norm",
                        implicit_grad_norm,
                        outer_epoch + 1,
                    )
                    tb_writer.add_scalar(
                        f"Outer_Grad_Norm/final_teacher_grad_norm",
                        final_teacher_grad_norm,
                        outer_epoch + 1,
                    )
                    tb_writer.add_scalar(
                        f"Learning_Rate/teacher_lr",
                        teacher_lr_scheduler.get_last_lr()[0],
                        outer_epoch + 1,
                    )
                    pbar.set_postfix(
                        {
                            "||g_λ||": f"{g_lambda_norm:.4f}",
                            "||g_ω||": f"{g_omega_norm:.4f}",
                            "||implicit||": f"{implicit_grad_norm:.4f}",
                            "||final_grad||": f"{final_teacher_grad_norm:.4f}",
                            "lr": f"{teacher_lr_scheduler.get_last_lr()[0]:.6f}",
                        }
                    )
                    pbar.update(1)

                    if (outer_epoch + 1) % self._make_test_per_epochs == 0 or (
                        outer_epoch + 1
                    ) == self._outer_epochs:
                        # 验证教师模型的性能
                        benign_acc = test_benign_accuracy(
                            model=teacher_model,
                            data_loader=self._val_loader,
                            device=device,
                        )
                        attack_success_rate = test_attack_success_rate(
                            model=teacher_model,
                            trigger_gen=self._trigger_preoptimizer,
                            data_loader=self._val_loader,
                            target_label=self._target_label,
                            device=device,
                        )
                        pbar.write(
                            f"Epoch {outer_epoch+1}/{self._outer_epochs}, Benign Acc: {benign_acc:.3%}, ASR: {attack_success_rate:.3%}"
                        )
                        tb_writer.add_scalar(
                            "Val/Benign_Acc", benign_acc, outer_epoch + 1
                        )
                        tb_writer.add_scalar(
                            "Val/ASR", attack_success_rate, outer_epoch + 1
                        )
                        # 可视化当前批次的前 20 张样本
                        vis_image = visualize_images(
                            [img for img in images[:20]],
                            standardized=True,
                        )
                        tb_writer.add_image(
                            "Training Data Samples (20)",
                            vis_image,
                            outer_epoch + 1,
                            dataformats="HWC",
                        )

                    if (outer_epoch + 1) % self._save_ckpts_per_epochs == 0 and (
                        outer_epoch + 1
                    ) < self._outer_epochs:
                        # 保存 Checkpoints
                        curr_random_states = get_curr_random_states()
                        ckpts = {
                            "teacher_model": teacher_model.state_dict(),
                            "teacher_optimizer": teacher_optimizer.state_dict(),
                            "teacher_lr_scheduler": teacher_lr_scheduler.state_dict(),
                            "current_epoch": outer_epoch + 1,
                            "time_start": time_start,
                            "time_consumed_by_val": time_consumed_by_val,
                            "time_save": time.time(),
                            "random_states": curr_random_states,
                        }
                        self.save_checkpoints(ckpts)
                        pbar.write(f"Checkpoints saved at epoch {outer_epoch+1}.")

                    # 记录验证耗时
                    time_val_end = time.perf_counter()
                    time_consumed_by_val += time_val_end - time_val_start

            time_end = time.time()
            torch.save(teacher_model.state_dict(), self._model_save_path)
            self._scar_teacher = teacher_model
            # 存储训练的一些信息以便于追溯
            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self._exp_id,
                    "exp_desc": self._exp_desc,
                    "tensorboard_log_id": tensorboard_log_id,
                    "trigger_gen_exp_id": self._trigger_preoptimizer.exp_id,
                    "params": {
                        "teacher_model_class": self._teacher_model_class.__name__,
                        "surrogate_model_class": self._surrogate_model_class.__name__,
                        "outer_epochs": self._outer_epochs,
                        "inner_updates": self._inner_updates,
                        "fixed_point_iters": self._fixed_point_iters,
                        "outer_grad_batches": self._outer_grad_batches,
                        "teacher_lr": self._teacher_lr,
                        "surrogate_lr": self._surrogate_lr,
                        "alpha": self._alpha,
                        "beta": self._beta,
                        "gamma": self._gamma,
                        "delta": self._delta,
                        "temperature": self._temperature,
                        "target_label": self._target_label,
                        "dataset_name": self._dataset_info.name,
                        "data_transform_class": self._data_transform_class.__name__,
                        "batch_size": self._batch_size,
                        "make_test_per_epochs": self._make_test_per_epochs,
                        "num_workers": self._num_workers,
                        "seed": self._seed,
                    },
                }
            )
            self.save_exp_info(exp_info, time_start, time_end, time_consumed_by_val)
            tb_writer.add_text(
                "Experiment Info",
                json.dumps(
                    exp_info,
                    indent=4,
                    ensure_ascii=False,
                    default=json_serialize_helper,
                ),
            )
            tb_writer.close()
            # 一切保存完毕后，移除掉临时的 Checkpoints 文件
            self.del_checkpoints()

        return self._scar_teacher

    def get_model(self):
        """
        get_scar_teacher 的别名
        """
        return self.get_scar_teacher()
