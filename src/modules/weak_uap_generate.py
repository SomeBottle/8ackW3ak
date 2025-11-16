"""
文中核心模块，Weak Trigger 的生成
"""

import torch
import torch.optim as optim
import os
import torch.nn.functional as F
import json
import time

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Type
from torch.utils.data import DataLoader
from modules.abc import TriggerGenerator
from data_augs.abc import MakeTransforms
from modules.abc import NormalTrainer
from utils.visualizer_trigger import TriggerVisualizer
from configs import (
    IMAGE_STANDARDIZE_STDS,
    CHECKPOINTS_SAVE_PATH,
    TENSORBOARD_LOGS_PATH,
)

from utils.funcs import (
    auto_select_device,
    auto_num_workers,
    temp_seed,
    apply_trigger_without_mask,
    get_base_exp_info,
    print_section,
    json_serialize_helper,
)
from utils.visualization import visualize_images
from utils.data import DatasetWithInfo, TransformedDataset
from utils.records import AverageLossRecorder

_default_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "weak_uap")
_default_num_workers = auto_num_workers()


class WeakUAPGenerator(TriggerGenerator):
    def __init__(
        self,
        normal_trainer: NormalTrainer,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        exp_id: str,
        l_inf_r: float,
        budget_asr: float,
        lambda_margin: float,
        mu_margin: float,
        lr: float,
        epochs: int,
        batch_size: int,
        target_label: int,
        optimizer_class: Type[optim.Optimizer],
        optimizer_params: dict,
        exp_desc: str = "",
        num_workers: int = _default_num_workers,
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        生成弱 UAP 触发器

        :param normal_trainer: NormalTrainer 实例
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param exp_id: 实验 ID
        :param l_inf_r: 触发器 L_inf 范数约束
        :param budget_asr: ASR 预算 (标签翻转比例)
        :param lambda_margin: margin 损失的权重
        :param mu_margin: margin 的裕度
        :param lr: 学习率
        :param epochs: 训练轮数
        :param batch_size: 训练批大小
        :param target_label: 目标标签
        :param optimizer_class: 优化器类
        :param optimizer_params: 优化器参数
        :param exp_desc: 实验描述
        :param num_workers: DataLoader 的 num_workers
        :param seed: 随机种子
        :param save_dir: 结果保存路径
        """
        super().__init__()
        # -------------------------------- 数据增强
        data_transform = data_transform_class(input_shape=dataset_info.shape)
        # -------------------------------- 数据集转换
        train_tensor_set = TransformedDataset(
            dataset=dataset_info.train_set, transform=data_transform.train_transforms
        )
        val_tensor_set = TransformedDataset(
            dataset=dataset_info.val_set, transform=data_transform.val_transforms
        )
        # -------------------------------- 保存路径初始化
        trigger_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(trigger_save_dir, exist_ok=True)

        self._trigger_save_path = os.path.join(trigger_save_dir, "trigger.pt")
        self.set_exp_save_dir(trigger_save_dir)
        self._model_class = normal_trainer.model_class
        self._data_transform_class = data_transform_class
        self._data_transform = data_transform
        self._normal_trainer = normal_trainer
        self._dataset_info = dataset_info
        self._exp_id = exp_id
        self._l_inf_r = l_inf_r
        self._budget_asr = budget_asr
        self._lambda_margin = lambda_margin
        self._mu_margin = mu_margin
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._target_label = target_label
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._num_workers = num_workers
        self._seed = seed
        self._exp_desc = exp_desc
        self._train_set = train_tensor_set
        self._val_set = val_tensor_set
        self._train_loader = DataLoader(
            train_tensor_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self._val_loader = DataLoader(
            val_tensor_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self._visualizer = TriggerVisualizer(
            dataset_info=dataset_info,
            data_transform_class=data_transform_class,
            trigger_gen=self,
        )
        self._trigger = None

    @property
    def exp_id(self) -> str:
        """
        获取实验 ID

        :return: 实验 ID
        """
        return self._exp_id

    def get_trigger(self) -> torch.Tensor:
        """
        获得训练好的触发器，如果不存在则训练
        """

        if self._trigger is not None:
            return self._trigger

        if os.path.exists(self._trigger_save_path):
            trigger = torch.load(self._trigger_save_path, map_location="cpu")
            self._trigger = trigger.cpu()
            return trigger

        # fix_seed 一定不能放到上面了！不然每次获取触发器都会重置随机数种子！！！
        # 惨痛的教训，这项会影响到数据增强的功能，导致随机变换意外固定！

        with temp_seed(self._seed):

            print_section(f"Weak Trigger Gen: {self.exp_id:.20s}")
            device = auto_select_device()  # 自动选择空闲显存最大的设备

            time_start = time.time()

            tensorboard_log_id = f"weak_uap_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH,
                tensorboard_log_id,
            )
            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # 载入之前训练好的模型
            model = self._model_class(num_classes=self._dataset_info.num_classes)
            model.to(device)
            model.load_state_dict(self._normal_trainer.get_trained_model().state_dict())
            model.eval()
            model.requires_grad_(False)

            # 初始化触发器，设值在 [-1, 1] 范围内
            img_c, img_h, img_w = self._dataset_info.shape
            uap_trigger = torch.zeros(
                (1, img_c, img_h, img_w), device=device, requires_grad=True
            )

            uap_optimizer = self._optimizer_class(
                [uap_trigger], lr=self._lr, **self._optimizer_params
            )
            uap_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                uap_optimizer, T_max=self._epochs
            )

            with tqdm(
                total=self._epochs, desc=f"Generating UAP Trigger({self._exp_id:.20s})"
            ) as pbar:
                current_asr = 0.0  # 上一轮的攻击成功率
                for epoch in range(self._epochs):
                    recorder_loss_push = AverageLossRecorder()
                    recorder_loss_margin = AverageLossRecorder()
                    # 这一轮所有数据上被误导误分类为目标的样本数
                    attack_success_count = 0
                    total_non_target_count = 0  # 这一轮所有非目标数据的样本数
                    for images, labels in self._train_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)
                        n_batch = images.size(0)

                        # 目标标签
                        target_labels = torch.full_like(labels, self._target_label)

                        triggered_images = apply_trigger_without_mask(
                            images, uap_trigger
                        )

                        triggered_outputs: torch.Tensor = model(triggered_images)
                        normal_outputs: torch.Tensor = model(images)
                        # 对数概率
                        triggered_log_p = F.log_softmax(triggered_outputs, dim=-1)
                        normal_log_p = F.log_softmax(normal_outputs, dim=-1)
                        # loss push: 目标类的对数概率尽量提升
                        loss_push = -(
                            triggered_log_p[torch.arange(n_batch), target_labels]
                            - normal_log_p[torch.arange(n_batch), target_labels]
                        ).mean()
                        non_true_mask = (
                            triggered_outputs.argmax(dim=-1) != labels
                        )  # 标签翻转的部分
                        attack_success_mark = non_true_mask & (
                            triggered_outputs.argmax(dim=-1) == target_labels
                        )  # 被攻击成功的部分 (不考虑原本就是目标标签的样本)

                        if non_true_mask.sum() == 0:
                            # 防止 mean 产生的 NaN
                            # 而且不能脱离对于 uap_trigger 的依赖，不然 backward 会报错
                            loss_margin = triggered_outputs.mean() * 0.0
                        else:
                            # loss margin: 控制标签翻转的程度
                            loss_margin = F.relu(
                                triggered_outputs[non_true_mask].max(dim=-1).values
                                - triggered_outputs[
                                    non_true_mask, labels[non_true_mask]
                                ]
                                - self._mu_margin
                            ).mean()

                        loss = 0.0

                        # 在没有超出预算的情况下，才优化 loss_push
                        if current_asr < self._budget_asr:
                            loss += loss_push

                        loss += self._lambda_margin * loss_margin

                        uap_optimizer.zero_grad()
                        loss.backward()
                        uap_optimizer.step()
                        recorder_loss_push.batch_update(loss_push, n_batch)
                        recorder_loss_margin.batch_update(
                            loss_margin, non_true_mask.sum().item()
                        )
                        attack_success_count += attack_success_mark.sum().item()

                        # 估计 Cur ASR 时分母也要排除掉原本就是目标标签的样本
                        total_non_target_count += (labels != target_labels).sum().item()

                        # 投影到 L_inf 范数约束
                        with torch.no_grad():
                            normalize_stds = torch.tensor(
                                IMAGE_STANDARDIZE_STDS, device=device
                            ).view(1, -1, 1, 1)
                            # 保持和输入图像同样的标准化尺度
                            uap_trigger.clip_(
                                -self._l_inf_r / normalize_stds,
                                self._l_inf_r / normalize_stds,
                            )

                    # 记录这一轮的 ASR
                    current_asr = attack_success_count / max(total_non_target_count, 1)
                    uap_scheduler.step()
                    pbar.set_postfix(
                        {
                            "Loss Push": f"{recorder_loss_push.avg_loss:.4f}",
                            "Loss Margin": f"{recorder_loss_margin.avg_loss:.4f}",
                            "Cur ASR": f"{current_asr:.4f}",
                            "LR": f"{uap_scheduler.get_last_lr()[0]:.4e}",
                        }
                    )
                    pbar.update(1)
                    tb_writer.add_scalar(
                        "Loss/Push", recorder_loss_push.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Loss/Margin", recorder_loss_margin.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar("Stats/CurASR", current_asr, epoch + 1)
                    tb_writer.add_scalar(
                        "Train/LR", uap_scheduler.get_last_lr()[0], epoch + 1
                    )

            # 保存触发器
            self._trigger = uap_trigger.detach().cpu()
            time_end = time.time()
            torch.save(self._trigger, self._trigger_save_path)

            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self._exp_id,
                    "exp_desc": self._exp_desc,
                    "tensorboard_log_id": tensorboard_log_id,
                    "normal_trainer_exp_id": self._normal_trainer.exp_id,  # 用于追溯的正常模型实验 ID
                    "params": {
                        "data_transform_class": self._data_transform_class.__name__,
                        "dataset_name": self._dataset_info.name,
                        "l_inf_r": self._l_inf_r,
                        "lambda_margin": self._lambda_margin,
                        "budget_asr": self._budget_asr,
                        "mu_margin": self._mu_margin,
                        "lr": self._lr,
                        "epochs": self._epochs,
                        "batch_size": self._batch_size,
                        "target_label": self._target_label,
                        "optimizer_class": self._optimizer_class.__name__,
                        "optimizer_params": self._optimizer_params,
                        "num_workers": self._num_workers,
                        "seed": self._seed,
                    },
                }
            )
            self.save_exp_info(exp_info, time_start, time_end, 0)
            tb_writer.add_text(
                "Experiment Info",
                json.dumps(
                    exp_info,
                    indent=4,
                    ensure_ascii=False,
                    default=json_serialize_helper,
                ),
            )
            # 可视化触发器
            orig_img, orig_trig, trig_img = self._visualizer.visualize_single(
                use_transform=True
            )
            trigger_vis_image = visualize_images(
                [
                    (orig_img, "Original"),
                    (trig_img, "Triggered"),
                    (orig_trig, "Trigger"),
                ],
                standardized=True,
            )
            tb_writer.add_image(
                "Trigger Visualization",
                trigger_vis_image,
                dataformats="HWC",
            )
            tb_writer.close()

        return self._trigger

    def generate(self):
        self.get_trigger()

    def apply_trigger(self, input_data: torch.Tensor, transform=None) -> torch.Tensor:
        trigger = self.get_trigger()
        trigger = trigger.to(input_data.device)
        if transform:
            trigger: torch.Tensor = transform(trigger)
        if input_data.dim() == 3:
            trigger = trigger.squeeze(0)
        return apply_trigger_without_mask(input_data, trigger)
