"""
复现论文 Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation 的代码

* Ref: Ge Y, Wang Q, Zheng B, et al. Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation[C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 826-834.

虽然 ADBA 有第三方实现: https://github.com/brcarry/ADBA, 但是其代码有明显问题: https://github.com/brcarry/ADBA/blob/98490bffe3379a02e4697464495bb0485c58da4b/main.py#L99C17-L100C43
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from .trigger_adba import ADBATrigger
from reprod_modules.abc import ADBABase
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Type
from modules.abc import TriggerGenerator
from data_augs.abc import MakeTransforms
from utils.data import DatasetWithInfo, TransformedDataset
from utils.visualizer_trigger import TriggerVisualizer
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
    apply_trigger_with_mask,
)
from utils.visualization import visualize_images

from utils.records import AverageLossRecorder
from utils.data import DatasetWithInfo, TransformedDataset

_default_save_dir = os.path.join(REPROD_CHECKPOINTS_SAVE_PATH, "adba_main")
_default_num_workers = auto_num_workers()


class ADBA(ADBABase):
    """
    复现论文 Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation 的代码模块
    """

    def __init__(
        self,
        teacher_model_class: Type[nn.Module],
        shadow_model_class: Type[nn.Module],
        epochs: int,
        alpha: float,
        temperature: float,
        beta: float,
        mu: float,
        p: int,
        c: float,
        target_label: int,
        batch_size: int,
        teacher_lr: float,
        trigger_lr: float,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        exp_id: str,
        exp_desc: str = "",
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 1,
        num_workers: int = _default_num_workers,
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        初始化 ADBA 模块

        :param teacher_model_class: 教师模型的类
        :param shadow_model_class: 影子学生模型的类
        :param epochs: 训练轮数
        :param alpha: 模拟蒸馏时 KD 项的权重 (α)
        :param temperature: 蒸馏温度 (h)
        :param beta: 教师学习后门知识的损失的权重 (β)
        :param mu: 后门触发器的 p 范数正则项权重 (μ)
        :param p: 后门触发器的 p 范数类型
        :param c: 后门触发器的数值范围上限 (c >= 1, 触发器值限制在 [0, 1/c] 之间)
        :param target_label: 后门攻击的目标标签 (k)
        :param batch_size: 训练批次大小
        :param teacher_lr: 教师模型 (以及影子学生模型) 学习率
        :param trigger_lr: 触发器优化学习率
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强变换类
        :param exp_id: 实验 ID，用于区分不同实验的保存目录
        :param exp_desc: 实验描述信息
        :param make_test_per_epochs: 每隔多少轮进行一次测试
        :param save_ckpts_per_epochs: 每隔多少轮保存一次模型
        :param num_workers: 数据加载的工作线程数
        :param seed: 随机种子
        :param save_dir: 实验结果保存目录
        """
        super().__init__()
        model_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(model_save_dir, exist_ok=True)

        self._final_ckpts_save_path = os.path.join(
            model_save_dir, "adba_final_ckpts.pth"
        )
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
        self._teacher_model_class = teacher_model_class
        self._shadow_model_class = shadow_model_class
        self._epochs = epochs
        self._alpha = alpha
        self._temperature = temperature
        self._beta = beta
        self._mu = mu
        self._p = p
        self._c = c
        self._target_label = target_label
        self._batch_size = batch_size
        self._teacher_lr = teacher_lr
        self._trigger_lr = trigger_lr
        self._dataset_info = dataset_info
        self._train_tensor_set = train_tensor_set
        self._val_tensor_set = val_tensor_set
        self._data_transform = data_transform
        self._data_transform_class = data_transform_class
        self._make_test_per_epochs = make_test_per_epochs
        self._save_ckpts_per_epochs = save_ckpts_per_epochs
        self._exp_id = exp_id
        self._exp_desc = exp_desc
        self._num_workers = num_workers
        self._seed = seed
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
        self._adba_teacher = None
        self._trigger_mask = None
        self._trigger_pattern = None

    @property
    def exp_id(self) -> str:
        return self._exp_id

    def _load_if_final_ckpts_exists(self) -> bool:
        """
        如果最终检查点存在，则加载并返回 True，否则返回 False

        :return: 是否加载了最终检查点
        """
        if os.path.exists(self._final_ckpts_save_path):
            device = auto_select_device()
            final_ckpts = torch.load(self._final_ckpts_save_path, map_location=device)
            adba_teacher = self._teacher_model_class(
                num_classes=self._dataset_info.num_classes
            )
            adba_teacher.load_state_dict(final_ckpts["teacher_model"])
            adba_teacher.to(device)
            self._adba_teacher = adba_teacher
            self._trigger_mask = final_ckpts["trigger_mask"].detach().to(device)
            self._trigger_pattern = final_ckpts["trigger_pattern"].detach().to(device)
            return True
        return False

    def get_trigger_generator(self) -> TriggerGenerator:
        if self._trigger_mask is None or self._trigger_pattern is None:
            if not self._load_if_final_ckpts_exists():
                raise RuntimeError(
                    "ADBA trigger not found. Please train the ADBA model first."
                )
        return ADBATrigger(
            trigger_pattern=self._trigger_pattern,
            trigger_mask=self._trigger_mask,
        )

    def get_adba_teacher(self) -> nn.Module:
        """
        获取训练好的 ADBA 教师模型

        :return: 训练好的 ADBA 教师模型
        """
        if self._adba_teacher is not None:
            return self._adba_teacher

        if self._load_if_final_ckpts_exists():
            # 如果最终检查点已经存在，直接载入返回
            return self._adba_teacher

        with temp_seed(self._seed):
            device = auto_select_device()
            teacher_model = self._teacher_model_class(
                num_classes=self._dataset_info.num_classes
            )
            shadow_model = self._shadow_model_class(
                num_classes=self._dataset_info.num_classes
            )
            teacher_model.to(device)
            shadow_model.to(device)

            print_section(f"ADBA Training: {self.exp_id:.20s}")

            # 开始时间
            time_start = time.time()
            # 进行验证时耗费的时间
            time_consumed_by_val = 0.0

            # TensorBoard 记录器
            tensorboard_log_id = f"adba_main_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH, tensorboard_log_id
            )
            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # ------------------------ 初始化触发器，文中没有提到怎么初始化的
            # 为了让初期训练稳定，这里用 1 初始化 trigger_mask，用随机值初始化 pattern
            trigger_mask = torch.ones(
                (1, 1, *self._dataset_info.shape[-2:]),
                device=device,
            )  # shape (1, 1, H, W)
            trigger_pattern = torch.normal(
                0, 1, (1, *self._dataset_info.shape), device=device
            )  # shape (1, C, H, W)

            # 原文把触发器限制到 [0, 1/c] 范围内
            trigger_pattern = trigger_pattern.clip(0, 1 / self._c)
            # 这里因为图像标准化到了 [-1, 1] 范围内，还需要对触发器进行一次变换
            # 因此范围变换： [0, 1/c] -> [-1, 2/c-1]
            trigger_pattern: torch.Tensor = self._data_transform.standardize(
                trigger_pattern
            )
            trigger_pattern.requires_grad = True
            trigger_mask.requires_grad = True

            c_tensor = torch.tensor(1 / self._c, device=device).view(1, 1, 1, 1)
            standardized_c_tensor = self._data_transform.standardize(
                c_tensor
            )  # shape (1, 3, 1, 1)

            # 模型训练
            teacher_model.requires_grad_(True)
            shadow_model.requires_grad_(True)

            # 优化器
            # 文中完全没有提到优化器、学习率等细节，
            # 这里对教师 / 影子学生模型用 SGD，对触发器用 RAdam
            teacher_optimizer = torch.optim.SGD(
                teacher_model.parameters(), lr=self._teacher_lr, momentum=0.9
            )
            shadow_optimizer = torch.optim.SGD(
                shadow_model.parameters(), lr=self._teacher_lr, momentum=0.9
            )
            trigger_optimizer = torch.optim.RAdam(
                [trigger_mask, trigger_pattern], lr=self._trigger_lr
            )
            # 学习率调度器
            teacher_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                teacher_optimizer, T_max=self._epochs
            )
            shadow_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                shadow_optimizer, T_max=self._epochs
            )

            start_epoch = 0

            # 如果有 Checkpoints 则载入
            if self.has_checkpoints():
                ckpts = self.load_checkpoints()
                # 载入之前的状态
                teacher_ckpt = ckpts["teacher_model"]
                shadow_ckpt = ckpts["shadow_model"]
                trigger_ckpts = ckpts["trigger"]
                teacher_model.load_state_dict(teacher_ckpt["model_state"])
                shadow_model.load_state_dict(shadow_ckpt["model_state"])
                with torch.no_grad():
                    # 载入触发器状态时应该临时关闭计算图构建
                    trigger_mask.copy_(trigger_ckpts["mask"])
                    trigger_pattern.copy_(trigger_ckpts["pattern"])
                teacher_optimizer.load_state_dict(teacher_ckpt["optimizer_state"])
                shadow_optimizer.load_state_dict(shadow_ckpt["optimizer_state"])
                trigger_optimizer.load_state_dict(trigger_ckpts["optimizer_state"])
                teacher_lr_scheduler.load_state_dict(teacher_ckpt["lr_scheduler_state"])
                shadow_lr_scheduler.load_state_dict(shadow_ckpt["lr_scheduler_state"])
                # 载入时间，轮数信息
                time_start = ckpts["time_start"]  # 训练最初开始时间，用于计算总耗时
                time_consumed_by_val = ckpts["time_consumed_by_val"]  # 训练中验证耗时
                ckpt_save_time = ckpts[
                    "time_save"
                ]  # 上次 ckpt 保存时间，用于从总耗时中减去
                time_consumed_by_val += time.time() - ckpt_save_time
                start_epoch = ckpts["current_epoch"]
                # 载入随机状态
                prev_random_states = ckpts["random_states"]
                load_random_states(prev_random_states)
                print(f"Loaded checkpoints from epoch {start_epoch}.")

            # 开始训练模型
            with tqdm(
                initial=start_epoch,
                total=self._epochs,
                desc="ADBA Training",
            ) as pbar:
                for epoch in range(start_epoch, self._epochs):
                    recorder_loss_l_t_clean = AverageLossRecorder()
                    recorder_loss_l_t_poisoned = AverageLossRecorder()
                    recorder_loss_l_s_hard = AverageLossRecorder()
                    recorder_loss_l_s_soft = AverageLossRecorder()
                    recorder_loss_l_trigger = AverageLossRecorder()
                    for images, labels in self._train_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)
                        n_batch = images.size(0)

                        # 生成后门样本的目标标签
                        labels_poisoned = torch.full_like(labels, self._target_label)

                        # ========== STEP 1 - 让教师模型学习正常知识和后门知识 ==========
                        teacher_model.train()

                        images_poisoned = apply_trigger_with_mask(
                            images, trigger_pattern.detach(), trigger_mask.detach()
                        )

                        teacher_outputs_clean = teacher_model(images)
                        loss_clean = F.cross_entropy(teacher_outputs_clean, labels)

                        teacher_outputs_poisoned = teacher_model(images_poisoned)
                        loss_poisoned = F.cross_entropy(
                            teacher_outputs_poisoned, labels_poisoned
                        )

                        # 原文的 L_t, Eq. (5)
                        loss_teacher = loss_clean + self._beta * loss_poisoned

                        teacher_optimizer.zero_grad()
                        loss_teacher.backward()
                        teacher_optimizer.step()
                        recorder_loss_l_t_clean.batch_update(loss_clean, n_batch)
                        recorder_loss_l_t_poisoned.batch_update(loss_poisoned, n_batch)

                        # ========== STEP 2 - 模拟蒸馏影子学生模型 ==========
                        teacher_model.eval()
                        shadow_model.train()

                        with torch.no_grad():
                            teacher_outputs_clean = teacher_model(images)
                            teacher_softmaxes_clean = F.softmax(
                                teacher_outputs_clean / self._temperature, dim=-1
                            )

                        shadow_outputs_clean = shadow_model(images)
                        loss_soft = F.kl_div(
                            F.log_softmax(
                                shadow_outputs_clean / self._temperature, dim=-1
                            ),
                            teacher_softmaxes_clean,
                            reduction="batchmean",
                        ) * (self._temperature**2)

                        loss_hard = F.cross_entropy(shadow_outputs_clean, labels)

                        # 原文的 L_s, Eq. (3)
                        loss_kd = (
                            self._alpha * loss_soft + (1 - self._alpha) * loss_hard
                        )

                        shadow_optimizer.zero_grad()
                        loss_kd.backward()
                        shadow_optimizer.step()
                        recorder_loss_l_s_hard.batch_update(loss_hard, n_batch)
                        recorder_loss_l_s_soft.batch_update(loss_soft, n_batch)

                        # ========== STEP 3 - 优化触发器使其能尽量让教师和学生误分类 ==========
                        teacher_model.eval()
                        shadow_model.eval()

                        images_poisoned = apply_trigger_with_mask(
                            images, trigger_pattern, trigger_mask
                        )

                        teacher_outputs_poisoned = teacher_model(images_poisoned)
                        shadow_outputs_poisoned = shadow_model(images_poisoned)

                        loss_trigger_teacher = F.cross_entropy(
                            teacher_outputs_poisoned, labels_poisoned
                        )
                        loss_trigger_shadow = F.cross_entropy(
                            shadow_outputs_poisoned, labels_poisoned
                        )

                        # 原文的 L_{m,δ}, Eq. (7)
                        loss_trigger = (
                            loss_trigger_teacher
                            + loss_trigger_shadow
                            + self._mu * torch.norm(trigger_mask, p=self._p)
                        )

                        trigger_optimizer.zero_grad()
                        loss_trigger.backward()
                        trigger_optimizer.step()
                        recorder_loss_l_trigger.batch_update(loss_trigger, n_batch)

                        # 将触发器限制到 c 指定的范围内
                        with torch.no_grad():
                            trigger_mask.clip_(0, 1)
                            # 将触发器图案限制到 [-1, 2/c-1] 的范围内
                            # 相当于原图的 [0, 1/c] 范围
                            lower_bound = torch.full_like(standardized_c_tensor, -1)
                            trigger_pattern.clip_(lower_bound, standardized_c_tensor)

                    # 更新学习率
                    teacher_lr_scheduler.step()
                    shadow_lr_scheduler.step()
                    pbar.set_postfix(
                        {
                            "L_t_clean": f"{recorder_loss_l_t_clean.avg_loss:.4f}",
                            "L_t_poi": f"{recorder_loss_l_t_poisoned.avg_loss:.4f}",
                            "L_s_hard": f"{recorder_loss_l_s_hard.avg_loss:.4f}",
                            "L_s_soft": f"{recorder_loss_l_s_soft.avg_loss:.4f}",
                            "L_tri": f"{recorder_loss_l_trigger.avg_loss:.4f}",
                        }
                    )
                    pbar.update(1)

                    time_val_start = time.perf_counter()
                    tb_writer.add_scalar(
                        "Train/L_t_clean", recorder_loss_l_t_clean.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Train/L_t_poisoned",
                        recorder_loss_l_t_poisoned.avg_loss,
                        epoch + 1,
                    )
                    tb_writer.add_scalar(
                        "Train/L_s_hard", recorder_loss_l_s_hard.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Train/L_s_soft", recorder_loss_l_s_soft.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Train/L_trigger", recorder_loss_l_trigger.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Train/LR", teacher_lr_scheduler.get_last_lr()[0], epoch + 1
                    )

                    if (epoch + 1) % self._make_test_per_epochs == 0 or (
                        epoch + 1
                    ) == self._epochs:
                        benign_acc = test_benign_accuracy(
                            model=teacher_model,
                            data_loader=self._val_loader,
                            device=device,
                        )
                        attack_success_rate = test_attack_success_rate(
                            model=teacher_model,
                            trigger=trigger_pattern.detach(),
                            trigger_mask=trigger_mask.detach(),
                            data_loader=self._val_loader,
                            target_label=self._target_label,
                            device=device,
                        )
                        pbar.write(
                            f"Epoch {epoch+1}/{self._epochs}, Benign Acc: {benign_acc:.3%}, ASR: {attack_success_rate:.3%}"
                        )
                        tb_writer.add_scalar("Val/Benign_Acc", benign_acc, epoch + 1)
                        tb_writer.add_scalar("Val/ASR", attack_success_rate, epoch + 1)
                        # 可视化当前批次的前 20 张样本
                        vis_image = visualize_images(
                            [img for img in images[:20]],
                            standardized=True,
                        )
                        tb_writer.add_image(
                            "Training Data Samples (20)",
                            vis_image,
                            epoch + 1,
                            dataformats="HWC",
                        )

                    if (epoch + 1) % self._save_ckpts_per_epochs == 0 and (
                        epoch + 1
                    ) < self._epochs:
                        # 保存 Checkpoints
                        curr_random_states = get_curr_random_states()
                        ckpts = {
                            "teacher_model": {
                                "model_state": teacher_model.state_dict(),
                                "optimizer_state": teacher_optimizer.state_dict(),
                                "lr_scheduler_state": teacher_lr_scheduler.state_dict(),
                            },
                            "shadow_model": {
                                "model_state": shadow_model.state_dict(),
                                "optimizer_state": shadow_optimizer.state_dict(),
                                "lr_scheduler_state": shadow_lr_scheduler.state_dict(),
                            },
                            "trigger": {
                                "mask": trigger_mask.detach(),
                                "pattern": trigger_pattern.detach(),
                                "optimizer_state": trigger_optimizer.state_dict(),
                            },
                            "time_start": time_start,
                            "time_consumed_by_val": time_consumed_by_val,
                            "time_save": time.time(),
                            "current_epoch": epoch + 1,
                            "random_states": curr_random_states,
                        }
                        self.save_checkpoints(ckpts)
                        pbar.write(f"Checkpoints saved at epoch {epoch+1}.")

                    # 计算验证耗时
                    time_val_end = time.perf_counter()
                    time_consumed_by_val += time_val_end - time_val_start

            time_end = time.time()
            torch.save(
                {
                    "teacher_model": teacher_model.state_dict(),
                    "trigger_mask": trigger_mask.detach(),
                    "trigger_pattern": trigger_pattern.detach(),
                },
                self._final_ckpts_save_path,
            )
            self._adba_teacher = teacher_model
            self._trigger_mask = trigger_mask.detach()
            self._trigger_pattern = trigger_pattern.detach()

            # 存储训练的一些信息以便于追溯
            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self._exp_id,
                    "exp_desc": self._exp_desc,
                    "tensorboard_log_id": tensorboard_log_id,
                    "params": {
                        "teacher_model_class": self._teacher_model_class.__name__,
                        "shadow_model_class": self._shadow_model_class.__name__,
                        "epochs": self._epochs,
                        "alpha": self._alpha,
                        "temperature": self._temperature,
                        "beta": self._beta,
                        "mu": self._mu,
                        "p": self._p,
                        "c": self._c,
                        "target_label": self._target_label,
                        "batch_size": self._batch_size,
                        "teacher_lr": self._teacher_lr,
                        "trigger_lr": self._trigger_lr,
                        "dataset_name": self._dataset_info.name,
                        "data_transform_class": self._data_transform_class.__name__,
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
            trigger_generator = self.get_trigger_generator()
            trigger_visualizer = TriggerVisualizer(
                dataset_info=self._dataset_info,
                data_transform_class=self._data_transform_class,
                trigger_gen=trigger_generator,
            )
            # 可视化触发器
            orig_img, orig_trig, trig_img = trigger_visualizer.visualize_single(
                use_transform=False
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
                "Trigger Visualization", trigger_vis_image, dataformats="HWC"
            )
            tb_writer.close()
            # 一切保存完毕后，移除掉临时的 Checkpoints 文件
            self.del_checkpoints()

    def get_model(self):
        """
        get_adba_teacher 的别名
        """
        return self.get_adba_teacher()
