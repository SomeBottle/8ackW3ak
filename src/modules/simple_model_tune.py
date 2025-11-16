"""
把触发器加入模型进行微调
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Type
from torch.utils.data import DataLoader, Dataset
from configs import (
    CHECKPOINTS_SAVE_PATH,
    TENSORBOARD_LOGS_PATH,
)
from data_augs.abc import MakeTransforms
from modules.abc import ModelTuner, TriggerGenerator, NormalTrainer, DataPoisoner
from utils.funcs import (
    auto_select_device,
    auto_num_workers,
    temp_seed,
    get_base_exp_info,
    test_benign_accuracy,
    test_attack_success_rate,
    print_section,
    json_serialize_helper,
    load_random_states,
    get_curr_random_states,
    freeze_last_n_layers,
)
from utils.visualization import visualize_images

from utils.records import AverageLossRecorder
from utils.data import DatasetWithInfo, TransformedDataset

_default_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "model_tune")
_default_num_workers = auto_num_workers()


class SimpleModelTuner(ModelTuner):
    """
    简单的模型微调，直接用有毒数据集微调模型
    """

    def __init__(
        self,
        normal_trainer: NormalTrainer,
        trigger_gen: TriggerGenerator,
        data_poisoner: DataPoisoner,
        target_label: int,
        exp_id: str,
        epochs: int,
        lr: float,
        batch_size: int,
        optimizer_class: Type[optim.Optimizer],
        optimizer_params: dict,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        layer_freeze_n: int = 0,
        exp_desc: str = "",
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 5,
        num_workers: int = _default_num_workers,
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        用有毒数据微调模型

        :param normal_trainer: NormalTrainer 实例
        :param trigger_gen: 触发器生成模块
        :param data_poisoner: 数据投毒模块
        :param target_label: 目标标签
        :param exp_id: 实验 ID
        :param epochs: 训练轮数
        :param lr: 学习率
        :param batch_size: 批大小
        :param optimizer_class: 优化器类
        :param optimizer_params: 优化器参数
        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强模块类
        :param layer_freeze_n: 冻结模型倒数 n 层
        :param exp_desc: 实验描述
        :param make_test_per_epochs: 每隔多少轮在验证集上测试一次
        :param save_ckpts_per_epochs: 每隔多少轮保存一次 Checkpoints
        :param num_workers: DataLoader 的 num_workers 参数
        :param seed: 随机种子
        :param save_dir: 模型保存路径
        """
        super().__init__()
        model_save_dir = os.path.join(save_dir, exp_id)
        os.makedirs(model_save_dir, exist_ok=True)

        self._model_save_path = os.path.join(model_save_dir, "backdoored_model.pth")
        self.set_exp_save_dir(model_save_dir)
        self._normal_trainer = normal_trainer
        self._target_label = target_label
        self._trigger_gen = trigger_gen
        self._data_poisoner = data_poisoner
        self._exp_id = exp_id
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._layer_freeze_n = layer_freeze_n
        self._exp_desc = exp_desc
        self._make_test_per_epochs = make_test_per_epochs
        self._save_ckpts_per_epochs = save_ckpts_per_epochs
        self._num_workers = num_workers
        self._seed = seed
        self._save_dir = save_dir
        # 验证集构造
        data_transform = data_transform_class(input_shape=dataset_info.shape)
        val_tensor_set = TransformedDataset(
            dataset=dataset_info.val_set, transform=data_transform.val_transforms
        )
        self._dataset_info = dataset_info
        self._poisoned_loader = None
        self._val_loader = DataLoader(
            val_tensor_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self._tuned_model = None

    @property
    def exp_id(self) -> str:
        return self._exp_id

    @property
    def model_class(self) -> Type[nn.Module]:
        return self._normal_trainer.model_class

    @property
    def trigger_generator(self) -> TriggerGenerator:
        return self._trigger_gen

    def get_tuned_model(self) -> nn.Module:
        """
        获得微调后的模型

        :return: 微调后的模型
        """

        if self._tuned_model is not None:
            return self._tuned_model

        # 临时应用种子
        with temp_seed(self._seed):
            device = auto_select_device()
            model = self._normal_trainer.model_class(
                num_classes=self._dataset_info.num_classes
            )
            model.to(device)

            if os.path.exists(self._model_save_path):
                model.load_state_dict(
                    torch.load(self._model_save_path, map_location=device)
                )
                self._tuned_model = model
                return self._tuned_model

            # 惰性构造有毒数据集 Dataloader
            if not self._poisoned_loader:
                self._poisoned_loader = DataLoader(
                    self._data_poisoner.get_poisoned_data(),
                    batch_size=self._batch_size,
                    shuffle=True,
                    num_workers=self._num_workers,
                    pin_memory=True,
                )

            time_start = time.time()
            time_consumed_by_val = 0.0

            # TensorBoard 记录器
            tensorboard_log_id = f"model_tune_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH, tensorboard_log_id
            )
            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # 载入正常模型
            model.load_state_dict(self._normal_trainer.get_trained_model().state_dict())

            print_section(f"Simple Model Tuning: {self.exp_id:.20s}")

            model.requires_grad_(True)

            # 冻结模型的倒数 n 层
            layers_frozen: list[nn.Module] = []
            if self._layer_freeze_n > 0:
                dummy_input = torch.randn(1, *self._dataset_info.shape)
                layers_frozen = freeze_last_n_layers(
                    model, self._layer_freeze_n, dummy_input
                )

            print(f"Layers frozen: {layers_frozen}")

            optimizer = self._optimizer_class(
                model.parameters(),  # 会自动忽略不参与优化的参数
                lr=self._lr,
                **self._optimizer_params,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self._epochs
            )

            start_epoch = 0

            # 如果有 Checkpoints 则载入
            if self.has_checkpoints():
                ckpts = self.load_checkpoints()
                model_ckpt = ckpts["model_state_dict"]
                optimizer_ckpt = ckpts["optimizer_state_dict"]
                scheduler_ckpt = ckpts["scheduler_state_dict"]
                model.load_state_dict(model_ckpt)
                optimizer.load_state_dict(optimizer_ckpt)
                scheduler.load_state_dict(scheduler_ckpt)
                # 载入时间、轮数信息
                time_start = ckpts["time_start"]
                time_consumed_by_val = ckpts["time_consumed_by_val"]
                ckpt_save_time = ckpts["time_save"]
                time_consumed_by_val += time.time() - ckpt_save_time
                start_epoch = ckpts["current_epoch"]
                # 载入随机状态
                prev_random_states = ckpts["random_states"]
                load_random_states(prev_random_states)
                print(f"Loaded checkpoints from epoch {start_epoch}.")

            # 微调模型
            with tqdm(
                initial=start_epoch,
                total=self._epochs,
                desc=f"Model Tuning({self._exp_id:.20s})",
            ) as pbar:
                for epoch in range(start_epoch, self._epochs):
                    model.train()
                    recorder_loss = AverageLossRecorder()

                    for images, labels in self._poisoned_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)

                        loss = F.cross_entropy(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        recorder_loss.batch_update(loss, images.size(0))

                    scheduler.step()
                    pbar.set_postfix(
                        {
                            "Loss": f"{recorder_loss.avg_loss:.4f}",
                            "LR": f"{scheduler.get_last_lr()[0]:.4e}",
                        }
                    )
                    pbar.update(1)
                    time_val_start = time.perf_counter()
                    tb_writer.add_scalar(
                        "Train/Loss", recorder_loss.avg_loss, epoch + 1
                    )
                    tb_writer.add_scalar(
                        "Train/LR", scheduler.get_last_lr()[0], epoch + 1
                    )

                    if (epoch + 1) % self._make_test_per_epochs == 0 or (
                        epoch + 1
                    ) == self._epochs:
                        # 在验证集上测试
                        benign_acc = test_benign_accuracy(
                            model, self._val_loader, device
                        )
                        attack_success_rate = test_attack_success_rate(
                            model,
                            trigger_gen=self._trigger_gen,
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
                            "Tuning Data Samples (20)",
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
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "time_start": time_start,
                            "time_consumed_by_val": time_consumed_by_val,
                            "time_save": time.time(),
                            "current_epoch": epoch + 1,
                            "random_states": curr_random_states,
                        }
                        self.save_checkpoints(ckpts)
                        pbar.write(f"Checkpoints saved at epoch {epoch+1}.")

                    # 计算验证耗时
                    time_consumed_by_val += time.perf_counter() - time_val_start

            time_end = time.time()
            torch.save(model.state_dict(), self._model_save_path)
            self._tuned_model = model
            # 存储训练的一些信息以便于追溯
            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self._exp_id,
                    "exp_desc": self._exp_desc,
                    "tensorboard_log_id": tensorboard_log_id,
                    "normal_trainer_exp_id": self._normal_trainer.exp_id,
                    "trigger_gen_exp_id": self._trigger_gen.exp_id,
                    "data_poisoner_exp_id": self._data_poisoner.exp_id,
                    "layers_frozen": [str(layer) for layer in layers_frozen],
                    "params": {
                        "layer_freeze_n": self._layer_freeze_n,
                        "target_label": self._target_label,
                        "dataset_name": self._dataset_info.name,
                        "epochs": self._epochs,
                        "lr": self._lr,
                        "batch_size": self._batch_size,
                        "optimizer_class": self._optimizer_class.__name__,
                        "optimizer_params": self._optimizer_params,
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

        return self._tuned_model

    def get_model(self) -> nn.Module:
        """
        get_tuned_model 的别名
        """
        return self.get_tuned_model()
