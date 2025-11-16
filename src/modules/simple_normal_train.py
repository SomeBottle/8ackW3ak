"""
正常训练一个模型的模块
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Type
from torch.utils.data import DataLoader, Dataset
from configs import (
    CHECKPOINTS_SAVE_PATH,
    TENSORBOARD_LOGS_PATH,
)
from data_augs.abc import MakeTransforms
from utils.funcs import (
    auto_select_device,
    auto_num_workers,
    temp_seed,
    get_base_exp_info,
    test_benign_accuracy,
    print_section,
    json_serialize_helper,
    load_random_states,
    get_curr_random_states,
)

from utils.records import AverageLossRecorder
from utils.data import IndexedDataset, DatasetWithInfo, TransformedDataset

from modules.abc import NormalTrainer

_default_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "base_train")
_default_num_workers = auto_num_workers()


class SimpleNormalTrainer(NormalTrainer):

    def __init__(
        self,
        exp_id: str,
        model_class: Type[nn.Module],
        epochs: int,
        lr: float,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        batch_size: int,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_params: dict,
        exp_desc: str = "",
        make_test_per_epochs: int = 10,
        save_ckpts_per_epochs: int = 5,
        num_workers: int = _default_num_workers,
        seed: int = 42,
        save_dir: str = _default_save_dir,
    ):
        """
        初始化模型正常训练模块

        :param exp_id: 实验 ID
        :param model_class: 模型类
        :param epochs: 训练轮数
        :param lr: 学习率
        :param dataset_info: 数据集信息
        :param data_transform: 数据增强模块
        :param batch_size: 训练批大小
        :param optimizer_class: 优化器类
        :param optimizer_params: 优化器参数
        :param exp_desc: 实验描述
        :param make_test_per_epochs: 每隔多少轮在验证集上测试一次
        :param save_ckpts_per_epochs: 每隔多少轮保存一次 Checkpoints
        :param num_workers: DataLoader 的 num_workers 参数
        :param seed: 随机种子
        :param save_dir: 模型保存路径
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
        model_save_dir = os.path.join(
            save_dir,
            exp_id,
        )
        os.makedirs(model_save_dir, exist_ok=True)

        self._model_save_path = os.path.join(model_save_dir, "normal_model.pth")
        self.set_exp_save_dir(model_save_dir)
        self._data_transform_class = data_transform_class
        self._exp_id = exp_id
        self._seed = seed
        self._model_class = model_class
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._make_test_per_epochs = make_test_per_epochs
        self._save_ckpts_per_epochs = save_ckpts_per_epochs
        self._dataset_info = dataset_info
        self._train_set = train_tensor_set
        self._val_set = val_tensor_set
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params
        self._exp_desc = exp_desc
        self._train_loader = DataLoader(
            IndexedDataset(dataset=train_tensor_set),
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
        self._trained_model = None

    @property
    def model_class(self) -> Type[nn.Module]:
        return self._model_class

    @property
    def exp_id(self) -> str:
        return self._exp_id

    def get_trained_model(self) -> nn.Module:
        """
        训练模型并返回训练好的模型

        :return: 训练好的模型
        """

        if self._trained_model is not None:
            return self._trained_model

        with temp_seed(self._seed):
            device = auto_select_device()  # 自动选择空闲显存最大的设备
            model = self._model_class(num_classes=self._dataset_info.num_classes).to(
                device
            )
            model.to(device)

            if os.path.exists(self._model_save_path):
                # 如果模型已经存在，直接载入返回
                model.load_state_dict(
                    torch.load(self._model_save_path, map_location=device)
                )
                self._trained_model = model
                return model

            print_section(f"Simple Normal Training: {self.exp_id:.20s}")

            # 开始时间
            time_start = time.time()

            # 进行验证时耗费的时间
            time_consumed_by_val = 0.0

            # TensorBoard 记录器
            tensorboard_log_id = f"normal_train_{self._exp_id}"
            tensorboard_log_dir = os.path.join(
                TENSORBOARD_LOGS_PATH, tensorboard_log_id
            )
            tb_writer = SummaryWriter(
                log_dir=tensorboard_log_dir, comment=self._exp_desc
            )

            # 模型训练
            model.requires_grad_(True)

            optimizer = self._optimizer_class(
                model.parameters(), lr=self._lr, **self._optimizer_params
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self._epochs,
            )

            num_train_samples = len(self._train_set)
            # 每个下标样本的遗忘次数
            forgetting_counts = torch.zeros(
                num_train_samples, dtype=torch.int32, device=device
            )
            # 每个样本上一次被采样时的分类状态，0 为未分类正确，1 为分类正确
            prev_predictions = torch.zeros(
                num_train_samples, dtype=torch.bool, device=device
            )

            start_epoch = 0

            # 如果有 Checkpoints 则载入
            if self.has_checkpoints():
                ckpts = self.load_checkpoints()
                # 载入之前的状态
                model_ckpt = ckpts["model_state_dict"]
                optimizer_ckpt = ckpts["optimizer_state_dict"]
                scheduler_ckpt = ckpts["scheduler_state_dict"]
                forgetting_ckpts = ckpts["forgetting_events"]
                model.load_state_dict(model_ckpt)
                optimizer.load_state_dict(optimizer_ckpt)
                scheduler.load_state_dict(scheduler_ckpt)
                forgetting_counts.copy_(forgetting_ckpts["forgetting_counts"])
                prev_predictions.copy_(forgetting_ckpts["prev_predictions"])
                # 载入时间，轮数信息
                time_start = ckpts["time_start"]
                time_consumed_by_val = ckpts["time_consumed_by_val"]
                ckpt_save_time = ckpts["time_save"]
                time_consumed_by_val += time.time() - ckpt_save_time
                start_epoch = ckpts["current_epoch"]
                # 载入随机状态
                prev_random_state = ckpts["random_state"]
                load_random_states(prev_random_state)
                print(f"Loaded checkpoints from epoch {start_epoch}.")

            with tqdm(
                initial=start_epoch,
                total=self._epochs,
                desc=f"Training Normal({self._exp_id:.20s})",
            ) as pbar:
                for epoch in range(start_epoch, self._epochs):
                    model.train()
                    recorder_loss = AverageLossRecorder()

                    for images, labels, indexes in self._train_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)
                        indexes: torch.Tensor = indexes.to(device)

                        outputs: torch.Tensor = model(images)
                        predicted = outputs.detach().argmax(dim=-1)
                        correctnesses = predicted == labels

                        loss = F.cross_entropy(outputs, labels)

                        if epoch > 0:
                            # 从第二轮开始记录遗忘事件
                            is_forgotten = (prev_predictions[indexes]) & (
                                ~correctnesses
                            )
                            forgetting_counts[indexes[is_forgotten]] += 1

                        prev_predictions[indexes] = correctnesses

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
                        val_acc = test_benign_accuracy(
                            model=model, data_loader=self._val_loader, device=device
                        )
                        pbar.write(
                            f"Epoch {epoch + 1}/{self._epochs}, Validation Accuracy: {val_acc:.3%}"
                        )
                        tb_writer.add_scalar("Val/Accuracy", val_acc, epoch + 1)

                    if (epoch + 1) % self._save_ckpts_per_epochs == 0 and (
                        epoch + 1
                    ) < self._epochs:
                        # 保存 Checkpoints
                        ckpts = {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "forgetting_events": {
                                "forgetting_counts": forgetting_counts.cpu(),
                                "prev_predictions": prev_predictions.cpu(),
                            },
                            "time_start": time_start,
                            "time_consumed_by_val": time_consumed_by_val,
                            "time_save": time.time(),
                            "current_epoch": epoch + 1,
                            "random_state": get_curr_random_states(),
                        }
                        self.save_checkpoints(ckpts)
                        pbar.write(f"Checkpoints saved at epoch {epoch+1}.")

                    # 计算验证耗时
                    time_consumed_by_val += time.perf_counter() - time_val_start

            time_end = time.time()
            # 保存模型和遗忘事件
            torch.save(model.state_dict(), self._model_save_path)
            self._trained_model = model
            # 存储训练的一些信息以便于追溯
            exp_info = get_base_exp_info()
            exp_info.update(
                {
                    "exp_id": self._exp_id,
                    "exp_desc": self._exp_desc,
                    "tensorboard_log_id": tensorboard_log_id,
                    "forgetting_counts": forgetting_counts.cpu(),
                    "params": {
                        "data_transform_class": self._data_transform_class.__name__,
                        "seed": self._seed,
                        "model_class": self._model_class.__name__,
                        "epochs": self._epochs,
                        "lr": self._lr,
                        "dataset_name": self._dataset_info.name,
                        "batch_size": self._batch_size,
                        "optimizer_class": self._optimizer_class.__name__,
                        "optimizer_params": self._optimizer_params,
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
        return self._trained_model

    def get_forgetting_counts(self):
        try:
            exp_info = self.get_exp_info()
        except FileNotFoundError:
            # 没有实验信息，训练模型
            self.get_trained_model()
            return self.get_forgetting_counts()

        if "forgetting_counts" not in exp_info:
            raise ValueError("Unexpected: forgetting_counts not found in exp_info.")
        return exp_info["forgetting_counts"].cpu()

    def get_model(self) -> nn.Module:
        """
        get_trained_model 的别名
        """
        return self.get_trained_model()
