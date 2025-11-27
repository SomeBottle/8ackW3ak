"""
NAD 缓解方法实现

* Repo: https://github.com/bboylyg/NAD/
* Ref: Li Y, Lyu X, Koren N, et al. Neural attention distillation: Erasing backdoor triggers from deep neural networks[J]. arXiv preprint arXiv:2101.05930, 2021.
"""

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from utils.data import DatasetWithInfo, DataLoaderDataIter, TransformedDataset
from utils.data_funcs import balanced_split_into_two
from data_augs import MakeSimpleTransforms
from utils.funcs import auto_select_device, temp_seed, print_section, auto_num_workers

from defense_modules.abc import DefenseModule
from configs import TENSORBOARD_LOGS_PATH, CHECKPOINTS_SAVE_PATH

_ckpt_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "nad")


class AT(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    """

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        # 计算学生和教师注意力图谱的 MSE 损失
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        # 根据论文，注意力图谱是激活值 p 次方的绝对值之和
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am


class NAD(DefenseModule):
    def __init__(
        self,
        test_id: str,
        model: nn.Module,
        dataset_info: DatasetWithInfo,
        *args,
        epochs_teacher: int = 10,
        epochs_student: int = 20,
        batch_size: int = 64,
        lr: float = 0.01,
        lr_factor: float = 0.1,
        lr_decay_epochs: list[int] = [2, 4, 6, 8],
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        p_at: int = 2,
        data_portion: float = 0.05,
        betas: list[float] = [],
        target_layers: list[str] = [],
        seed=42,
        **kwargs,
    ):
        """
        初始化 NAD 缓解模块

        :param test_id: 用来标记本次 NAD 运行的测试 ID
        :param model: 待缓解模型
        :param dataset_info: 数据集信息
        :param epochs_teacher: 教师模型训练轮数
        :param epochs_student: 学生模型训练轮数
        :param batch_size: 训练批次大小
        :param lr: 学习率
        :param lr_factor: 学习率衰减因子
        :param lr_decay_epochs: 学习率衰减的轮数列表
        :param momentum: 动量因子
        :param weight_decay: 权重衰减
        :param p_at: 注意力转移损失 AT 的 p 超参
        :param data_portion: 用于注意力蒸馏的数据比例
        :param betas: 各层注意力蒸馏损失权重列表
        :param target_layers: 需要进行注意力蒸馏的层名称列表 (必须和 betas 列表长度一致)
        :param seed: 随机种子
        :raise ValueError: 如果 betas 和 target_layers 长度不一致则抛出该异常
        """
        if len(betas) != len(target_layers):
            raise ValueError("Length of betas and target_layers must be the same.")
        self._test_id = test_id
        self._model = copy.deepcopy(model)  # 不影响原模型
        self._transforms_maker = MakeSimpleTransforms(input_shape=dataset_info.shape)
        self._dataset_info = dataset_info
        self._epochs_teacher = epochs_teacher
        self._epochs_student = epochs_student
        self._batch_size = batch_size
        self._lr = lr
        self._lr_factor = lr_factor
        self._lr_decay_epochs = set(lr_decay_epochs)
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._p_at = p_at
        self._betas = betas
        self._data_portion = data_portion
        self._target_layers = set(target_layers)
        self._save_dir = os.path.join(_ckpt_save_dir, test_id)
        self._seed = seed

        # 随机选取指定比例的数据用于注意力蒸馏
        _, sub_train_set = balanced_split_into_two(
            dataset=dataset_info.train_set,
            latter_size_or_ratio=data_portion,
            random_state=seed,
        )

        self._data_loader = DataLoader(
            dataset=TransformedDataset(
                sub_train_set, transform=self._transforms_maker.train_transforms
            ),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=auto_num_workers(),
            pin_memory=True,
        )
        os.makedirs(self._save_dir, exist_ok=True)

    @classmethod
    def is_mitigation(cls) -> bool:
        return True

    def _adjust_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        """
        根据预设的衰减策略调整学习率

        :param optimizer: 优化器
        :param epoch: 当前训练轮数
        """
        if epoch in self._lr_decay_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self._lr_factor

    def mitigate(self) -> nn.Module:
        """
        执行 NAD 缓解

        :return: 缓解后的模型
        :raise RuntimeError: 如果钩子未能正确捕获指定层的输出则抛出该异常
        """
        device = auto_select_device()

        # 最终缓解的模型的保存路径
        model_save_path = os.path.join(self._save_dir, "nad_mitigated_model.pth")

        if os.path.exists(model_save_path):
            print(f"Found existing mitigated model at {model_save_path}, loading it...")
            mitigated_model = self._model
            mitigated_model.load_state_dict(
                torch.load(model_save_path, map_location=device)
            )
            return mitigated_model

        print_section(f"NAD Defense: {self._test_id}")

        tensorboard_log_id = f"nad_{self._test_id}"
        tensorboard_log_dir = os.path.join(TENSORBOARD_LOGS_PATH, tensorboard_log_id)
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        with temp_seed(self._seed):
            # -------------------- STAGE 1: 微调教师模型 --------------------
            teacher_model = copy.deepcopy(self._model)
            teacher_model.to(device)
            teacher_model.requires_grad_(True)
            teacher_model.train()

            teacher_optimizer = torch.optim.SGD(
                teacher_model.parameters(),
                lr=self._lr,
                momentum=self._momentum,
                weight_decay=self._weight_decay,
            )

            with tqdm(
                range(self._epochs_teacher), desc="NAD Stage 1 - Tune Teacher"
            ) as pbar:
                for epoch in range(self._epochs_teacher):
                    self._adjust_learning_rate(teacher_optimizer, epoch)
                    epoch_loss = 0.0
                    for images, labels in self._data_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        teacher_optimizer.zero_grad()
                        outputs = teacher_model(images)
                        loss = F.cross_entropy(outputs, labels)
                        loss.backward()
                        teacher_optimizer.step()

                        epoch_loss += loss.item()

                    avg_epoch_loss = epoch_loss / len(self._data_loader)
                    tb_writer.add_scalar("Stage1/Teacher_Loss", avg_epoch_loss, epoch)
                    tb_writer.add_scalar(
                        "Stage1/Teacher_LR",
                        teacher_optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    pbar.set_postfix(
                        {
                            "Avg Loss": f"{avg_epoch_loss:.4f}",
                            "LR": f"{teacher_optimizer.param_groups[0]['lr']:.6f}",
                        }
                    )
                    pbar.update(1)

            # 教师模型微调完成后冻结
            teacher_model.requires_grad_(False)
            teacher_model.eval()

            # -------------------- STAGE 2: 训练学生模型 --------------------
            student_model = self._model
            student_model.to(device)
            student_model.requires_grad_(True)
            student_model.train()

            criterion_AT = AT(p=self._p_at)

            student_optimizer = torch.optim.SGD(
                student_model.parameters(),
                lr=self._lr,
                momentum=self._momentum,
                weight_decay=self._weight_decay,
            )

            with tqdm(
                range(self._epochs_student), desc="NAD Stage 2 - Distill Student"
            ) as pbar:
                for epoch in range(self._epochs_student):
                    self._adjust_learning_rate(student_optimizer, epoch)
                    epoch_loss = 0.0
                    for images, labels in self._data_loader:
                        images: torch.Tensor = images.to(device)
                        labels: torch.Tensor = labels.to(device)

                        # 这里要拦截指定的层的输出
                        teacher_outputs: list[torch.Tensor] = []
                        student_outputs: list[torch.Tensor] = []

                        def _teacher_hook(module, input, output):
                            teacher_outputs.append(output)

                        def _student_hook(module, input, output):
                            student_outputs.append(output)

                        hook_list = []

                        for name, module in teacher_model.named_modules():
                            # 只给叶子模块注册 hook
                            if (list(module.children())) > 0:
                                continue
                            if name in self._target_layers:
                                hook = module.register_forward_hook(_teacher_hook)
                                hook_list.append(hook)

                        for name, module in student_model.named_modules():
                            # 只给叶子模块注册 hook
                            if (list(module.children())) > 0:
                                continue
                            if name in self._target_layers:
                                hook = module.register_forward_hook(_student_hook)
                                hook_list.append(hook)

                        if (
                            len(teacher_outputs) != len(student_outputs)
                            or len(teacher_outputs) == 0
                            or len(student_outputs) == 0
                        ):
                            raise RuntimeError(
                                "Hooks did not capture the correct number of layer outputs."
                            )

                        outputs_student = student_model(images)
                        _ = teacher_model(images)

                        for hook in hook_list:
                            hook.remove()

                        loss = F.cross_entropy(outputs_student, labels)
                        for idx in range(len(self._betas)):
                            loss_at = criterion_AT(
                                student_outputs[idx], teacher_outputs[idx]
                            )
                            loss += self._betas[idx] * loss_at

                        student_optimizer.zero_grad()
                        loss.backward()
                        student_optimizer.step()
                        epoch_loss += loss.item()

                    avg_epoch_loss = epoch_loss / len(self._data_loader)
                    tb_writer.add_scalar("Stage2/Student_Loss", avg_epoch_loss, epoch)
                    tb_writer.add_scalar(
                        "Stage2/Student_LR",
                        student_optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    pbar.set_postfix(
                        {
                            "Avg Loss": f"{avg_epoch_loss:.4f}",
                            "LR": f"{student_optimizer.param_groups[0]['lr']:.6f}",
                        }
                    )
                    pbar.update(1)

        torch.save(student_model.state_dict(), model_save_path)

        tb_writer.close()

        return student_model
