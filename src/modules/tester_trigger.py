"""
触发器可视化测试模块

一方面对触发器进行可视化，另一方面利用 LPIPS 指标评判触发器的隐蔽性
"""

import os
import json
import torch
import torchvision.transforms.v2 as T

from lpips import LPIPS
from utils.funcs import auto_select_device, temp_seed
from modules.abc import TesterBase, TriggerGenerator
from typing import Type
from data_augs.abc import MakeTransforms
from utils.data import DatasetWithInfo, TransformedDataset
from torchvision.utils import save_image


class TriggerTester(TesterBase):
    """
    触发器可视化 & 测试模块
    """

    def __init__(
        self,
        dataset_info: DatasetWithInfo,
        data_transform_class: Type[MakeTransforms],
        trigger_gen: TriggerGenerator,
        vis_save_dir: str,
        num_samples: int = 5,
        seed: int = 42,
    ):
        """
        构建触发器可视化测试模块

        :param dataset_info: 数据集信息
        :param data_transform_class: 数据增强类
        :param trigger_gen: 触发器生成器
        :param vis_save_dir: 可视化结果保存目录
        :param num_samples: 用于可视化和评估的样本数量
        :param seed: 随机种子
        """
        super().__init__()
        # -------------------------------- 数据增强
        data_transform = data_transform_class(input_shape=dataset_info.shape)

        transformed_test_set = TransformedDataset(
            dataset=dataset_info.test_set, transform=data_transform.val_transforms
        )

        self._trigger_gen = trigger_gen
        self._seed = seed
        self._num_samples = num_samples
        self._dataset_info = dataset_info
        self._dataset = transformed_test_set
        self._vis_save_dir = vis_save_dir
        self._data_transform = data_transform
        self._lpips_criterion = LPIPS(net="alex")

        os.makedirs(self._vis_save_dir, exist_ok=True)

    def test(self, model=None, device=None) -> dict:
        """
        随机选取图像进行触发器叠加测试

        :param model: 模型，此处没有用到，仅为了符合 TesterBase 接口
        :param device: 运行设备，如果为 None 则自动选择
        :return: LPIPS 测试结果 {chosen_indices: list[int], lpips_values: dict[int, float]}
        """
        if device is None:
            device = auto_select_device()

        # 随机选取样本
        with temp_seed(self._seed):
            indices = torch.randperm(len(self._dataset))[: self._num_samples]
            sampled_data = [self._dataset[i][0] for i in indices]
            sampled_batch = torch.stack(sampled_data).to(
                device
            )  # shape (num_samples, C, H, W)

        # 叠加了触发器的样本
        triggered_batch = self._trigger_gen.apply_trigger(sampled_batch)

        # 均值图像，用作对比
        mean_image = torch.zeros(1, *self._dataset_info.shape).to(device)
        triggered_mean_image = self._trigger_gen.apply_trigger(mean_image)

        # 上采样 sampled_batch 和 triggered_batch 到 224x224
        # 我们要对比的是放大后图像的异常程度，人类观察者肯定也是放大观察的。
        resize_transform = T.Resize(
            (224, 224), interpolation=T.InterpolationMode.NEAREST
        )

        sampled_batch = resize_transform(sampled_batch)
        triggered_batch = resize_transform(triggered_batch)
        triggered_mean_image = resize_transform(triggered_mean_image)

        # 计算 LPIPS 指标
        self._lpips_criterion.to(device)
        lpips_values: torch.Tensor = self._lpips_criterion(
            triggered_batch, sampled_batch
        )
        lpips_values = lpips_values.detach().squeeze(
            dim=[1, 2, 3]
        )  # shape (num_samples,)
        lpips_values_list: list[float] = lpips_values.cpu().tolist()

        # 把图像逆标准化到 [0, 1] 范围以便保存
        destandardized_originals = self._data_transform.destandardize(sampled_batch)
        destandardized_triggered = self._data_transform.destandardize(triggered_batch)
        destandardized_triggered_mean = self._data_transform.destandardize(
            triggered_mean_image
        )

        for i, idx in enumerate(indices):
            idx = idx.item()
            save_image(
                destandardized_originals[i],
                os.path.join(self._vis_save_dir, f"original_{idx}.png"),
            )
            save_image(
                destandardized_triggered[i],
                os.path.join(self._vis_save_dir, f"triggered_{idx}.png"),
            )

        # 保存触发器图像
        save_image(
            destandardized_triggered_mean.squeeze(0),
            os.path.join(self._vis_save_dir, "trigger_only.png"),
        )

        chosen_indices = indices.tolist()
        lpips_result = {
            idx: lpips_values_list[i] for i, idx in enumerate(chosen_indices)
        }
        result = {"chosen_indices": chosen_indices, "lpips_values": lpips_result}

        # 结果和图像一起保存
        with open(os.path.join(self._vis_save_dir, "lpips_results.json"), "w") as f:
            json.dump(result, f, indent=4)

        return result
