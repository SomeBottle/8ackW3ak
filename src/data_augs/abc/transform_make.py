"""
根据数据尺寸生成数据增强的模块抽象类
"""

import torch
import torchvision.transforms.v2 as transformsV2

from abc import ABC, abstractmethod
from configs import IMAGE_STANDARDIZE_MEANS, IMAGE_STANDARDIZE_STDS


class MakeTransforms(ABC):

    _normalize = transformsV2.Compose(
        [
            transformsV2.ToImage(),  # PIL.Image -> Tensor，变换维度顺序
            transformsV2.ToDtype(torch.float32, scale=True),  # 归一化到 [0, 1]
        ]
    )
    _standardize = transformsV2.Compose(
        [
            transformsV2.Normalize(
                mean=IMAGE_STANDARDIZE_MEANS, std=IMAGE_STANDARDIZE_STDS
            )
        ]
    )
    _destandardize = transformsV2.Compose(
        [
            transformsV2.Normalize(
                mean=[
                    -m / s
                    for m, s in zip(IMAGE_STANDARDIZE_MEANS, IMAGE_STANDARDIZE_STDS)
                ],
                std=[1 / s for s in IMAGE_STANDARDIZE_STDS],
            )
        ]
    )
    _normalize_standardize = transformsV2.Compose([_normalize, _standardize])

    @abstractmethod
    def __init__(self, input_shape: tuple[int, int, int]):
        """
        初始化数据增强模块

        :param input_shape: 输入图片的尺寸，格式为 (C, H, W)
        """
        pass

    @property
    @abstractmethod
    def train_transforms(self) -> transformsV2.Compose:
        """
        获取训练集的数据增强
        """
        pass

    @property
    @abstractmethod
    def val_transforms(self) -> transformsV2.Compose:
        """
        获取验证集的数据增强
        """
        pass

    @property
    @abstractmethod
    def distill_transforms(self) -> transformsV2.Compose:
        """
        获取蒸馏集的数据增强
        """
        pass

    @property
    @abstractmethod
    def tensor_train_transforms(self) -> transformsV2.Compose:
        """
        获取针对 Tensor 训练集的数据增强 (图像已经标准化到 [-1, 1])
        """
        pass

    @property
    @abstractmethod
    def tensor_val_transforms(self) -> transformsV2.Compose:
        """
        获取针对 Tensor 验证集的数据增强 (图像已经标准化到 [-1, 1])
        """
        pass

    @property
    @abstractmethod
    def tensor_distill_transforms(self) -> transformsV2.Compose:
        """
        获取针对 Tensor 蒸馏集的数据增强 (图像已经标准化到 [-1, 1])
        """
        pass

    @property
    @abstractmethod
    def tensor_trigger_transforms(self) -> transformsV2.Compose:
        """
        获取针对触发器的数据增强，触发器肯定是 Tensor (已经标准化到 [-1, 1])
        """
        pass

    @property
    def normalize(self) -> transformsV2.Compose:
        """
        获取图像归一化的变换
        """
        return self._normalize

    @property
    def standardize(self) -> transformsV2.Compose:
        """
        获取图像标准化的变换
        """
        return self._standardize

    @property
    def destandardize(self) -> transformsV2.Compose:
        """
        获取图像逆标准化的变换
        """
        return self._destandardize

    @property
    def normalize_standardize(self) -> transformsV2.Compose:
        """
        获取图像归一化和标准化的变换
        """
        return self._normalize_standardize
