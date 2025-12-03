"""
所有模型的基类
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ModelBase(nn.Module, ABC):
    @property
    @abstractmethod
    def classifier(self) -> nn.Module:
        """
        分类器模块

        :return: 分类器模块
        """
        pass

    @abstractmethod
    def forward(
        self, x, feat=False
    ) -> torch.Tensor | tuple[list[torch.Tensor], torch.Tensor]:
        """
        前向传播

        :param x: 输入张量，形状为 (B, 3, H, W)
        :param feat: 是否返回特征
        :return: 如果 feat 为 False，返回分类结果张量，形状为 (B, num_classes)；
                 如果 feat 为 True，返回一个元组 (features, out)，
                 其中 features 为特征列表，最后一个元素是倒数第一个特征，倒数第二个元素是倒数第二个特征，
                    通常倒数第一个特征的形状为 (B, flat_dim)，倒数第二个特征的形状在 CNN 和 ViT 这种架构差异极大的模型中有所不同;
                    out 为分类结果张量，形状为 (B, num_classes)。
        """
        pass

    @abstractmethod
    def feature_to_output(self, feature: torch.Tensor, feat_level: int) -> torch.Tensor:
        """
        将指定特征映射到分类结果

        :param feature: 特征张量
        :param feat_level: 特征层级，1 表示倒数第一个特征，2 表示倒数第二个特征，依此类推
        :return: 分类结果张量，形状为 (B, num_classes)
        """
        pass

    @abstractmethod
    def get_gradcam_feature_layer(self) -> nn.Module:
        """
        获取用于 Grad-CAM 可视化的特征层

        :return: 特征层模块
        """
        pass
