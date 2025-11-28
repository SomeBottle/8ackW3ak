"""MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.

Repository: https://github.com/kuangliu/pytorch-cifar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.abc import ModelBase


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(ModelBase):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10):
        """
        初始化 MobileNetV2模型

        :param num_classes: 类别数
        """
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(
            320, 1280, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    # feat=True 时还会返回分类层前未池化的特征
    def forward(self, x, feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        feature_l_2 = out
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feature_l_1 = out
        out = self.linear(out)
        if feat:
            return [feature_l_2, feature_l_1], out
        return out

    def feature_to_output(self, feature: torch.Tensor, feat_level: int) -> torch.Tensor:
        if feat_level == 1:
            out = self.linear(feature)
        elif feat_level == 2:
            out = F.avg_pool2d(feature, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            raise ValueError(f"Unsupported feat_level: {feat_level}")

        return out

    @property
    def classifier(self) -> nn.Module:
        return self.linear


def test():
    net = MobileNetV2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())


# test()
