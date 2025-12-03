"""DenseNet in PyTorch.

Ref: https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.abc import ModelBase


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(ModelBase):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, feat=False):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        feature_l_2 = out  # 倒数第二个特征
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        feature_l_1 = out  # 倒数第一个特征
        out = self.linear(out)
        if feat:
            return [feature_l_2, feature_l_1], out
        return out

    def feature_to_output(self, feature: torch.Tensor, feat_level: int) -> torch.Tensor:
        if feat_level == 1:
            # 倒数第一个特征
            out = self.linear(feature)
        elif feat_level == 2:
            # 倒数第二个特征
            out = F.avg_pool2d(F.relu(self.bn(feature)), 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            raise ValueError(f"Unsupported feat_level: {feat_level}")
        return out

    def get_gradcam_feature_layer(self) -> nn.Module:
        return self.dense4[-1]

    @property
    def classifier(self) -> nn.Module:
        return self.linear


class DenseNetBC121(DenseNet):
    def __init__(self, num_classes=10):
        super(DenseNetBC121, self).__init__(
            Bottleneck,
            [6, 12, 24, 16],
            growth_rate=32,
            reduction=0.5,
            num_classes=num_classes,
        )


class DenseNetBC169(DenseNet):
    def __init__(self, num_classes=10):
        super(DenseNetBC169, self).__init__(
            Bottleneck,
            [6, 12, 32, 32],
            growth_rate=32,
            reduction=0.5,
            num_classes=num_classes,
        )


class DenseNetBC201(DenseNet):
    def __init__(self, num_classes=10):
        super(DenseNetBC201, self).__init__(
            Bottleneck,
            [6, 12, 48, 32],
            growth_rate=32,
            reduction=0.5,
            num_classes=num_classes,
        )


class DenseNetBC161(DenseNet):
    def __init__(self, num_classes=10):
        super(DenseNetBC161, self).__init__(
            Bottleneck,
            [6, 12, 36, 24],
            growth_rate=48,
            reduction=0.5,
            num_classes=num_classes,
        )
