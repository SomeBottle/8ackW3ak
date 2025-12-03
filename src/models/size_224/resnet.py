"""
ResNet v1 for ImageNet

Input: 3x224x224

Repository: https://github.com/aaron-xichen/pytorch-playground
"""

import torch.nn as nn
import math

from collections import OrderedDict
from models.abc import ModelBase

__all__ = ["ResNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m["conv1"] = conv3x3(inplanes, planes, stride)
        m["bn1"] = nn.BatchNorm2d(planes)
        m["relu1"] = nn.ReLU(inplace=True)
        m["conv2"] = conv3x3(planes, planes)
        m["bn2"] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m["conv1"] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m["bn1"] = nn.BatchNorm2d(planes)
        m["relu1"] = nn.ReLU(inplace=True)
        m["conv2"] = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        m["bn2"] = nn.BatchNorm2d(planes)
        m["relu2"] = nn.ReLU(inplace=True)
        m["conv3"] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m["bn3"] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


class ResNet(ModelBase):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m["conv1"] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m["bn1"] = nn.BatchNorm2d(64)
        m["relu1"] = nn.ReLU(inplace=True)
        m["maxpool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))

        self.group2 = nn.Sequential(
            OrderedDict([("fc", nn.Linear(512 * block.expansion, num_classes))])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feat=False):
        x = self.group1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature_l_2 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feature_l_1 = x
        x = self.group2(x)

        if feat:
            return [feature_l_2, feature_l_1], x
        return x

    def feature_to_output(self, feature, feat_level: int):
        if feat_level == 1:
            out = self.group2(feature)
        elif feat_level == 2:
            out = self.avgpool(feature)
            out = out.view(out.size(0), -1)
            out = self.group2(out)
        else:
            raise ValueError(f"Unsupported feat_level: {feat_level}")
        return out

    def get_gradcam_feature_layer(self) -> nn.Module:
        return self.layer4[-1]

    @property
    def classifier(self) -> nn.Module:
        return self.group2.fc


class ResNet14(ResNet):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet14 模型

        :param num_classes: 类别数
        """
        super(ResNet14, self).__init__(
            BasicBlock, [2, 2, 1, 1], num_classes=num_classes
        )


class ResNet18(ResNet):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet18 模型

        :param num_classes: 类别数
        """
        super(ResNet18, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes
        )


class ResNet26(ResNet):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet26 模型

        :param num_classes: 类别数
        """
        super(ResNet26, self).__init__(
            BasicBlock, [3, 3, 3, 3], num_classes=num_classes
        )


class ResNet34(ResNet):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet34 模型

        :param num_classes: 类别数
        """
        super(ResNet34, self).__init__(
            BasicBlock, [3, 4, 6, 3], num_classes=num_classes
        )


class ResNet50(ResNet):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet50 模型

        :param num_classes: 类别数
        """
        super(ResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes
        )


class ResNet101(ResNet):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet101 模型

        :param num_classes: 类别数
        """
        super(ResNet101, self).__init__(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes
        )


class ResNet152(ResNet):
    def __init__(self, num_classes=10):
        """
        初始化 ResNet152 模型

        :param num_classes: 类别数
        """
        super(ResNet152, self).__init__(
            Bottleneck, [3, 8, 36, 3], num_classes=num_classes
        )
