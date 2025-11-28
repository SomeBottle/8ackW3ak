"""VGG 11 / 13 / 16 / 19 in PyTorch.

* for 224x224 images

Repository: https://github.com/kuangliu/pytorch-cifar
"""

import torch
import torch.nn as nn

from models.abc import ModelBase

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG_BN(ModelBase):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG_BN, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x, feat=False):
        feature = self.features(x)
        feature_l_2 = feature
        out = self.avg_pool(feature)
        out = out.view(out.size(0), -1)
        feature_l_1 = out
        out = self._classifier(out)
        if feat:
            return [feature_l_2, feature_l_1], out
        return out

    def feature_to_output(self, feature: torch.Tensor, feat_level: int) -> torch.Tensor:
        if feat_level == 1:
            out = self._classifier(feature)
        elif feat_level == 2:
            out = self.avg_pool(feature)
            out = out.view(out.size(0), -1)
            out = self._classifier(out)
        else:
            raise ValueError(f"Unsupported feat_level: {feat_level}")
        return out

    @property
    def classifier(self) -> nn.Module:
        return self._classifier


class VGG16(VGG_BN):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__("VGG16", num_classes=num_classes)


class VGG19(VGG_BN):
    def __init__(self, num_classes=10):
        super(VGG19, self).__init__("VGG19", num_classes=num_classes)


def test():
    net = VGG_BN("VGG11")
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
