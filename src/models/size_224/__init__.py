"""适用于输入尺寸为 224x224 的模型"""

from .densenet import DenseNetBC121
from .resnet import ResNet18, ResNet26, ResNet34
from .shufflenetv2 import ShuffleNetV2_20x
from .mobilenetv2 import MobileNetV2
from .vgg import VGG16, VGG19
from .convnextv2 import ConvNeXtV2_Tiny
