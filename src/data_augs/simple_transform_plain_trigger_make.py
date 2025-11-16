import torchvision.transforms.v2 as transformsV2

from .simple_transform_make import MakeSimpleTransforms


class MakeSimpleTransformsPlainTrigger(MakeSimpleTransforms):
    """
    简单数据增强

    仅随机裁剪 + 翻转，**触发器不会有任何数据增强**
    """

    def __init__(self, input_shape: tuple[int, int, int]):
        super().__init__(input_shape)

    @property
    def tensor_trigger_transforms(self) -> transformsV2.Compose:
        # 恒等映射，什么都不做
        return transformsV2.Compose([transformsV2.Lambda(lambda x: x)])
