import torchvision.transforms.v2 as transformsV2

from .abc import MakeTransforms
from configs import IMAGE_STANDARDIZE_MEANS


class MakeSimpleTransforms(MakeTransforms):
    """
    简单数据增强

    仅随机裁剪 + 翻转
    """

    def __init__(self, input_shape: tuple[int, int, int]):
        super().__init__(input_shape)
        self._input_shape = input_shape
        _, img_h, img_w = input_shape
        shorter_side = min(img_h, img_w)

        # 随机裁剪，填充 0
        self._random_crop_transform_fill_0 = transformsV2.RandomCrop(
            size=(img_h, img_w),
            padding=int(shorter_side * 0.125),  # 1/8
            fill=0,
        )
        # 随机裁剪，填充均值 (0.5)
        self._random_crop_transform_fill_mean = transformsV2.RandomCrop(
            size=(img_h, img_w),
            padding=int(shorter_side * 0.125),  # 1/8
            fill=IMAGE_STANDARDIZE_MEANS[0],
        )
        self._random_flip_transform = transformsV2.Compose(
            [
                transformsV2.RandomHorizontalFlip(),
                transformsV2.RandomVerticalFlip(),
            ]
        )
        # 组合成转换
        self._train_transforms = transformsV2.Compose(
            [
                self.normalize,  # 归一化到 [0, 1]
                self._random_flip_transform,  # 随机翻转
                self._random_crop_transform_fill_mean,  # 随机裁剪，填充均值
                self.standardize,  # 标准化，让训练更稳定 (均值标准化后正好是 0)
            ]
        )
        self._tensor_train_transforms = transformsV2.Compose(
            [
                self._random_flip_transform,
                # 这里因为数据已经被归一化标准化，直接填 0 即可
                self._random_crop_transform_fill_0,
            ]
        )

    @property
    def train_transforms(self) -> transformsV2.Compose:
        return self._train_transforms

    @property
    def val_transforms(self) -> transformsV2.Compose:
        return self.normalize_standardize

    @property
    def distill_transforms(self) -> transformsV2.Compose:
        # 蒸馏采用和训练集一样的变换
        return self.train_transforms

    @property
    def tensor_train_transforms(self) -> transformsV2.Compose:
        return self._tensor_train_transforms

    @property
    def tensor_val_transforms(self) -> transformsV2.Compose:
        return transformsV2.Compose([])

    @property
    def tensor_distill_transforms(self) -> transformsV2.Compose:
        return self.tensor_train_transforms

    @property
    def tensor_trigger_transforms(self) -> transformsV2.Compose:
        return self.tensor_train_transforms
