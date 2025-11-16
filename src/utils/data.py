"""
自定义 Dataset / DataLoader
以及其他的一些数据集相关方法
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import random

from torch.utils.data import Dataset, DataLoader
from data_augs.abc import MakeTransforms
from typing import Iterable, Type

from utils.data_funcs import balanced_split_into_two
from configs import DATASET_INFOS


class DatasetWithInfo:
    """
    数据集信息类

    用于存储数据集及其信息，方便后续使用
    """

    @classmethod
    def from_name(cls, dataset_name: str):
        """
        获取指定名字的数据集

        :param dataset_name: 数据集名，大小写不敏感
        :raises ValueError: 数据集不支持（没有在 DATASET_INFOS 中定义）
        """
        dataset_name = dataset_name.lower()
        dataset_info = DATASET_INFOS.get(dataset_name)
        if dataset_info is None:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
        return DatasetWithInfo(
            name=dataset_name,
            shape=dataset_info["shape"],
            num_classes=dataset_info["num_classes"],
            train_set=dataset_info["train_set"],
            val_set=dataset_info["val_set"],
            test_set=dataset_info["test_set"],
        )

    @classmethod
    def all_names(cls) -> list[str]:
        """
        获取所有支持的数据集名字

        :return: 数据集名字列表
        """
        return list(DATASET_INFOS.keys())

    @classmethod
    def split_into_two(
        cls,
        dataset_info: "DatasetWithInfo",
        split_val: bool = False,
        split_test: bool = False,
        seed: int | None = None,
    ) -> tuple["DatasetWithInfo", "DatasetWithInfo"]:
        """
        均衡地(按类别)随机划分数据集 DatasetWithInfo 为 2 份

        划分时保证每个类别的样本尽量被均匀地分配到每一份中，如果原数据集类别间样本数平衡，这样划分得到的数据集也是平衡的。

        :param dataset_info: 数据集信息
        :param split_val: 是否划分验证集
        :param split_test: 是否划分测试集
        :param seed: 划分的随机种子
        :return: 划分后的两个数据集信息
        """
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
        train_set = dataset_info.train_set
        val_set = dataset_info.val_set
        test_set = dataset_info.test_set

        split_train_sets = balanced_split_into_two(
            dataset=train_set,
            latter_size_or_ratio=0.5,  # 划分为两部分
            random_state=seed,
        )
        if split_val:
            split_val_sets = balanced_split_into_two(
                dataset=val_set,
                latter_size_or_ratio=0.5,  # 划分为两部分
                random_state=seed,
            )
        if split_test:
            split_test_sets = balanced_split_into_two(
                dataset=test_set,
                latter_size_or_ratio=0.5,  # 划分为两部分
                random_state=seed,
            )

        split_dataset_infos = []
        for p in range(2):
            sub_dataset_info = DatasetWithInfo(
                name=dataset_info.name + f"_part{p+1}",
                shape=dataset_info.shape,
                num_classes=dataset_info.num_classes,
                train_set=split_train_sets[p],
                val_set=split_val_sets[p] if split_val else val_set,
                test_set=split_test_sets[p] if split_test else test_set,
            )
            split_dataset_infos.append(sub_dataset_info)

        return tuple(split_dataset_infos)

    def __init__(
        self,
        name: str,
        shape: tuple,
        num_classes: int,
        train_set: Dataset,
        val_set: Dataset,
        test_set: Dataset,
    ):
        """
        初始化

        :param name: 数据集名称
        :param shape: 数据集图片的形状
        :param num_classes: 数据集类别数
        :param train_set: 训练集
        :param val_set: 验证集
        :param test_set: 测试集
        """
        self._name = name
        self._shape = shape  # (C, H, W)
        self._num_classes = num_classes
        self._train_set = train_set
        self._val_set = val_set
        self._test_set = test_set

    def set_train_set(self, train_set: Dataset):
        """
        设置训练集(主要用于模拟投毒的情况)

        :param train_set: 训练集
        """
        self._train_set = train_set

    @property
    def name(self) -> str:
        """
        数据集名称

        :return: 数据集名称
        """
        return self._name

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        数据集图像形状 (C, H, W)

        :return: 图像形状
        """
        return self._shape

    @property
    def num_classes(self) -> int:
        """
        数据集类别数目

        :return: 类别数目
        """
        return self._num_classes

    @property
    def train_set(self) -> Dataset:
        """
        获取训练集
        """
        return self._train_set

    @property
    def val_set(self) -> Dataset:
        """
        获取验证集
        """
        return self._val_set

    @property
    def test_set(self) -> Dataset:
        """
        获取测试集
        """
        return self._test_set


class TransformedDataset(Dataset):
    """
    在原数据集上应用 transform，不影响原数据集
    """

    def __init__(self, dataset: Dataset, transform=None):
        """
        初始化

        :param dataset: 原数据集
        :param transform: Transform 对象
        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


class IndexedDataset(Dataset):
    """
    带数据下标的数据集
    """

    def __init__(self, dataset: Dataset, transform=None):
        """
        初始化带数据下标的数据集

        :param dataset: 原始数据集
        :param transform: 可选的变换函数
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label, index


class LabelFilteredSubset(Dataset):
    """
    某一标签下的数据子集
    """

    def __init__(
        self,
        dataset: Dataset,
        label_or_labels: int | list[int] | None = None,
        except_label_or_labels: int | list[int] | None = None,
        transform=None,
    ):
        """
        初始化

        注：label 和 except_label 只有一个会生效

        :param dataset: 原数据集
        :param label_or_labels: 标签，用于筛出特定标签的数据。可以是单个标签，也可以是标签列表
        :param except_label_or_labels: 标签，用于筛出不是这个标签的数据。可以是单个标签，也可以是标签列表
        :param transform: Transform
        :note: label 和 except_label 只有一个会生效
        """
        self.dataset = dataset
        self.transform = transform
        # 符合要求的数据下标
        if label_or_labels is not None:
            if isinstance(label_or_labels, int):
                label_or_labels = [label_or_labels]
            label_or_labels = set(label_or_labels)
            self.indices = [
                i for i, (_, l) in enumerate(dataset) if l in label_or_labels
            ]
        elif except_label_or_labels is not None:
            if isinstance(except_label_or_labels, int):
                except_label_or_labels = [except_label_or_labels]
            except_label_or_labels = set(except_label_or_labels)
            self.indices = [
                i for i, (_, l) in enumerate(dataset) if l not in except_label_or_labels
            ]

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)


class PoisonedDataset(Dataset):
    """
    图像投毒数据集
    """

    def __init__(
        self,
        dataset: Dataset,
        data_shape: tuple[int, int, int],
        trigger_gen: "TriggerGenerator", # type: ignore
        indexes: Iterable[int],
        target_label: int,
        data_transform_class: Type[MakeTransforms],
    ):
        """
        初始化图像投毒数据集

        :param dataset: 原始数据集，如果不是 Tensor 数据集会自动用 train_transforms 转换
        :param data_shape: 数据形状，(C, H, W)
        :param trigger_gen: 触发器生成模块
        :param indexes: 需要投毒的样本索引列表
        :param target_label: 目标标签
        :param data_transform: 数据增强模块类
        """
        self._data_transform = data_transform_class(input_shape=data_shape)
        self._dataset = dataset
        self._trigger_gen = trigger_gen
        self._indexes_set = set(indexes)
        self._target_label = target_label

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        image, label = self._dataset[index]
        # 如果不是 Tensor 先把图像转换为 Tensor 并应用训练变换
        if not torch.is_tensor(image):
            image = self._data_transform.train_transforms(image)
        if index in self._indexes_set:
            # 该样本需要投毒
            image = self._trigger_gen.apply_trigger(image, self._data_transform.tensor_trigger_transforms)
            label = self._target_label
        return image, label


class DataLoaderDataIter:

    def __init__(self, data_loader: DataLoader):
        """
        DataLoader 包装器，可以用于不断取出 DataLoader 的下一批数据，不会抛出 StopIteration 异常

        :param data_loader: DataLoader
        """
        self._data_loader = data_loader
        self._iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        next_data = next(self._iter, None)
        if next_data is None:
            # 重新迭代
            self._iter = iter(self._data_loader)
            next_data = next(self._iter, None)
        return next_data
