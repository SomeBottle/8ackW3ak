"""
一些数据相关的工具函数
"""

import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit


def balanced_split_into_two(
    dataset: Dataset,
    latter_size_or_ratio: int | float,
    random_state: int | None = None,
) -> tuple[Dataset, Dataset]:
    """
    均衡地将数据集划分为两部分

    :param dataset: 数据集
    :param latter_size_or_ratio: 后一部分数据集的大小(样本数量)，或者为后一部分数据集的比例(0 < latter_size < 1)
    :param random_state: 划分的随机种子
    :return: 前一部分数据集和后一部分数据集
    :raise ValueError: 如果 latter_size <= 0 或者 latter_size >= len(dataset)
    """
    if latter_size_or_ratio <= 0 or latter_size_or_ratio >= len(dataset):
        raise ValueError(
            "latter_size_or_ratio must be greater than 0 and less than the dataset size."
        )

    # 获取所有样本的标签
    dataset_size = len(dataset)
    all_labels = np.array([label for _, label in dataset])
    if isinstance(latter_size_or_ratio, float) and 0 < latter_size_or_ratio < 1:
        # 是比例
        latter_ratio = latter_size_or_ratio
    else:
        # 是样本数量
        latter_ratio = latter_size_or_ratio / dataset_size  # 计算后一个数据集占的比例
    # 使用 StratifiedShuffleSplit 保持类别分布
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=latter_ratio, random_state=random_state
    )

    first_indices, second_indices = next(sss.split(np.zeros(dataset_size), all_labels))

    first_dataset = Subset(dataset, first_indices)
    second_dataset = Subset(dataset, second_indices)

    return first_dataset, second_dataset
