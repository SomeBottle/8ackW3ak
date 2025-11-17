import torchvision
from os import path

# from utils.data_funcs import balanced_split_into_two

# 数据集路径
DATASET_PATH = "./datasets/"
# 数据集划分时的随机种子
SEED_DATASET_SPLIT = 10492

##################
# CIFAR-10 数据集
##################

_cifar10_trainset_all = torchvision.datasets.CIFAR10(
    root=DATASET_PATH,
    train=True,
    download=False,
)
_cifar10_testset = torchvision.datasets.CIFAR10(
    root=DATASET_PATH,
    train=False,
    download=False,
)
# Old: CIFAR10 训练集有 50000 张图片，额外划分成 45000 张训练集和 5000 张验证集
# _cifar10_trainset, _cifar10_valset = balanced_split_into_two(
#     _cifar10_trainset_all,
#     latter_size_or_ratio=5000,
#     random_state=SEED_DATASET_SPLIT,
# )

# New: 直接使用官方的测试集作为验证集，和其他没有公布测试集标签的数据集保持一致
_cifar10_trainset = _cifar10_trainset_all
_cifar10_valset = _cifar10_testset

##################
# CINIC-10 数据集
##################

_cinic10_trainset = torchvision.datasets.ImageFolder(
    root=path.join(DATASET_PATH, "cinic_10/train"),
    allow_empty=True,
)

_cinic10_valset = torchvision.datasets.ImageFolder(
    root=path.join(DATASET_PATH, "cinic_10/valid"),
    allow_empty=True,
)

_cinic10_testset = torchvision.datasets.ImageFolder(
    root=path.join(DATASET_PATH, "cinic_10/test"),
    allow_empty=True,
)

##################
# GTSRB 数据集
##################

# _gtsrb_trainset = torchvision.datasets.ImageFolder(
#     root=path.join(DATASET_PATH, "GTSRB_224_balanced/train"),
#     allow_empty=True,
# )

# _gtsrb_testset = torchvision.datasets.ImageFolder(
#     root=path.join(DATASET_PATH, "GTSRB_224_balanced/test"),
#     allow_empty=True,
# )

# _gtsrb_valset = _gtsrb_testset

##################
# ImageNet-50 数据集
##################

_imagenet50_trainset = torchvision.datasets.ImageFolder(
    root=path.join(DATASET_PATH, "imagenet_50_224/train"),
    allow_empty=True,
)

_imagenet50_valset = torchvision.datasets.ImageFolder(
    root=path.join(DATASET_PATH, "imagenet_50_224/val"),
    allow_empty=True,
)

# ImageNet-50 测试集没有标签，直接用验证集
_imagenet50_testset = _imagenet50_valset


# 数据集的各种配置信息
DATASET_INFOS = {
    "cifar10": {
        # 为了方便，此处维度顺序为 (C, H, W)
        "shape": (3, 32, 32),
        "num_classes": 10,
        "train_set": _cifar10_trainset,
        "val_set": _cifar10_valset,
        "test_set": _cifar10_testset,
    },
    "cinic10": {
        "shape": (3, 32, 32),
        "num_classes": 10,
        "train_set": _cinic10_trainset,
        "val_set": _cinic10_valset,
        "test_set": _cinic10_testset,
    },
    # "gtsrb": {
    #     "shape": (3, 224, 224),
    #     "num_classes": 43,
    #     "train_set": _gtsrb_trainset,
    #     "val_set": _gtsrb_valset,
    #     "test_set": _gtsrb_testset,
    # },
    "imagenet50": {
        "shape": (3, 224, 224),
        "num_classes": 50,
        "train_set": _imagenet50_trainset,
        "val_set": _imagenet50_valset,
        "test_set": _imagenet50_testset,
    },
}
