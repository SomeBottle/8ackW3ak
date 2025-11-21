"""
可视化相关的工具函数
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import torchvision.transforms.v2 as transformsV2
import math

from io import BytesIO
from torch import Tensor
from PIL import Image

from sklearn.manifold import TSNE
from collections.abc import Iterable

from configs import IMAGE_STANDARDIZE_MEANS, IMAGE_STANDARDIZE_STDS

# 展示多张图片时每行多少张
IMAGES_PER_ROW = 5

# 图像反标准化
_destandardize = transformsV2.Normalize(
    mean=[-m / s for m, s in zip(IMAGE_STANDARDIZE_MEANS, IMAGE_STANDARDIZE_STDS)],
    std=[1 / s for s in IMAGE_STANDARDIZE_STDS],
)


def _create_dir_if_not_exists(file_path: str):
    """
    如果目录不存在则创建目录

    :param file_path: 文件路径
    """
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def visualize_image(
    image: Tensor | Image.Image,
    standardized: bool = True,
    title: str = "",
    save_to: str | None = None,
) -> np.ndarray:
    """
    可视化 torch 中的图片或 PIL 图片

    :param image: 图片 Tensor or PIL 图片, shape (C, H, W)
    :param standardized: 是否已经标准化到 [-1, 1] 范围内，如果为 True 则会逆转换到 [0, 1] 范围内。仅对 Tensor 输入类型有效
    :param title: 标题
    :param save_to: 保存路径，如果为 None 则直接 plt.show()
    :return: 图片的 numpy array 格式, shape (H, W, C)
    """
    plt.close()
    # torch 中图片的维度顺序为 CxHxW, 而 matplotlib 中图片的维度顺序为 HxWxC，需要改变维度顺序
    if isinstance(image, Tensor):
        if standardized:
            image = _destandardize(image)
        plt_image = image.permute(1, 2, 0)
        plt.imshow(plt_image.numpy())
    else:
        plt.imshow(image)
    plt.title(title)
    if save_to:
        _create_dir_if_not_exists(save_to)
        plt.savefig(save_to)
    else:
        plt.show()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = np.array(Image.open(buf))
    buf.close()
    return img


def visualize_images(
    image_and_titles: list[tuple],
    standardized: bool | tuple[bool, ...] = True,
    save_to: str | None = None,
) -> np.ndarray:
    """
    可视化多张图片

    :param image_and_titles: 图片 (Tensor or PIL 图片) 和标题列表 [(image1, title1), (image2, title2), image3, ...]，只有图片则 title 为空字符串
    :param standardized: 是否已经标准化到 [-1, 1] 范围内，如果为 True 则会逆转换到 [0, 1] 范围内（仅对 Tensor 输入类型有效）。可以接受一个 bool 元组，分别为每个图片进行指定。
    :param save_to: 保存路径，如果为 None 则直接 plt.show()
    :return: 图片的 numpy array 格式, shape (H, W, C)
    """
    plt.close()
    rows = math.ceil(len(image_and_titles) / IMAGES_PER_ROW)
    cols = min(len(image_and_titles), IMAGES_PER_ROW)
    _, axes = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        # 有点坑啊老哥，1x1 时 subplots 直接返回 axes 而不是 axes 列表
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.reshape(-1)  # 将二维数组转为一维

    for i, input in enumerate(image_and_titles):
        if isinstance(input, (list, tuple)):
            image, title = input
        else:
            image, title = input, ""

        if isinstance(image, Tensor):
            image = image.detach().cpu()
            if (
                isinstance(standardized, bool)
                and standardized
                or isinstance(standardized, Iterable)
                and standardized[i]
            ):
                image = _destandardize(image)

            axes[i].imshow(image.permute(1, 2, 0).numpy())
        else:
            axes[i].imshow(image)
        axes[i].axis("off")  # 展示图片，不要碍眼的坐标轴
        axes[i].set_title(title, fontsize="smaller")
    plt.tight_layout()  # 自动调整子图间距
    if save_to:
        _create_dir_if_not_exists(save_to)
        plt.savefig(save_to, bbox_inches="tight")
    else:
        plt.show()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = np.array(Image.open(buf))
    buf.close()
    return img


def visualize_distribution(
    data: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    label_texts: dict = {},
    title: str = "",
    metric: str = "euclidean",
    save_to: str | None = None,
) -> np.ndarray:
    """
    利用 t-SNE 可视化高维数据分布

    :param data: 高维数据 Tensor 或 np array, shape (n_samples, n_features)
    :param labels: 数据标签 Tensor 或 np array, shape (n_samples,)
    :param label_texts: 标签文本字典，存储每个标签对应的文本标记
    :param title: 标题
    :param metric: 距离度量，用于 t-SNE，一般用 "euclidean" 或 "cosine"
    :param save_to: 保存路径，如果为 None 则直接 plt.show()
    """
    t_sne = TSNE(n_components=2, metric=metric)
    data_np = data
    labels_np = labels
    if isinstance(data, Tensor):
        data_np = data.detach().cpu().numpy()
    if isinstance(labels, Tensor):
        labels_np = labels.detach().cpu().numpy()
    # 降维到 2 维
    data_2d = t_sne.fit_transform(data_np)
    # 可视化
    plt.close()
    _, ax = plt.subplots()

    for label in np.unique(labels_np):
        label_text = label_texts.get(label, f"cluster {label}")
        ax.scatter(
            data_2d[labels_np == label, 0],
            data_2d[labels_np == label, 1],
            label=label_text,
        )
    ax.set_title(f"{title} ({metric})", fontsize="smaller")
    ax.legend()
    if save_to:
        _create_dir_if_not_exists(save_to)
        plt.savefig(save_to, bbox_inches="tight")
    else:
        plt.show()


def visualize_records(
    records: dict[str, list[float] | tuple[list[float], dict]],
    records_x: dict[str, list[float]] = {},
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    save_to: str | None = None,
) -> np.ndarray:
    """
    可视化实验记录

    :param records: 实验记录字典 {record_name: [record_1, record_2, ...]}，或者 {record_name: ([record_1, record_2, ...], {option1: value1, option2: value2, ...})}，后者可以指定 matplotlib 样式
    :param records_x: x 轴数据字典 {record_name: [x_1, x_2, ...]}，用于数据记录密度不同的情况。如果没有指定则一律按默认的 x 轴数据 [0, 1, 2, ...] 进行绘制
    :param title: 标题
    :param save_to: 保存路径，如果为 None 则直接 plt.show()
    :return: 图片的 numpy array 格式, shape (H, W, C)
    """
    plt.close()
    _, ax = plt.subplots()
    max_data_len = 0
    for name, record in records.items():
        if (
            isinstance(record, (tuple, list))
            and len(record) > 1
            and isinstance(record[1], dict)
        ):
            y, options = record
            max_data_len = max(max_data_len, len(y))
            if records_x is not None and name in records_x:
                x = records_x[name]
            else:
                x = np.arange(len(y))
            ax.plot(x, y, label=name, **options)
        else:
            max_data_len = max(max_data_len, len(record))
            if records_x is not None and name in records_x:
                x = records_x[name]
            else:
                x = np.arange(len(record))
            ax.plot(x, record, label=name)
    ax.set_title(title, fontsize="smaller")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if save_to:
        _create_dir_if_not_exists(save_to)
        plt.savefig(save_to, bbox_inches="tight")
    else:
        plt.show()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = np.array(Image.open(buf))
    buf.close()
    return img


def visualize_roc_curve(
    fpr: list[float] | np.ndarray,
    tpr: list[float] | np.ndarray,
    title: str = "",
    x_label: str = "FPR",
    y_label: str = "TPR",
    save_to: str | None = None,
) -> np.ndarray:
    """
    可视化 ROC 曲线

    :param fpr: 假正例率列表或数组
    :param tpr: 真正例率列表或数组
    :param title: 标题
    :param x_label: x 轴标签
    :param y_label: y 轴标签
    :param save_to: 保存路径，如果为 None 则直接 plt.show()
    """
    plt.close()
    _, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.set_title(title, fontsize="smaller")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    if save_to:
        _create_dir_if_not_exists(save_to)
        plt.savefig(save_to, bbox_inches="tight")
    else:
        plt.show()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = np.array(Image.open(buf))
    buf.close()
    return img


def join_images(
    image_paths: list[str],
    images_per_row: int = IMAGES_PER_ROW,
    fig_size: tuple[int, int] = (300, 200),
    save_to: str | None = None,
):
    """
    把多张图像拼接成一张图像

    :param image_paths: 图像路径列表
    :param images_per_row: 每行多少张图像
    :param fig_size: 单张图像的大小(W, H)，单位为像素 px
    :param save_to: 保存路径，如果为 None 则直接 plt.show()
    """
    images = [Image.open(path) for path in image_paths]
    # 计算需要的行数
    num_rows = math.ceil(len(images) / images_per_row)

    # 转换为英寸
    px = 1 / plt.rcParams["figure.dpi"]
    image_w, image_h = fig_size
    fig, axes = plt.subplots(
        num_rows,
        images_per_row,
        figsize=(image_w * px * images_per_row, image_h * px * num_rows),
    )
    if num_rows == 1 and images_per_row == 1:
        axes = np.array([axes])
    elif num_rows == 1 or images_per_row == 1:
        axes = axes.flatten()
    else:
        axes = axes.reshape(-1)  # 将二维数组转为一维

    for i in range(len(axes)):
        if i < len(images):
            axes[i].imshow(images[i])
            axes[i].axis("off")
        else:
            # 如果图像数量不足则填充空白
            axes[i].axis("off")
            axes[i].set_visible(False)

    plt.tight_layout()  # 自动调整子图间距

    if save_to:
        _create_dir_if_not_exists(save_to)
        plt.savefig(save_to, bbox_inches="tight")
    else:
        plt.show()
