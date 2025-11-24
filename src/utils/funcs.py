"""
一些工具函数
"""

import random
import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from contextlib import contextmanager
from configs import DATALOADER_MAX_NUM_WORKERS, get_selected_device
from typing import Iterable


def get_grad_norm(
    params_or_grads: Iterable[torch.Tensor] | Iterable[torch.nn.Parameter],
    norm_type: float = 2.0,
) -> tuple[float, float]:
    """
    计算一组参数的梯度范数

    :param params_or_grads: 模型参数或梯度张量的可迭代对象
    :param norm_type: 范数类型
    :return: (范数, 计算耗时), 计算耗时单位是秒
    """
    start_time = time.perf_counter()
    grads = []
    for pg in params_or_grads:
        if pg is None:
            continue
        if isinstance(pg, torch.nn.Parameter):
            if pg.grad is not None:
                grads.append(pg.grad)
        else:
            grads.append(pg)

    total_norm = 0.0
    for g in grads:
        param_norm = g.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    end_time = time.perf_counter()
    return total_norm, end_time - start_time


def json_serialize_helper(obj):
    """
    JSON 序列化辅助函数，用于处理无法直接序列化的对象 (如 Tensor)
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def print_section(title: str, symbol: str = "=", length: int = 50):
    """
    打印一个居中标题，前后有分隔线

    :param title: 标题内容
    :param symbol: 分隔线符号
    :param length: 分隔线长度
    """
    length = max(length, len(title) + 2)
    num_spaces = length // 2 - len(title) // 2
    print(symbol * length)
    print(" " * num_spaces + title)
    print(symbol * length)


def config_check(config: dict, config_structure: dict):
    """
    检查实验配置是否完备，填上一些默认值

    :param config: 实验配置字典
    :raises ValueError: 配置不完备
    """

    def _scanner(conf: dict, conf_struct: dict, path: str = ""):
        for key, val in conf_struct.items():
            if key.endswith("?"):
                # ? 结尾的是可选字段
                key = key[:-1]
                if key not in conf:
                    continue
            default = None
            dtype = None
            if isinstance(val, tuple):
                dtype, default = val
            if key not in conf:
                if default is None:
                    raise ValueError(f"Config missing key: {path}.{key}")
                conf[key] = default
            if isinstance(val, dict):
                if not isinstance(conf[key], dict):
                    raise ValueError(f"Config key {path}.{key} should be dict")
                _scanner(conf[key], val, path=f"{path}.{key}" if path else key)
                continue
            if not isinstance(conf[key], dtype):
                raise ValueError(
                    f"Config key {path}.{key} should be {dtype.__name__}, got {conf[key]}({type(conf[key]).__name__})"
                )

    _scanner(config, config_structure)


def get_dataset_labels(dataset: Dataset) -> list:
    """
    获得数据集的所有标签

    :param dataset: 数据集
    :return: 数据集的所有标签
    """
    return list(set([label for _, label in dataset]))


def get_time_str() -> str:
    """
    获得当前时间字符串
    :return: 当前时间字符串，格式为 YYYY_MM_DD_HH_MM_SS
    """
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))


def get_base_exp_info() -> dict:
    """
    获取基础实验信息模板，目前包含：

    * 实验时间戳 (秒级)

    :return: 基础实验信息模板
    """
    return {
        "time": get_timestamp(),
    }


def get_timestamp(ms: bool = False) -> int:
    """
    获取当前时间戳

    :param ms: 是否以毫秒为单位
    :return: 当前时间戳
    """
    if ms:
        return int(time.time() * 1000)
    else:
        return int(time.time())


def fix_seed(seed: int):
    """
    固定全局的随机种子

    :param seed: 种子
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_curr_random_states() -> dict:
    """
    获取当前的随机数状态

    :return: 当前的随机数状态
    """
    states = {
        "torch": torch.get_rng_state(),
        "torch_cuda": None,
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def load_random_states(states: dict):
    """
    加载之前保存的随机数状态

    :param states: 之前保存的随机数状态
    """
    torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    np.random.set_state(states["numpy"])
    random.setstate(states["random"])


@contextmanager
def temp_seed(seed: int):
    """
    临时应用随机种子 (上下文管理器)

    :param seed: 种子
    """
    # 保存状态
    prev_states = get_curr_random_states()

    fix_seed(seed)

    try:
        yield
    finally:
        # 恢复状态
        load_random_states(prev_states)


@contextmanager
def temp_eval(model: nn.Module):
    """
    临时切换模型到 eval 模式 (上下文管理器)

    :param model: 模型
    """
    prev_mode = model.training
    model.eval()
    try:
        yield
    finally:
        if prev_mode:
            model.train()


def shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    计算一批概率分布的香农熵

    :param probs: 概率分布，shape: (B, num_classes) or (num_classes,)
    :param eps: 防止除零的极小值
    :return: 香农熵，shape: (B,) or scalar
    """
    # 防止概率过低出现无穷值
    probs = torch.clamp(probs, min=eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def auto_select_device() -> torch.device:
    """
    自动选择设备，优先选择剩余显存较多的 GPU

    :return: 选择的设备
    """
    selected_device = get_selected_device()
    if selected_device != "auto":
        return torch.device(selected_device)

    selected_device = torch.device("cpu")
    if torch.cuda.is_available():
        selected_gpu_free_mem = 0
        for i in range(torch.cuda.device_count()):
            free_mem, _ = torch.cuda.mem_get_info(i)
            if free_mem > selected_gpu_free_mem:
                selected_gpu_free_mem = free_mem
                selected_device = torch.device(f"cuda:{i}")
    return selected_device


def auto_num_workers(max_workers: int = DATALOADER_MAX_NUM_WORKERS) -> int:
    """
    自动选择 DataLoader 的 num_workers 参数

    :param max_workers: 最大的 num_workers
    :return: 选择的 num_workers
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 0
    return min(cpu_count // 2, max_workers)


def apply_trigger_without_mask(
    images: torch.Tensor, trigger: torch.Tensor
) -> torch.Tensor:
    """
    把触发器应用到图像上，没有掩码

    :param images: 输入图像，已经标准化到 [-1, 1]，形状为 (C, H, W) 或 (B, C, H, W)
    :param trigger: 触发器，形状为 (C, H, W) 或 (B, C, H, W)
    :return: 应用触发器后的图像，形状同输入图像
    """
    trigger = trigger.to(images.device)

    if trigger.dim() == 3 and images.dim() == 4:
        trigger = trigger.unsqueeze(0)

    triggered_images = images + trigger
    if images.dim() == 3 and triggered_images.dim() == 4:
        triggered_images = triggered_images.squeeze(0)
    return triggered_images.clip(-1, 1)


def apply_trigger_with_mask(
    images: torch.Tensor, trigger_pattern: torch.Tensor, trigger_mask: torch.Tensor
) -> torch.Tensor:
    """
    把触发器应用到图像上，有掩码

    :param images: 输入图像，已经标准化到 [-1, 1]，形状为 (C, H, W) 或 (B, C, H, W)
    :param trigger_pattern: 触发器图案，也已经在 [-1, 1] 范围内，形状为 (C, H, W) 或 (B, C, H, W)
    :param trigger_mask: 触发器掩码，形状匹配 trigger_pattern，取值为 [0, 1]
    :return: 应用触发器后的图像, 形状同输入图像
    """
    trigger_pattern = trigger_pattern.to(images.device)
    trigger_mask = trigger_mask.clip(0, 1).to(images.device)

    if trigger_pattern.dim() == 3 and images.dim() == 4:
        trigger_pattern = trigger_pattern.unsqueeze(0)
        trigger_mask = trigger_mask.unsqueeze(0)

    triggered_images = images * (1 - trigger_mask) + trigger_pattern * trigger_mask
    if images.dim() == 3 and triggered_images.dim() == 4:
        triggered_images = triggered_images.squeeze(0)
    return triggered_images.clip(-1, 1)


def test_attack_success_rate(
    model: nn.Module,
    data_loader: DataLoader,
    target_label: int,
    trigger: torch.Tensor = None,
    trigger_mask: torch.Tensor = None,
    trigger_gen: "TriggerGenerator" = None,  # type: ignore
    device: torch.device = None,
    except_target: bool = True,
) -> float:
    """
    测试模型在后门攻击上的成功率

    (注: trigger, trigger_gen 至少要提供一个，如果提供了 trigger 和 trigger_mask，会按掩码叠加触发器)

    :param model: 要测试的模型
    :param data_loader: 数据加载器
    :param target_label: 目标标签
    :param trigger: 触发器图案张量
    :param trigger_mask: 触发器掩码张量
    :param trigger_gen: 触发器生成模块
    :param device: 运行设备，如果为 None 则自动选择
    :param except_target: 是否排除原本就是目标标签的样本
    :return: 模型在后门攻击上的成功率
    :raises ValueError: trigger 和 trigger_gen 均未提供
    """
    if trigger is None and trigger_gen is None:
        raise ValueError("Either trigger or trigger_gen must be provided.")

    if device is None:
        device = auto_select_device()
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            if trigger_gen is not None:
                images = trigger_gen.apply_trigger(images)
            else:
                if trigger_mask is not None:
                    images = apply_trigger_with_mask(images, trigger, trigger_mask)
                else:
                    images = apply_trigger_without_mask(images, trigger)
            target_labels = torch.full(
                (images.size(0),), target_label, dtype=torch.long
            ).to(device)
            non_target_mask = labels != target_label
            if except_target:
                # 排除掉原本就是目标标签的样本
                images = images[non_target_mask]
                target_labels = target_labels[non_target_mask]
                if images.size(0) == 0:
                    continue
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += images.size(0)
            correct += (predicted == target_labels).sum().item()

    asr = correct / total
    return asr


def test_benign_accuracy(
    model: nn.Module, data_loader: DataLoader, device: torch.device = None
) -> float:
    """
    测试模型在干净数据上的准确率

    :param model: 要测试的模型
    :param data_loader: 数据加载器
    :param device: 运行设备，如果为 None 则自动选择
    :return: 模型在干净数据上的准确率
    """
    if device is None:
        device = auto_select_device()
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def freeze_last_n_layers(
    model: nn.Module, n: int, dummy_input: torch.Tensor
) -> list[nn.Module]:
    """
    (原地操作) 冻结模型的最后 n 层参数

    :param model: 模型实例
    :param n: 要冻结的层数 (>=1)
    :param dummy_input: 用于执行前向传播的虚拟输入张量
    :return: 冻结的层列表
    """
    if n < 1:
        print("No layers to freeze.")
        return []
    # 自 Python 3.7 起，字典 items 是有序的
    layer_dict: dict[int, nn.Module] = {}
    dummy_input = dummy_input.to(next(model.parameters()).device)

    def _layer_capturer(module: nn.Module, args):
        """
        作为模块 forward 的 hook，以获得模型的层序列

        :param module: 模块
        :param args: 输入数据
        """
        # 没有参数的模块直接忽略
        if not any(True for _ in module.parameters(recurse=False)):
            return
        module_id = id(module)
        # 如果这个模块出现过，说明是复用模块
        # 删掉之前的记录，以更新模块出现的位置
        if module_id in layer_dict:
            del layer_dict[module_id]
        layer_dict[module_id] = module

    # 临时给模型加 hook
    hooks = []
    for modules in model.modules():
        # 只给叶子模块注册 hook (防止嵌套)
        if len(list(modules.children())) > 0:
            continue
        h = modules.register_forward_pre_hook(_layer_capturer)
        hooks.append(h)

    with temp_eval(model):
        # 有的模型的 BatchNorm 层在 train 模式下要求输入数据量大于 1
        # 这里临时切换为评估模式
        model(dummy_input)

    # 移除所有 hook
    for h in hooks:
        h.remove()

    # 转换为列表
    layer_seq = list(layer_dict.values())
    layer_frozen: list[nn.Module] = []

    # 冻结最后 n 层
    freeze_cnt = 0
    for layer in layer_seq[-n:]:
        for param in layer.parameters():
            param.requires_grad = False
        freeze_cnt += 1
        layer_frozen.append(layer)

    print(f"Froze last {freeze_cnt} layers of the model.")

    return layer_frozen


def get_module_seq(model: nn.Module, dummy_input: torch.Tensor) -> list[nn.Module]:
    """
    获得模型的模块序列

    :param model: 模型实例
    :param dummy_input: 用于执行前向传播的虚拟输入张量
    :return: 模型的模块序列
    """
    # 自 Python 3.7 起，字典 items 是有序的
    layer_dict: dict[int, nn.Module] = {}
    dummy_input = dummy_input.to(next(model.parameters()).device)

    def _layer_capturer(module: nn.Module, args):
        """
        作为模块 forward 的 hook，以获得模型的层序列

        :param module: 模块
        :param args: 输入数据
        """
        # 没有参数的模块直接忽略
        if not any(True for _ in module.parameters(recurse=False)):
            return
        module_id = id(module)
        # 如果这个模块出现过，说明是复用模块
        # 删掉之前的记录，以更新模块出现的位置
        if module_id in layer_dict:
            del layer_dict[module_id]
        layer_dict[module_id] = module

    # 临时给模型加 hook
    hooks = []
    for modules in model.modules():
        # 只给叶子模块注册 hook (防止嵌套)
        if len(list(modules.children())) > 0:
            continue
        h = modules.register_forward_pre_hook(_layer_capturer)
        hooks.append(h)

    with temp_eval(model):
        # 有的模型的 BatchNorm 层在 train 模式下要求输入数据量大于 1
        # 这里临时切换为评估模式
        model(dummy_input)

    # 移除所有 hook
    for h in hooks:
        h.remove()

    # 转换为列表
    layer_seq = list(layer_dict.values())

    return layer_seq


def get_last_linear_layer(model: nn.Module, dummy_input: torch.Tensor) -> nn.Linear:
    """
    获取模型中的最后一个线性层

    :param model: 模型实例
    :param dummy_input: 用于执行前向传播的虚拟输入张量
    :return: 模型最后一个线性层
    :raises RuntimeError: 模型中没有线性层
    """
    last_linear: nn.Linear = None
    dummy_input = dummy_input.to(next(model.parameters()).device)

    def _linear_capturer(module, args):
        """
        作为线性层 forward 的 hook，以获得最后一层的线性层模块

        :nonlocal last_linear: 最后一个线性层模块
        :param module: 模块
        :param args: 输入数据
        """
        nonlocal last_linear
        if isinstance(module, nn.Linear):
            last_linear = module

    # 临时给模型加 hook
    hooks = []
    for modules in model.modules():
        # 只给叶子模块注册 hook (防止嵌套)
        if len(list(modules.children())) > 0:
            continue
        h = modules.register_forward_pre_hook(_linear_capturer)
        hooks.append(h)

    with temp_eval(model):
        # 有的模型的 BatchNorm 层在 train 模式下要求输入数据量大于 1
        # 这里临时切换为评估模式
        model(dummy_input)

    # 移除所有 hook
    for h in hooks:
        h.remove()

    if last_linear is None:
        raise RuntimeError("No linear layer exists in the model.")

    return last_linear
