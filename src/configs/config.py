import copy

# Checkpoints 保存路径
CHECKPOINTS_SAVE_PATH = "./bw_checkpoints/"

# 复现模块的 Checkpoints 保存路径
REPROD_CHECKPOINTS_SAVE_PATH = "./reprod_checkpoints/"

# TensorBoard 日志保存路径
TENSORBOARD_LOGS_PATH = "./tb_logs/"

# Dataloader 最大的 num_workers
DATALOADER_MAX_NUM_WORKERS = 16

# 选择的设备，设备名或者 'auto'
# 这项设置会被命令行参数 --device 覆盖掉
SELECTED_DEVICE = "auto"

# 实验信息存储文件名
EXP_INFO_FILE_NAME = "exp.info"
# 实验 Checkpoints 存储文件名
EXP_CKPTS_FILE_NAME = "exp.ckpts"

# 图像数据标准化时各个通道用的均值和标准差，保证图像值在 [-1, 1] 内
IMAGE_STANDARDIZE_MEANS = (0.5, 0.5, 0.5)
IMAGE_STANDARDIZE_STDS = (0.5, 0.5, 0.5)

# -------------- 配置文件结构原则 -------------- #
#
# 仅把每个模块不同的配置写进单独的 factory 模块中
# 比如 SCAR 和 OSCAR 都会用到 teacher 的配置，但是只有 SCAR 需要 surrogate
# 那么 teacher 的配置就写在这里，surrogate 配置写到 factory_scar.py 中
#
# ------------ 配置文件结构原则 END ------------ #

# BackWeak 主实验配置文件结构
# (数据类型, 默认值), 没有默认值 (None) 的一律必须提供
# 注意 id 的默认值是 "default"，这种情况下会自动对配置进行哈希以生成唯一 id
# NOTE: 以 ? 结尾的字段是非必要的。
BACKWEAK_EXP_CONFIG_STRUCTURE = {
    "basic": {
        "dataset_name": (str, None),
        "seed": (int, 42),
        "teacher": {
            "model": (str, None),
            "data_transform": (str, None),
        },
    },
    "validate": {
        "make_test_per_epochs": (int, 10),
        "save_ckpts_per_epochs": (int, 5),
    },
    "base_train": {
        "id": (str, "default"),
        "desc": (str, ""),
        "trainer": (str, None),
    },
    "backdoor": {
        "target_label": (int, None),
    },
    "trigger_gen": {
        "id": (str, "default"),
        "desc": (str, ""),
        "generator": (str, None),
    },
    "data_poison": {
        "id": (str, "default"),
        "desc": (str, ""),
        "poisoner": (str, None),
    },
    "teacher_tune": {
        "id": (str, "default"),
        "desc": (str, ""),
        "tuner": (str, None),
    },
    "defense?": {
        "defender": (str, "none"),
        "for": (str, "teacher"),  # defender 所防御的模型，teacher / student
        "params": (dict, {}),
    },
    "distill": {
        "distiller": (str, None),
        "dataset_name?": (str, "auto"),  # 默认采用 basic.dataset_name 分割的数据集
        "student": {
            "model": (str, None),
            "data_transform": (str, None),
        },
    },
    "test": {
        "clean": {
            "id": (str, "default"),
            "desc": (str, ""),
            "test": (bool, None),
        },
        "poisoned": {
            "id": (str, "default"),
            "desc": (str, ""),
            "test": (bool, None),
        },
    },
    "test_trigger?": {
        "perform": (bool, False),  # 默认不测试触发器 (可见性指标，可视化)
        "num_samples": (int, 5),  # 用于触发器可视化和评估的样本数量
    },
}

# SCAR 实验配置文件结构
# (数据类型, 默认值), 没有默认值 (None) 的一律必须提供
# 注意 id 的默认值是 "default"，这种情况下会自动对配置进行哈希以生成唯一 id
# NOTE: 以 ? 结尾的字段是非必要的。
SCAR_EXP_CONFIG_STRUCTURE = {
    "basic": {
        "dataset_name": (str, None),
        "seed": (int, 42),
    },
    "validate": {
        "make_test_per_epochs": (int, 10),
        "save_ckpts_per_epochs": (int, 2),
    },
    "benign_train": {
        "id": (str, "default"),
        "desc": (str, ""),
        "trainer": (str, None),
        "teacher": {
            "model": (str, None),
            "data_transform": (str, None),
        },
    },
    "benign_distill": {
        "id": (str, "default"),
        "desc": (str, ""),
        "distiller": (str, None),
        "student": {
            "model": (str, None),
            "data_transform": (str, None),
        },
    },
    "backdoor": {
        "target_label": (int, None),
    },
    "trigger_gen": {
        "id": (str, "default"),
        "desc": (str, ""),
        "generator": (str, None),
    },
    "scar": {
        "id": (str, "default"),
        "desc": (str, ""),
        "solution": (str, None),
        "teacher": {
            "model": (str, None),
            "data_transform": (str, None),
        },
    },
    "test_distill": {
        "id": (str, "default"),
        "desc": (str, ""),
        "distiller": (str, None),
        "student": {
            "model": (str, None),
            "data_transform": (str, None),
        },
    },
    "test_trigger?": {
        "perform": (bool, False),  # 默认不测试触发器 (可见性指标，可视化)
        "num_samples": (int, 5),  # 用于触发器可视化和评估的样本数量
    },
}

# ADBA 实验配置文件结构
# (数据类型, 默认值), 没有默认值 (None) 的一律必须提供
# 注意 id 的默认值是 "default"，这种情况下会自动对配置进行哈希以生成唯一 id
# NOTE: 以 ? 结尾的字段是非必要的。
ADBA_EXP_CONFIG_STRUCTURE = {
    "basic": {
        "dataset_name": (str, None),
        "seed": (int, 42),
    },
    "validate": {
        "make_test_per_epochs": (int, 10),
        "save_ckpts_per_epochs": (int, 2),
    },
    "benign_train": {
        "id": (str, "default"),
        "desc": (str, ""),
        "trainer": (str, None),
    },
    "backdoor": {
        "target_label": (int, None),
    },
    "adba": {
        "id": (str, "default"),
        "desc": (str, ""),
        "epochs": (int, None),
        "alpha": (float, None),
        "temperature": (float, None),
        "beta": (float, None),
        "mu": (float, None),
        "p": (int, None),
        "c": (int, None),
        "batch_size": (int, None),
        "teacher_lr": (float, None),
        "trigger_lr": (float, None),
        "shadow": {
            "model": (str, None),
        },
    },
    "test_distill": {
        "id": (str, "default"),
        "desc": (str, ""),
        "distiller": (str, None),
        "student": {
            "model": (str, None),
            "data_transform": (str, None),
        },
    },
    "test_trigger?": {
        "perform": (bool, False),  # 默认不测试触发器 (可见性指标，可视化)
        "num_samples": (int, 5),  # 用于触发器可视化和评估的样本数量
    },
}


_selected_device = SELECTED_DEVICE


def set_selected_device(device: str):
    """
    设置选择的设备

    :param device: 设备名或者 'auto'
    """
    global _selected_device
    _selected_device = device


def get_selected_device() -> str:
    """
    获取选择的设备
    """
    return _selected_device
