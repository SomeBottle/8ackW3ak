"""
实验中能用到的数据增强模块
"""

from .simple_transform_make import MakeSimpleTransforms
from .simple_transform_plain_trigger_make import MakeSimpleTransformsPlainTrigger

__all__ = [
    "MakeSimpleTransforms",
    "MakeSimpleTransformsPlainTrigger",
]
