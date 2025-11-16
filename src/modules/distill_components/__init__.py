"""
蒸馏方法相关的工具类
"""

from .simkd_feature_pair import SimKDFeaturePair
from .simkd_student import SimKDStudent

__all__ = [
    "SimKDStudent",
    "SimKDFeaturePair",
]
