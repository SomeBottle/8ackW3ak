"""
SCAR - 复现论文 Taught Well Learned Ill 的代码

* Ref: Chen Y, Li B, Yuan Y, et al. Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack[J]. arXiv preprint arXiv:2509.23871, 2025.
"""

from .scar import SCAR
from .trigger_preoptimizer import SCARTriggerPreoptimizer

__all__ = ["SCAR", "SCARTriggerPreoptimizer"]
