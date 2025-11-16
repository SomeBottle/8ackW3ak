"""
OSCAR - 基于 Taught Well Learned Ill 论文进行简化的代码模块

* Ostensibly Stealthy distillation-Conditional bAckdooR attack (OSCAR)
* 类似于论文 4.3 节 w/o F_s 的消融实验方案
"""

from .oscar import OSCAR

__all__ = ["OSCAR"]
