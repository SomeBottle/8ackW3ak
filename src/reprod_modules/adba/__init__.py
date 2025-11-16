"""
ADBA - Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation

实现模块

* Ref: Ge Y, Wang Q, Zheng B, et al. Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation[C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 826-834.
"""

from .adba import ADBA
from .trigger_adba import ADBATrigger

__all__ = ["ADBA", "ADBATrigger"]
