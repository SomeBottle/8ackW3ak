"""
专供 ADBA 模块使用的触发器
"""

import torch

from modules.abc import TriggerGenerator
from utils.funcs import apply_trigger_with_mask


class ADBATrigger(TriggerGenerator):
    """
    ADBA 专用触发器类
    """

    def __init__(
        self,
        trigger_pattern: torch.Tensor,
        trigger_mask: torch.Tensor,
    ):
        """
        初始化 ADBA 触发器

        :param trigger_pattern: 触发器图案张量, shape (1, C, H, W)
        :param trigger_mask: 触发器掩码张量, shape (1, C, H, W)
        """
        super().__init__()
        self._trigger_pattern = trigger_pattern.detach()
        self._trigger_mask = trigger_mask.detach()

    @property
    def exp_id(self) -> str:
        return ""  # 这个模块没有实验 ID

    def generate(self):
        pass

    def apply_trigger(self, input_data, transform=None):
        trigger_mask = self._trigger_mask.to(input_data.device)
        trigger_pattern = self._trigger_pattern.to(input_data.device)
        if transform is not None:
            # 通常 ADBA 不会用 transform，但是这里还是写一下吧
            # 要对 mask 和 pattern 施加相同的变换
            temp_batch = torch.cat([trigger_mask, trigger_pattern], dim=0)
            transformed = transform(temp_batch)
            trigger_mask, trigger_pattern = torch.split(transformed, 1, dim=0)
        if input_data.dim() == 3:
            trigger_mask = trigger_mask.squeeze(0)
            trigger_pattern = trigger_pattern.squeeze(0)
        return apply_trigger_with_mask(input_data, trigger_pattern, trigger_mask)
