"""
ADBA 抽象类
"""

from modules.abc import TriggerGenerator
from abc import abstractmethod
from modules.abc.exp_base import ExpBase


class ADBABase(ExpBase):
    """
    ADBA 抽象类
    """

    @abstractmethod
    def get_trigger_generator(self) -> TriggerGenerator:
        """
        获取触发器生成器

        :return: 触发器生成器
        """
        pass
