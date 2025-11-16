"""
记录实验数据的一些类
"""
from torch import Tensor

class AverageLossRecorder:
    """
    损失值记录器
    """

    def __init__(self):
        """
        初始化损失值记录器
        """
        self._running_loss = 0.0
        self._num_samples = 0

    def batch_update(self, loss: Tensor | float, batch_size: int):
        """
        更新损失值

        :param loss: 当前批次的平均损失值
        :param batch_size: 当前批次的样本数
        """
        if isinstance(loss, Tensor):
            loss = loss.detach().item()
        self._running_loss += loss * batch_size
        self._num_samples += batch_size

    @property
    def avg_loss(self) -> float:
        """
        获取当前平均损失值

        :return: 平均损失值
        """
        if self._num_samples == 0:
            return 0.0
        return self._running_loss / self._num_samples
