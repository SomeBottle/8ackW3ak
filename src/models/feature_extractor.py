"""
模型包装器，把模型 pre-logits (最后一个线性层之前) 部分的输出作为特征输出
"""

import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(self, model: nn.Module):
        """
        初始化模型包装器，把模型 pre-logits (最后一个线性层之前) 部分的输出作为特征输出
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        前向传播，返回 pre-logits 的输出作为特征

        :param x: 输入数据
        :return: pre-logits 的输出特征
        """
        # 临时给模型添加 pre-hook
        feature = None
        # 标记线性层是不是最后一层
        is_linear_last = False

        def _linear_capturer(module, inputs):
            """
            作为线性层 forward 的 pre-hook，以获得最后一层的输入

            :param module: 线性层模块
            :param inputs: 输入数据
            """
            nonlocal feature, is_linear_last
            if isinstance(module, nn.Linear):
                feature = inputs[0]
                is_linear_last = True
            else:
                is_linear_last = False

        # 给每个层都注册 pre-hook
        hooks = []
        for module in self.model.modules():
            h = module.register_forward_pre_hook(_linear_capturer)
            hooks.append(h)

        self.model(x)

        # 移除所有的 pre-hook
        for h in hooks:
            h.remove()

        if feature is None:
            raise RuntimeError("No linear layer exists in the model.")

        if not is_linear_last:
            raise RuntimeError("The last layer is not a linear layer.")

        return feature
