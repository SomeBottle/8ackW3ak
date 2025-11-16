"""
SimKD 学生教师结对模块

- 蒸馏学生时用的是这个模块，成对输出学生和教师的特征用于匹配

* Ref: Chen D, Mei J P, Zhang H, et al. Knowledge distillation with the reused teacher classifier[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11933-11942.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simkd_student import SimKDStudent


class SimKDFeaturePair(nn.Module):
    """
    SimKD 学生教师特征结对模块
    """

    def __init__(self, simkd_student: SimKDStudent, teacher_model: nn.Module):
        """
        初始化 SimKD 特征结对模块

        :param simkd_student: SimKD 学生模型包装实例
        :param teacher_model: 教师模型实例 (training 模式会保持不变)
        """
        super().__init__()
        self._simkd_student = simkd_student
        self._teacher_model = teacher_model

    def train(self, mode: bool = True, *args, **kwargs):
        """
        (覆写) 设置模块为训练或评估模式, 保持教师模型的模式不变

        :param mode: 是否为训练模式
        """
        teacher_training = self._teacher_model.training
        super().train(mode, *args, **kwargs)
        self._teacher_model.train(teacher_training)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播，返回学生和教师的同尺度特征以及学生的分类输出

        :param x: 输入张量
        :return: (教师特征, 学生特征, 学生分类输出)
        """
        with torch.no_grad():
            teacher_features: torch.Tensor
            teacher_features, _ = self._teacher_model(x, feat=True)
            teacher_feature_l_2, teacher_feature_l_1 = teacher_features

        student_features, student_logits = self._simkd_student.feature_class_forward(x)

        if self._simkd_student.chosen_feature_level == -2:
            s_H, s_W = student_features.shape[2:4]

            # feature_forward 使得 s_H <= t_H, s_W <= t_W
            # 当 t_H, t_W 较大时，进行池化
            t_H, t_W = teacher_feature_l_2.shape[2:4]
            if s_H * s_W < t_H * t_W:
                teacher_feature_out = F.adaptive_avg_pool2d(
                    teacher_feature_l_2, (s_H, s_W)
                )
            else:
                teacher_feature_out = teacher_feature_l_2

        else:
            teacher_feature_out = teacher_feature_l_1

        return teacher_feature_out, student_features, student_logits
