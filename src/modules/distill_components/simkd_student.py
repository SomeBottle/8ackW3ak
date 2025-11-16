"""
SimKD 学生包装模块

- 利用 SimKD 蒸馏完成后实际上是用的这个模块进行推理

* Ref: Chen D, Mei J P, Zhang H, et al. Knowledge distillation with the reused teacher classifier[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 11933-11942.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.funcs import temp_eval
from models.abc import ModelBase


class SimKDStudent(nn.Module):
    """
    SimKD 学生模型包装类
    """

    def __init__(
        self,
        student_model: ModelBase,
        teacher_model: ModelBase,
        input_shape: tuple[int, int, int],
        factor: int = 2,
    ):
        """
        初始化 SimKD 包装的学生模型

        * 注意，设置这个模块为 eval 模式后，学生会被一并设置为 eval 模式

        :param student_model: 学生模型实例
        :param teacher_model: 教师模型实例 (主要借用其分类器)
        :param input_shape: 输入数据的形状 (C, H, W)
        :param factor: 特征转换层的缩放因子, factor 越大，转换层越窄(参数越少)
        :raises NotImplementedError: 如果教师或学生模型的 forward 方法不支持 feat 参数
        """
        super().__init__()

        dummy_input = torch.randn(1, *input_shape)
        dummy_input_teacher = dummy_input.to(next(teacher_model.parameters()).device)
        dummy_input_student = dummy_input.to(next(student_model.parameters()).device)

        # 检查教师和学生模型是否在 forward 上支持了 feat 参数
        try:
            features_teacher: torch.Tensor
            with temp_eval(teacher_model):
                # 因为 dummy_input 只有一个数据，经过含有 BatchNorm1d 层的模型时，若在 train 模式下则会报错，所以要临时关闭 BN 层
                features_teacher, _ = teacher_model(dummy_input_teacher, feat=True)
        except (TypeError, ValueError):
            raise NotImplementedError(
                f"Teacher model {teacher_model.__class__.__name__} does not support feat=True in forward method."
            )
        try:
            features_student: torch.Tensor
            with temp_eval(student_model):
                features_student, _ = student_model(dummy_input_student, feat=True)
        except (TypeError, ValueError):
            raise NotImplementedError(
                f"Student model {student_model.__class__.__name__} does not support feat=True in forward method."
            )

        feature_teacher_l_2, feature_teacher_l_1 = features_teacher
        feature_student_l_2, feature_student_l_1 = features_student

        if feature_teacher_l_2.dim() == feature_student_l_2.dim():
            # 二者倒数第二个特征形状维度相同，就可以用倒数第二个特征进行对齐
            shape_feature_teacher = feature_teacher_l_2.shape
            shape_feature_student = feature_student_l_2.shape
            # 学生输出特征的通道数
            chans_feature_student = shape_feature_student[1]
            # 教师输出特征的通道数
            chans_feature_teacher = shape_feature_teacher[1]

            # 转换层 (Bottleneck)，把学生模型的特征进行变换以匹配教师特征维度
            self._transform_layer = nn.Sequential(
                # 通道维度变换 (B, C_feat_stu, H, W) -> (B, C_feat_tea // factor, H, W)
                nn.Conv2d(
                    chans_feature_student,
                    chans_feature_teacher // factor,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                # 对特征进行归一化，加速训练，shape 不变
                nn.BatchNorm2d(chans_feature_teacher // factor),
                nn.ReLU(inplace=True),
                # 在维度变换后进一步提取特征，shape 不变
                nn.Conv2d(
                    chans_feature_teacher // factor,
                    chans_feature_teacher // factor,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(chans_feature_teacher // factor),
                nn.ReLU(inplace=True),
                # 通道升维 (B, C_feat_tea // factor, H, W) -> (B, C_feat_tea, H, W)
                nn.Conv2d(
                    chans_feature_teacher // factor,
                    chans_feature_teacher,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(chans_feature_teacher),
                nn.ReLU(inplace=True),
            )
            # (B, C, H, W) -> (B, C, 1, 1)
            self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self._chosen_feature_level = -2  # 选择倒数第二个特征进行对齐
        else:
            # 否则只能用倒数第一个特征进行对齐
            shape_feature_teacher = feature_teacher_l_1.shape
            shape_feature_student = feature_student_l_1.shape

            dim_teacher = shape_feature_teacher[1]
            dim_student = shape_feature_student[1]

            dim_bottleneck = max(1, dim_teacher // factor)

            self._transform_layer = nn.Sequential(
                nn.Linear(dim_student, dim_bottleneck, bias=False),
                nn.BatchNorm1d(dim_bottleneck),
                nn.ReLU(inplace=True),
                nn.Linear(dim_bottleneck, dim_bottleneck, bias=False),
                nn.BatchNorm1d(dim_bottleneck),
                nn.ReLU(inplace=True),
                nn.Linear(dim_bottleneck, dim_teacher, bias=False),
                nn.BatchNorm1d(dim_teacher),
                nn.ReLU(inplace=True),
            )
            self._chosen_feature_level = -1  # 选择倒数第一个特征进行对齐

        # teacher_model 分类器
        teacher_classifier = teacher_model.classifier

        self._shape_feature_teacher = shape_feature_teacher
        self._shape_feature_student = shape_feature_student
        self._student_model = student_model
        # 深拷贝，防止影响到教师
        self._teacher_classifier = copy.deepcopy(teacher_classifier)
        self._teacher_classifier.eval()
        self._teacher_classifier.requires_grad_(False)

        # 把自己移动到学生模型所在设备
        self.to(next(student_model.parameters()).device)

    @property
    def chosen_feature_level(self) -> int:
        """
        获取用于对齐的特征层级

        :return: -2 表示倒数第二个特征，-1 表示倒数第一个特征
        """
        return self._chosen_feature_level

    def feature_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回学生在分类层前未经池化和平铺的特征表示

        注: 输出的特征尺寸只可能 <= 教师的特征尺寸

        :param x: 输入张量，形状为 (B, C, H, W)
        :return: 对齐到教师的学生特征张量，形状为 (B, C_feat_teacher, H_feat, W_feat)
        """
        features_student, _ = self._student_model(x, feat=True)

        if self._chosen_feature_level == -2:
            # 学生特征，形状应该是 (B, C_feat_student, H_feat_stu, W_feat_stu)
            feature_student_l_2 = features_student[-2]
            s_H, s_W = self._shape_feature_student[2:4]
            t_H, t_W = self._shape_feature_teacher[2:4]
            if s_H * s_W > t_H * t_W:
                # 学生特征图更大，下采样到和教师一致，迫使学生学习如何概括信息
                # (这里是为了和训练保持一致，训练时对齐教师和学生的特征图必须要尺寸相同)
                feature_student_l_2 = F.adaptive_avg_pool2d(
                    feature_student_l_2, (t_H, t_W)
                )

            # 如果学生特征图较小则保持不变，训练时教师会下采样到学生特征图尺寸

            # 转换通道维度 (B, C_feat_stu, H, W) -> (B, C_feat_tea, H, W)
            trans_feature_student = self._transform_layer(feature_student_l_2)

        else:
            # 学生特征，形状应该是 (B, dim_stu)
            feature_student_l_1 = features_student[-1]
            # 转换通道维度 (B, C_feat_stu) -> (B, C_feat_tea)
            trans_feature_student = self._transform_layer(feature_student_l_1)

        return trans_feature_student

    def feature_class_forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回学生的特征和分类结果

        :param x: 输入张量，形状为 (B, C, H, W)
        :return: (学生模型的特征, 分类输出)，形状分别为 (B, C_feat_tea, H_feat, W_feat) 和 (B, num_classes)；或 (B, C_dim_tea) 和 (B, num_classes)
        """
        feature_student_aligned = self.feature_forward(x)
        if self._chosen_feature_level == -2:
            # (B, C_feat_tea, H, W) -> (B, C_feat_tea, 1, 1)
            pooled_feature: torch.Tensor = self._avg_pool(feature_student_aligned)
            # (B, C_feat_tea, 1, 1) -> (B, C_feat_tea)
            flat_feature = pooled_feature.view(pooled_feature.size(0), -1)
            # (B, num_classes)
            # 推理前强制教师分类器为 eval 模式，且关闭梯度
            self._teacher_classifier.eval()
            self._teacher_classifier.requires_grad_(False)
            output = self._teacher_classifier(flat_feature)
        else:
            # (B, C_feat_tea)
            flat_feature = feature_student_aligned
            # (B, num_classes)
            self._teacher_classifier.eval()
            self._teacher_classifier.requires_grad_(False)
            output = self._teacher_classifier(flat_feature)
        return feature_student_aligned, output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回学生的分类结果

        :param x: 输入张量，形状为 (B, C, H, W)
        :return: 分类输出，形状为 (B, num_classes)
        """
        _, output = self.feature_class_forward(x)
        return output
