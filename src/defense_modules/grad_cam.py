"""
GRAD-CAM 可视化模块

对比展示以下几种情况的 Grad-CAM 可视化结果：

1. 原始模型 + 原始图像
2. 原始模型 + 触发图像
3. 后门模型 + 原始图像
4. 后门模型 + 触发图像

* Repo: https://github.com/jacobgil/pytorch-grad-cam
* Ref: Selvaraju R R, Cogswell M, Das A, et al. Grad-cam: Visual explanations from deep networks via gradient-based localization[C]//Proceedings of the IEEE international conference on computer vision. 2017: 618-626.
"""

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

from utils.data import DatasetWithInfo, DataLoaderDataIter, TransformedDataset
from data_augs import MakeSimpleTransforms
from utils.funcs import (
    auto_select_device,
    temp_seed,
    get_timestamp,
    print_section,
    auto_num_workers,
)

from modules.abc import TriggerGenerator
from models.abc import ModelBase
from defense_modules.abc import DefenseModule
from configs import TENSORBOARD_LOGS_PATH, CHECKPOINTS_SAVE_PATH

_ckpt_save_dir = os.path.join(CHECKPOINTS_SAVE_PATH, "grad_cam")


class GradCAMVis(DefenseModule):
    def __init__(
        self,
        test_id: str,
        model: ModelBase,
        benign_model: ModelBase,
        dataset_info: DatasetWithInfo,
        trigger_generator: TriggerGenerator,
        *args,
        num_images: int = 8,
        seed=42,
        **kwargs,
    ):
        """
        初始化 GradCAM 可视化模块

        :param test_id: 用来标记本次 NC 运行的测试 ID
        :param model: 待检测模型
        :param benign_model: 良性模型
        :param dataset_info: 数据集信息
        :param trigger_generator: 触发器生成器对象
        :param num_images: 可视化的图像数量
        :param seed: 随机种子
        """
        self._test_id = test_id
        self._model = copy.deepcopy(model)  # 不影响原模型
        self._benign_model = copy.deepcopy(benign_model)  # 不影响原模型
        self._transforms_maker = MakeSimpleTransforms(input_shape=dataset_info.shape)
        self._dataset_info = dataset_info
        self._trigger_generator = trigger_generator
        self._seed = seed
        self._save_dir = os.path.join(_ckpt_save_dir, test_id)
        self._data_loader = DataLoader(
            dataset=TransformedDataset(
                dataset_info.val_set,
                transform=self._transforms_maker.normalize_standardize,
            ),
            batch_size=num_images,
            shuffle=True,
            num_workers=auto_num_workers(),
        )
        os.makedirs(self._save_dir, exist_ok=True)

    @classmethod
    def is_mitigation(cls) -> bool:
        return False

    def detect(self) -> dict:
        """
        执行 Grad-CAM 可视化检测

        :return: 检测结果字典
        """
        tensorboard_log_id = f"grad_cam_{self._test_id}"
        tensorboard_log_dir = os.path.join(TENSORBOARD_LOGS_PATH, tensorboard_log_id)
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        print_section(f"Grad-CAM: {self._test_id}")

        device = auto_select_device()

        self._model.to(device)
        self._model.eval()

        self._benign_model.to(device)
        self._benign_model.eval()

        with temp_seed(self._seed):
            # 取出一批图像
            data_batch = next(iter(self._data_loader))
            images, _ = data_batch
            images: torch.Tensor = images.to(device)  # shape (B, C, H, W)
            triggered_images = self._trigger_generator.apply_trigger(images)
            # 转换回 [0, 1] 范围内的 numpy 图像
            destd_images: torch.Tensor = self._transforms_maker.destandardize(images)
            destd_images_np = (
                destd_images.permute(0, 2, 3, 1).cpu().numpy()
            )  # shape (B, H, W, C)
            destd_triggered_images: torch.Tensor = self._transforms_maker.destandardize(
                triggered_images
            )
            destd_triggered_images_np = (
                destd_triggered_images.permute(0, 2, 3, 1).cpu().numpy()
            )  # shape (B, H, W, C)

            n_batch = images.size(0)

            # ------------------ 在后门模型上测试
            # 存储 (原图, 触发图) 的 Grad-CAM 结果
            backdoor_grad_cam_results: list[tuple[np.ndarray, np.ndarray]] = []
            # 初始化 Grad-CAM
            with GradCAM(
                model=self._model,
                target_layers=[self._model.get_gradcam_feature_layer()],
            ) as cam:
                # 先输入正常图像进行测试
                grayscale_cam = cam(
                    input_tensor=images,
                    targets=None,  # 默认使用模型预测置信度最高的类别作为目标
                )  # shape (B, H, W)

                # 再输入触发图像进行测试
                triggered_grayscale_cam = cam(
                    input_tensor=triggered_images,
                    targets=None,
                )  # shape (B, H, W)

                for i in range(n_batch):
                    vis = show_cam_on_image(
                        img=destd_images_np[i],
                        mask=grayscale_cam[i],
                        use_rgb=True,
                    )  # shape (H, W, 3), np.uint8

                    triggered_vis = show_cam_on_image(
                        img=destd_triggered_images_np[i],
                        mask=triggered_grayscale_cam[i],
                        use_rgb=True,
                    )  # shape (H, W, 3), np.uint8

                    backdoor_grad_cam_results.append((vis, triggered_vis))

            # ------------------ 在良性模型上测试
            # 存储 (原图, 触发图) 的 Grad-CAM 结果
            benign_grad_cam_results: list[tuple[np.ndarray, np.ndarray]] = []
            # 初始化 Grad-CAM
            with GradCAM(
                model=self._benign_model,
                target_layers=[self._benign_model.get_gradcam_feature_layer()],
            ) as cam:
                # 先输入正常图像进行测试
                grayscale_cam = cam(
                    input_tensor=images,
                    targets=None,  # 默认使用模型预测置信度最高的类别作为目标
                )  # shape (B, H, W)

                # 再输入触发图像进行测试
                triggered_grayscale_cam = cam(
                    input_tensor=triggered_images,
                    targets=None,
                )  # shape (B, H, W)

                for i in range(n_batch):
                    vis = show_cam_on_image(
                        img=destd_images_np[i],
                        mask=grayscale_cam[i],
                        use_rgb=True,
                    )  # shape (H, W, 3), np.uint8

                    triggered_vis = show_cam_on_image(
                        img=destd_triggered_images_np[i],
                        mask=triggered_grayscale_cam[i],
                        use_rgb=True,
                    )  # shape (H, W, 3), np.uint8

                    benign_grad_cam_results.append((vis, triggered_vis))

            # ------------------ 保存图像
            benign_save_dir = os.path.join(self._save_dir, f"benign")
            os.makedirs(benign_save_dir, exist_ok=True)
            for i, (benign_vis, benign_triggered_vis) in enumerate(
                benign_grad_cam_results
            ):
                img_benign_vis = Image.fromarray(benign_vis)
                img_benign_triggered_vis = Image.fromarray(benign_triggered_vis)

                img_benign_vis.save(
                    os.path.join(
                        benign_save_dir,
                        f"benign_vis_{i}.png",
                    )
                )

                img_benign_triggered_vis.save(
                    os.path.join(
                        benign_save_dir,
                        f"benign_triggered_vis_{i}.png",
                    )
                )

                # 拼接后存入 TensorBoard 日志
                concat_benign = Image.new(
                    "RGB",
                    (benign_vis.shape[1] * 2, benign_vis.shape[0]),
                )

                concat_benign.paste(img_benign_vis, (0, 0))
                concat_benign.paste(img_benign_triggered_vis, (benign_vis.shape[1], 0))

                tb_writer.add_image(
                    tag=f"Benign Grad-CAM/{i}",
                    img_tensor=np.array(concat_benign),  # shape (H, W*2, 3)
                    dataformats="HWC",
                )

            backdoor_save_dir = os.path.join(self._save_dir, f"backdoor")
            os.makedirs(backdoor_save_dir, exist_ok=True)
            for i, (backdoor_vis, backdoor_triggered_vis) in enumerate(
                backdoor_grad_cam_results
            ):
                img_backdoor_vis = Image.fromarray(backdoor_vis)
                img_backdoor_triggered_vis = Image.fromarray(backdoor_triggered_vis)

                img_backdoor_vis.save(
                    os.path.join(
                        backdoor_save_dir,
                        f"backdoor_vis_{i}.png",
                    )
                )

                img_backdoor_triggered_vis.save(
                    os.path.join(
                        backdoor_save_dir,
                        f"backdoor_triggered_vis_{i}.png",
                    )
                )

                # 拼接后存入 TensorBoard 日志
                concat_backdoor = Image.new(
                    "RGB",
                    (backdoor_vis.shape[1] * 2, backdoor_vis.shape[0]),
                )

                concat_backdoor.paste(img_backdoor_vis, (0, 0))
                concat_backdoor.paste(
                    img_backdoor_triggered_vis, (backdoor_vis.shape[1], 0)
                )

                tb_writer.add_image(
                    tag=f"Backdoor Grad-CAM/{i}",
                    img_tensor=np.array(concat_backdoor),  # shape (H, W*2, 3)
                    dataformats="HWC",
                )

        tb_writer.close()

        return {
            "save_path": {
                "backdoor_grad_cam_results": backdoor_save_dir,
                "benign_grad_cam_results": benign_save_dir,
            },
        }
