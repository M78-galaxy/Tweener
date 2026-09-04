"""多帧上下文火柴人栅格 Dataset。"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from stickman.sequence_dataset import TIME_SCALE
from stickman.skeleton import CANVAS, render


def pose_to_foreground(
    pose,
    size=CANVAS,
):
    """把姿态渲染为前景为 1、背景为 0 的张量。"""
    pose = np.asarray(
        pose,
        dtype=np.float32,
    )

    if pose.shape != (17, 2):
        raise ValueError(
            "pose 必须是 (17, 2)，"
            f"实际是 {pose.shape}"
        )

    if not np.isfinite(pose).all():
        raise ValueError(
            "pose 中存在 NaN 或无穷值"
        )

    image = render(
        pose,
        size=size,
    )

    pixels = np.array(
        image,
        dtype=np.float32,
        copy=True,
    )

    # render 输出黑线白底。
    # 神经网络使用前景 1、背景 0。
    foreground = 1.0 - pixels / 255.0

    return torch.from_numpy(
        foreground
    )


class RasterSequenceDataset(Dataset):
    """读取多帧姿态并实时渲染为栅格图。"""

    def __init__(
        self,
        roots,
        size=CANVAS,
    ):
        self.size = int(size)

        if self.size <= 0:
            raise ValueError(
                "size 必须大于 0"
            )

        records = {}

        for root in roots:
            root = Path(root)

            if not root.is_dir():
                raise FileNotFoundError(
                    f"序列目录不存在：{root}"
                )

            stride_name = root.name

            if not stride_name.startswith("s"):
                raise ValueError(
                    "序列根目录必须是 "
                    f"s2、s4 或 s8：{root}"
                )

            expected_stride = int(
                stride_name[1:]
            )

            for meta_path in root.glob(
                "*/meta.json"
            ):
                records[
                    meta_path
                ] = expected_stride

        self.records = sorted(
            records.items(),
            key=lambda item: str(item[0]),
        )

        if not self.records:
            raise RuntimeError(
                "没有找到任何序列 meta.json"
            )

        self.strides = [
            stride
            for _, stride in self.records
        ]

    def __len__(self):
        return len(self.records)

    def stride_sampling_weights(self):
        """让 s2、s4、s8 获得相同采样概率。"""
        counts = Counter(self.strides)

        return torch.tensor(
            [
                1.0 / counts[stride]
                for stride in self.strides
            ],
            dtype=torch.double,
        )

    def __getitem__(self, index):
        (
            meta_path,
            expected_stride,
        ) = self.records[index]

        with meta_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            meta = json.load(file)

        stride = int(meta["stride"])
        context = int(meta["context"])

        if stride != expected_stride:
            raise RuntimeError(
                f"{meta_path} 的 stride={stride}，"
                f"但所在目录是 s{expected_stride}"
            )

        poses = meta["poses"]
        offsets = meta["time_offsets"]

        left = np.asarray(
            poses["left"],
            dtype=np.float32,
        )
        target_pose = np.asarray(
            poses["target"],
            dtype=np.float32,
        )
        right = np.asarray(
            poses["right"],
            dtype=np.float32,
        )

        expected_context_shape = (
            context,
            17,
            2,
        )

        if left.shape != expected_context_shape:
            raise RuntimeError(
                f"{meta_path} 的 left "
                f"形状错误：{left.shape}"
            )

        if right.shape != expected_context_shape:
            raise RuntimeError(
                f"{meta_path} 的 right "
                f"形状错误：{right.shape}"
            )

        if target_pose.shape != (17, 2):
            raise RuntimeError(
                f"{meta_path} 的 target "
                f"形状错误：{target_pose.shape}"
            )

        if not np.isfinite(left).all():
            raise RuntimeError(
                f"{meta_path} 的 left "
                "存在非有限值"
            )

        if not np.isfinite(right).all():
            raise RuntimeError(
                f"{meta_path} 的 right "
                "存在非有限值"
            )

        if not np.isfinite(
            target_pose
        ).all():
            raise RuntimeError(
                f"{meta_path} 的 target "
                "存在非有限值"
            )

        left_offsets = torch.tensor(
            offsets["left"],
            dtype=torch.float32,
        )
        right_offsets = torch.tensor(
            offsets["right"],
            dtype=torch.float32,
        )

        if left_offsets.shape != (context,):
            raise RuntimeError(
                f"{meta_path} 左时间偏移错误"
            )

        if right_offsets.shape != (context,):
            raise RuntimeError(
                f"{meta_path} 右时间偏移错误"
            )

        if not torch.all(
            left_offsets < 0
        ):
            raise RuntimeError(
                f"{meta_path} 左时间必须小于 0"
            )

        if not torch.all(
            right_offsets > 0
        ):
            raise RuntimeError(
                f"{meta_path} 右时间必须大于 0"
            )

        if left_offsets[-1].item() != -stride:
            raise RuntimeError(
                f"{meta_path} 左端点偏移错误"
            )

        if right_offsets[0].item() != stride:
            raise RuntimeError(
                f"{meta_path} 右端点偏移错误"
            )

        all_poses = np.concatenate(
            (
                left,
                right,
            ),
            axis=0,
        )

        # 6 张前景图，形状 (6, H, W)。
        context_images = torch.stack(
            [
                pose_to_foreground(
                    pose,
                    self.size,
                )
                for pose in all_poses
            ],
            dim=0,
        )

        target_image = (
            pose_to_foreground(
                target_pose,
                self.size,
            )
            .unsqueeze(0)
        )

        time_offsets = torch.cat(
            (
                left_offsets,
                right_offsets,
            )
        )

        time_values = (
            time_offsets / TIME_SCALE
        )

        # 每个时间偏移扩展成一张常数图。
        # 这样卷积网络得到与 Transformer
        # 相同的 time-to-arrival 信息。
        time_maps = (
            time_values[:, None, None]
            .expand(
                context * 2,
                self.size,
                self.size,
            )
        )

        # 前 6 通道是图像，后 6 通道是时间。
        model_input = torch.cat(
            (
                context_images,
                time_maps,
            ),
            dim=0,
        )

        prev_image = (
            context_images[
                context - 1
            ].unsqueeze(0)
        )
        next_image = (
            context_images[
                context
            ].unsqueeze(0)
        )

        # 纯像素平均：通常会出现双线和灰线。
        pixel_lerp = (
            prev_image + next_image
        ) * 0.5

        # 坐标先 lerp 再渲染：线条清楚，
        # 但旋转动作的中间姿态可能不正确。
        pose_lerp = (
            pose_to_foreground(
                (
                    left[-1]
                    + right[0]
                ) * 0.5,
                self.size,
            )
            .unsqueeze(0)
        )

        return {
            "input": model_input,
            "left_poses": torch.from_numpy(
                left.copy()
            ),
            "right_poses": torch.from_numpy(
                right.copy()
            ),
            "left_offsets": (
                left_offsets.to(
                    dtype=torch.int64
                )
            ),
            "right_offsets": (
                right_offsets.to(
                    dtype=torch.int64
                )
            ),
            "meta_path": str(meta_path),
            "context_images": (
                context_images
            ),
            "time_maps": time_maps,
            "time_offsets": (
                time_offsets.to(
                    dtype=torch.int64
                )
            ),
            "target": target_image,
            "pixel_lerp": pixel_lerp,
            "pose_lerp": pose_lerp,
            "prev_image": prev_image,
            "next_image": next_image,
            "target_pose": torch.from_numpy(
                target_pose.copy()
            ),
            "stride": stride,
            "context": context,
            "clip": meta_path.parent.name,
            "motion": meta["motion"],
            "kind": meta["kind"],
        }