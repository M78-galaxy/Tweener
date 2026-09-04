"""多帧上下文姿态序列 Dataset。"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from stickman.dataset import normalize_endpoints


# 当前 k=3、最大 stride=8，
# 最大绝对时间偏移为 10 帧。
TIME_SCALE = 10.0


class SequencePoseDataset(Dataset):
    """读取 make_real_sequences.py 生成的序列。"""

    def __init__(self, roots):
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
                    "序列根目录名称必须是 "
                    f"s2、s4 或 s8，实际是：{root}"
                )

            stride = int(
                stride_name[1:]
            )

            for meta_path in root.glob(
                "*/meta.json"
            ):
                records[meta_path] = stride

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
        """返回让三个 stride 等概率采样的权重。"""
        counts = Counter(self.strides)

        return torch.tensor(
            [
                1.0 / counts[stride]
                for stride in self.strides
            ],
            dtype=torch.double,
        )

    def __getitem__(self, index):
        meta_path, expected_stride = (
            self.records[index]
        )

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

        left = torch.tensor(
            poses["left"],
            dtype=torch.float32,
        )
        target_pose = torch.tensor(
            poses["target"],
            dtype=torch.float32,
        )
        right = torch.tensor(
            poses["right"],
            dtype=torch.float32,
        )

        left_offsets = torch.tensor(
            offsets["left"],
            dtype=torch.float32,
        )
        right_offsets = torch.tensor(
            offsets["right"],
            dtype=torch.float32,
        )

        expected_context_shape = (
            context,
            17,
            2,
        )

        if (
            tuple(left.shape)
            != expected_context_shape
        ):
            raise RuntimeError(
                f"{meta_path} 的 left 形状错误："
                f"{tuple(left.shape)}"
            )

        if (
            tuple(right.shape)
            != expected_context_shape
        ):
            raise RuntimeError(
                f"{meta_path} 的 right 形状错误："
                f"{tuple(right.shape)}"
            )

        if target_pose.shape != (17, 2):
            raise RuntimeError(
                f"{meta_path} 的 target 形状错误："
                f"{tuple(target_pose.shape)}"
            )

        if left_offsets.shape != (context,):
            raise RuntimeError(
                f"{meta_path} 的左侧时间偏移错误"
            )

        if right_offsets.shape != (context,):
            raise RuntimeError(
                f"{meta_path} 的右侧时间偏移错误"
            )

        if not torch.all(
            left_offsets < 0
        ):
            raise RuntimeError(
                f"{meta_path} 的左侧时间必须小于 0"
            )

        if not torch.all(
            right_offsets > 0
        ):
            raise RuntimeError(
                f"{meta_path} 的右侧时间必须大于 0"
            )

        if left_offsets[-1].item() != -stride:
            raise RuntimeError(
                f"{meta_path} 的左端点偏移错误"
            )

        if right_offsets[0].item() != stride:
            raise RuntimeError(
                f"{meta_path} 的右端点偏移错误"
            )

        if not torch.isfinite(left).all():
            raise RuntimeError(
                f"{meta_path} 的 left 存在非有限值"
            )

        if not torch.isfinite(
            target_pose
        ).all():
            raise RuntimeError(
                f"{meta_path} 的 target 存在非有限值"
            )

        if not torch.isfinite(right).all():
            raise RuntimeError(
                f"{meta_path} 的 right 存在非有限值"
            )

        # v2 的两个端点仍然是：
        # 左上下文最后一帧和右上下文第一帧。
        prev_pose = left[-1]
        next_pose = right[0]

        (
            _,
            _,
            center,
            scale,
        ) = normalize_endpoints(
            prev_pose,
            next_pose,
        )

        left_normalized = (
            left - center
        ) / scale

        target_normalized = (
            target_pose - center
        ) / scale

        right_normalized = (
            right - center
        ) / scale

        # 六个姿态 token，每个 token 34 维。
        tokens = torch.cat(
            (
                left_normalized,
                right_normalized,
            ),
            dim=0,
        ).reshape(
            context * 2,
            34,
        )

        time_offsets = torch.cat(
            (
                left_offsets,
                right_offsets,
            )
        )

        # 连续时间特征缩放到约 [-1, 1]。
        time_features = (
            time_offsets / TIME_SCALE
        ).unsqueeze(-1)

        prev_normalized = (
            left_normalized[-1]
        )
        next_normalized = (
            right_normalized[0]
        )

        lerp_normalized = (
            prev_normalized
            + next_normalized
        ) * 0.5

        residual_target = (
            target_normalized
            - lerp_normalized
        )

        return {
            "tokens": tokens,
            "time": time_features,
            "time_offsets": (
                time_offsets.to(
                    dtype=torch.int64
                )
            ),
            "target": (
                residual_target.reshape(-1)
            ),
            "lerp": lerp_normalized,
            "gt": target_normalized,
            "center": center,
            "scale": scale,
            "stride": stride,
            "context": context,
            "clip": meta_path.parent.name,
            "motion": meta["motion"],
            "kind": meta["kind"],
        }