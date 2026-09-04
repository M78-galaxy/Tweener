"""真实姿态三连帧的 PyTorch Dataset。"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

MIN_SCALE = 1e-6


def joint_midpoint(pose, left_index, right_index):
    """计算左右两个关节的中点。"""
    return (
        pose[left_index] + pose[right_index]
    ) * 0.5


def normalize_triplet(prev, gt, next_pose):
    """使用前后帧共享的中心和尺度归一化三连帧。

    中心：前后两帧髋部中心的平均位置。
    尺度：前后两帧躯干长度的平均值。
    """
    prev_hip = joint_midpoint(
        prev,
        LEFT_HIP,
        RIGHT_HIP,
    )
    next_hip = joint_midpoint(
        next_pose,
        LEFT_HIP,
        RIGHT_HIP,
    )

    prev_shoulder = joint_midpoint(
        prev,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
    )
    next_shoulder = joint_midpoint(
        next_pose,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
    )

    center = (prev_hip + next_hip) * 0.5

    prev_torso = torch.linalg.vector_norm(
        prev_shoulder - prev_hip
    )
    next_torso = torch.linalg.vector_norm(
        next_shoulder - next_hip
    )

    scale = (prev_torso + next_torso) * 0.5
    scale = torch.clamp(scale, min=MIN_SCALE)

    prev_normalized = (prev - center) / scale
    gt_normalized = (gt - center) / scale
    next_normalized = (next_pose - center) / scale

    return (
        prev_normalized,
        gt_normalized,
        next_normalized,
        center,
        scale,
    )

def normalize_endpoints(prev, next_pose):
    """归一化推理时只有前后两帧的情况。"""
    middle = (prev + next_pose) * 0.5

    (
        prev_normalized,
        _,
        next_normalized,
        center,
        scale,
    ) = normalize_triplet(
        prev,
        middle,
        next_pose,
    )

    return (
        prev_normalized,
        next_normalized,
        center,
        scale,
    )

class PoseTripletDataset(Dataset):
    """读取 make_real_triplets.py 生成的三连帧。"""

    def __init__(self, roots):
        self.meta_paths = []

        for root in roots:
            root = Path(root)

            if not root.is_dir():
                raise FileNotFoundError(
                    f"数据目录不存在：{root}"
                )

            self.meta_paths.extend(
                root.glob("*/meta.json")
            )

        self.meta_paths = sorted(
            set(self.meta_paths),
            key=str,
        )

        if not self.meta_paths:
            raise RuntimeError(
                "没有找到任何三连帧 meta.json"
            )

    def __len__(self):
        return len(self.meta_paths)

    def __getitem__(self, index):
        meta_path = self.meta_paths[index]

        with meta_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            meta = json.load(file)

        try:
            poses = meta["poses"]

            prev = torch.tensor(
                poses["prev"],
                dtype=torch.float32,
            )
            gt = torch.tensor(
                poses["gt"],
                dtype=torch.float32,
            )
            next_pose = torch.tensor(
                poses["next"],
                dtype=torch.float32,
            )
        except KeyError as error:
            raise RuntimeError(
                f"{meta_path} 缺少字段：{error}"
            ) from error

        for name, pose in (
            ("prev", prev),
            ("gt", gt),
            ("next", next_pose),
        ):
            if pose.shape != (17, 2):
                raise RuntimeError(
                    f"{meta_path} 中 {name} "
                    f"形状错误：{tuple(pose.shape)}"
                )

            if not torch.isfinite(pose).all():
                raise RuntimeError(
                    f"{meta_path} 中 {name} "
                    "存在 NaN 或无穷值"
                )

        (
            prev_normalized,
            gt_normalized,
            next_normalized,
            center,
            scale,
        ) = normalize_triplet(
            prev,
            gt,
            next_pose,
        )

        lerp_normalized = (
            prev_normalized + next_normalized
        ) * 0.5

        # 模型学习对 lerp 的修正量，而不是整套绝对坐标。
        residual_target = (
            gt_normalized - lerp_normalized
        )

        model_input = torch.cat(
            (
                prev_normalized.reshape(-1),
                next_normalized.reshape(-1),
            )
        )

        return {
            "input": model_input,
            "target": residual_target.reshape(-1),
            "prev": prev_normalized,
            "lerp": lerp_normalized,
            "gt": gt_normalized,
            "next": next_normalized,
            "center": center,
            "scale": scale,
            "clip": meta_path.parent.name,
            "motion": meta["motion"],
            "kind": meta["kind"],
            "stride": int(meta["stride"]),
        }