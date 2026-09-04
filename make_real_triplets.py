#!/usr/bin/env python3
"""把真实视频姿态序列转换成火柴人三连帧。"""

import argparse
import json
from pathlib import Path

import numpy as np

from stickman.skeleton import (
    CANVAS,
    LIMB_BONES,
    render,
)


def load_pose_sequence(path: Path):
    """读取并验证 extract_poses.py 生成的姿态序列。"""
    if not path.exists():
        raise FileNotFoundError(f"姿态文件不存在：{path}")

    data = np.load(path)

    required = {
        "poses",
        "confidence",
        "detected",
        "fps",
        "width",
        "height",
        "video",
    }

    missing = required - set(data.files)
    if missing:
        raise RuntimeError(
            f"姿态文件缺少字段：{sorted(missing)}"
        )

    poses = data["poses"].astype(np.float32)
    confidence = data["confidence"].astype(np.float32)
    detected = data["detected"].astype(bool)

    if poses.ndim != 3 or poses.shape[1:] != (17, 2):
        raise RuntimeError(
            f"poses 应为 (帧数, 17, 2)，实际是 {poses.shape}"
        )

    frame_count = poses.shape[0]

    if confidence.shape != (frame_count, 17):
        raise RuntimeError(
            f"confidence 形状错误：{confidence.shape}"
        )

    if detected.shape != (frame_count,):
        raise RuntimeError(
            f"detected 形状错误：{detected.shape}"
        )

    if not np.isfinite(poses[detected]).all():
        raise RuntimeError("检测成功帧中存在 NaN 或无穷值")

    metadata = {
        "fps": float(data["fps"]),
        "width": int(data["width"]),
        "height": int(data["height"]),
        "video": str(data["video"]),
    }

    return poses, confidence, detected, metadata


def compute_valid_frames(
    poses,
    confidence,
    detected,
    confidence_threshold: float,
    min_body_fraction: float,
    margin: float,
    min_bone_ratio: float,
    max_frame_jump: float,
):
    """判断每一帧是否适合组成三连帧。

    不使用全身最低置信度，因为侧面动作中被遮挡肢体
    可能长期低置信。这里检查身体关节中有多少比例可信。
    """
    # COCO-17 的第 5～16 项是肩、肘、腕、髋、膝、踝。
    body_confidence = confidence[:, 5:]

    confident_fraction = (
        body_confidence >= confidence_threshold
    ).mean(axis=1)

    confidence_ok = (
        confident_fraction >= min_body_fraction
    )

    # 坐标必须留在画布内，并与边缘保持一定距离。
    lower_ok = (poses >= margin).all(axis=(1, 2))
    upper_ok = (poses <= 1.0 - margin).all(axis=(1, 2))
    boundary_ok = lower_ok & upper_ok

    finite_ok = np.isfinite(poses).all(axis=(1, 2))

    # 计算每帧八段四肢骨骼的长度。
    limb_bones = np.asarray(
        LIMB_BONES,
        dtype=np.int32,
    )

    limb_lengths = np.linalg.norm(
        poses[:, limb_bones[:, 0]]
        - poses[:, limb_bones[:, 1]],
        axis=2,
    )

    # 每段骨骼使用整段视频的中位长度作为参考。
    # 中位数不容易被少量折叠帧影响。
    reference_lengths = np.nanmedian(
        limb_lengths,
        axis=0,
    )

    # 某段骨骼短于自身参考长度的一定比例，
    # 就认为这一帧出现了关节折叠。
    bone_ok = (
        limb_lengths
        >= reference_lengths[None, :] * min_bone_ratio
    ).all(axis=1)

    collapsed_frames = np.flatnonzero(
        ~bone_ok
    )

    # 计算相邻两帧中，移动距离最大的关节。
    joint_displacement = np.linalg.norm(
        np.diff(poses, axis=0),
        axis=2,
    )
    max_displacement = np.nanmax(
        joint_displacement,
        axis=1,
    )

    # bad_transitions 保存跳变之后那一帧的索引。
    bad_transitions = (
        np.flatnonzero(
            max_displacement > max_frame_jump
        )
        + 1
    )

    temporal_ok = np.ones(
        len(poses),
        dtype=bool,
    )

    # 跳变两侧的帧都标记为无效。
    temporal_ok[bad_transitions] = False
    temporal_ok[bad_transitions - 1] = False

    valid = (
        detected
        & finite_ok
        & confidence_ok
        & boundary_ok
        & bone_ok
        & temporal_ok
    )

    return (
        valid,
        confident_fraction,
        collapsed_frames,
        bad_transitions,
    )


def write_triplets(
    poses,
    confidence,
    valid_frames,
    source_metadata,
    input_path: Path,
    output_root: Path,
    motion: str,
    kind: str,
    stride: int,
    step: int,
    size: int,
):
    """从姿态序列生成 prev/gt/next 三连帧。"""
    frame_count = poses.shape[0]
    output_root.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    # i 是 prev，i+stride 是 gt，i+2*stride 是 next。
    for i in range(
        0,
        frame_count - 2 * stride,
        step,
    ):
        prev_index = i
        gt_index = i + stride
        next_index = i + 2 * stride

        indices = [
            prev_index,
            gt_index,
            next_index,
        ]

        # 三个关键帧必须全部通过质量检查。
        if not valid_frames[
            prev_index:next_index + 1
        ].all():
            skipped += 1
            continue

        clip_name = (
            f"{input_path.stem}_s{stride}_{i:04d}"
        )
        clip_dir = output_root / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        clip_poses = {
            "prev": poses[prev_index],
            "gt": poses[gt_index],
            "next": poses[next_index],
        }

        clip_confidence = {
            "prev": confidence[prev_index],
            "gt": confidence[gt_index],
            "next": confidence[next_index],
        }

        for tag, pose in clip_poses.items():
            image = render(pose, size=size)
            image.save(clip_dir / f"{tag}.png")

        meta = {
            "motion": motion,
            "kind": kind,
            "stride": stride,
            "size": size,
            "source": {
                "pose_file": str(input_path),
                "video": source_metadata["video"],
                "fps": source_metadata["fps"],
                "width": source_metadata["width"],
                "height": source_metadata["height"],
            },
            "frame_indices": {
                "prev": prev_index,
                "gt": gt_index,
                "next": next_index,
            },
            "poses": {
                tag: pose.tolist()
                for tag, pose in clip_poses.items()
            },
            "confidence": {
                tag: values.tolist()
                for tag, values in clip_confidence.items()
            },
        }

        with open(
            clip_dir / "meta.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                meta,
                file,
                ensure_ascii=False,
                indent=2,
            )

        written += 1

    return written, skipped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "poses",
        type=Path,
        help="extract_poses.py 生成的 .npz 文件",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/real_triplets"),
        help="三连帧输出目录",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="prev→gt 和 gt→next 的帧间隔",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="相邻候选三连帧的起始帧间隔",
    )
    parser.add_argument(
        "--motion",
        default="walk",
        help="动作名称",
    )
    parser.add_argument(
        "--kind",
        default="medium_cyclic",
        help="动作类型",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=CANVAS,
        help="火柴人 PNG 尺寸",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="单关节可信阈值",
    )
    parser.add_argument(
        "--min-body-fraction",
        type=float,
        default=0.7,
        help="身体关节中至少多少比例必须可信",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="姿态与画布边缘的最小距离",
    )
    parser.add_argument(
        "--min-bone-ratio",
        type=float,
        default=0.5,
        help=(
            "四肢骨长相对视频中位骨长的最小比例，"
            "低于该比例视为骨骼折叠"
        ),
    )
    parser.add_argument(
        "--max-frame-jump",
        type=float,
        default=0.12,
        help=(
            "相邻帧单个关节允许的最大移动距离，"
            "使用归一化坐标"
        ),
    )

    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("stride 必须大于 0")

    if not 0.0 <= args.min_bone_ratio <= 1.0:
        raise ValueError(
            "min_bone_ratio 必须在 0 到 1 之间"
        )

    if args.step <= 0:
        raise ValueError("step 必须大于 0")

    poses, confidence, detected, metadata = (
        load_pose_sequence(args.poses)
    )

    (
        valid_frames,
        confident_fraction,
        collapsed_frames,
        bad_transitions,
    ) = compute_valid_frames(
        poses,
        confidence,
        detected,
        args.confidence_threshold,
        args.min_body_fraction,
        args.margin,
        args.min_bone_ratio,
        args.max_frame_jump,
    )

    written, skipped = write_triplets(
        poses=poses,
        confidence=confidence,
        valid_frames=valid_frames,
        source_metadata=metadata,
        input_path=args.poses,
        output_root=args.out,
        motion=args.motion,
        kind=args.kind,
        stride=args.stride,
        step=args.step,
        size=args.size,
    )

    print(f"姿态文件：{args.poses}")
    print(f"总帧数：{len(poses)}")
    print(f"有效帧数：{int(valid_frames.sum())}")
    print(f"无效帧数：{int((~valid_frames).sum())}")

    invalid_indices = np.flatnonzero(~valid_frames)
    print(
        f"无效帧索引："
        f"{invalid_indices.tolist()}"
    )

    print(
        "身体可信比例范围："
        f"{confident_fraction.min():.2f}"
        " ～ "
        f"{confident_fraction.max():.2f}"
    )

    print(
        f"骨骼折叠帧："
        f"{collapsed_frames.tolist()}"
    )

    print(
        f"时间跳变位置："
        f"{bad_transitions.tolist()}"
    )

    print(f"写入三连帧：{written}")
    print(f"跳过候选：{skipped}")
    print(f"输出目录：{args.out}")


if __name__ == "__main__":
    main()