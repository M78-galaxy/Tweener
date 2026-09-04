#!/usr/bin/env python3
"""从真实姿态序列生成多帧上下文样本。"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from make_real_triplets import (
    compute_valid_frames,
    load_pose_sequence,
)
from stickman.skeleton import CANVAS


def write_sequences(
    poses,
    confidence,
    valid_frames,
    source_metadata,
    input_path,
    output_root,
    motion,
    kind,
    stride,
    context,
    size,
    quality_filter,
):
    """生成左 k 帧、目标帧、右 k 帧序列。"""
    stride_root = (
        output_root
        / f"k{context}"
        / f"s{stride}"
    )
    stride_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0
    skipped = 0
    candidates = 0

    frame_count = len(poses)

    # 和 v2 保持相同的 prev 起点网格。
    for prev_index in range(
        0,
        frame_count - 2 * stride,
        stride,
    ):
        left_start = (
            prev_index - context + 1
        )

        target_index = (
            prev_index + stride
        )

        next_index = (
            prev_index + 2 * stride
        )

        right_end = (
            next_index + context - 1
        )

        # 视频边界不足以提供完整上下文。
        if left_start < 0:
            continue

        if right_end >= frame_count:
            continue

        candidates += 1

        # 从最左输入帧到最右输入帧之间，
        # 所有姿态都必须通过质量检查。
        if not valid_frames[
            left_start:right_end + 1
        ].all():
            skipped += 1
            continue

        left_indices = list(
            range(
                left_start,
                prev_index + 1,
            )
        )

        right_indices = list(
            range(
                next_index,
                right_end + 1,
            )
        )

        left_offsets = [
            index - target_index
            for index in left_indices
        ]

        right_offsets = [
            index - target_index
            for index in right_indices
        ]

        clip_name = (
            f"{input_path.stem}"
            f"_s{stride}"
            f"_k{context}"
            f"_{prev_index:04d}"
        )

        clip_dir = (
            stride_root / clip_name
        )
        clip_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        meta = {
            "motion": motion,
            "kind": kind,
            "stride": stride,
            "context": context,
            "size": size,
            "source": {
                "pose_file": str(input_path),
                "video": source_metadata["video"],
                "fps": source_metadata["fps"],
                "width": source_metadata["width"],
                "height": source_metadata["height"],
            },
            "quality_filter": quality_filter,
            "frame_indices": {
                "left": left_indices,
                "target": target_index,
                "right": right_indices,
            },
            "time_offsets": {
                "left": left_offsets,
                "right": right_offsets,
            },
            "poses": {
                "left": (
                    poses[left_indices]
                    .tolist()
                ),
                "target": (
                    poses[target_index]
                    .tolist()
                ),
                "right": (
                    poses[right_indices]
                    .tolist()
                ),
            },
            "confidence": {
                "left": (
                    confidence[left_indices]
                    .tolist()
                ),
                "target": (
                    confidence[target_index]
                    .tolist()
                ),
                "right": (
                    confidence[right_indices]
                    .tolist()
                ),
            },
        }

        with (
            clip_dir / "meta.json"
        ).open(
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

    return written, skipped, candidates, stride_root


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "poses",
        help="extract_poses.py 生成的 .npz",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="该视频的序列输出根目录",
    )
    parser.add_argument(
        "--motion",
        default=None,
    )
    parser.add_argument(
        "--kind",
        required=True,
    )
    parser.add_argument(
        "--strides",
        nargs="+",
        type=int,
        default=[2, 4, 8],
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=CANVAS,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--min-body-fraction",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--min-bone-ratio",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--max-frame-jump",
        type=float,
        default=0.12,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.context < 1:
        raise ValueError(
            "context 必须至少为 1"
        )

    if any(
        stride <= 0
        for stride in args.strides
    ):
        raise ValueError(
            "所有 stride 都必须大于 0"
        )

    input_path = Path(args.poses)
    output_root = Path(args.out)

    motion = (
        args.motion
        if args.motion is not None
        else input_path.stem
    )

    (
        poses,
        confidence,
        detected,
        source_metadata,
    ) = load_pose_sequence(input_path)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
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

    quality_filter = {
        "confidence_threshold": (
            args.confidence_threshold
        ),
        "min_body_fraction": (
            args.min_body_fraction
        ),
        "margin": args.margin,
        "min_bone_ratio": (
            args.min_bone_ratio
        ),
        "max_frame_jump": (
            args.max_frame_jump
        ),
    }

    config_root = (
        output_root / f"k{args.context}"
    )
    config_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "pose_file": str(input_path),
        "motion": motion,
        "kind": args.kind,
        "strides": args.strides,
        "context": args.context,
        "size": args.size,
        "quality_filter": quality_filter,
        "total_frames": len(poses),
        "valid_frames": int(
            valid_frames.sum()
        ),
        "invalid_frames": np.flatnonzero(
            ~valid_frames
        ).tolist(),
        "collapsed_frames": (
            collapsed_frames.tolist()
        ),
        "bad_transitions": (
            bad_transitions.tolist()
        ),
    }

    with (
        config_root / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"姿态文件：{input_path}")
    print(f"总帧数：{len(poses)}")
    print(
        "有效帧数："
        f"{int(valid_frames.sum())}"
    )
    print(
        "身体可信比例范围："
        f"{confident_fraction.min():.2f}"
        " ～ "
        f"{confident_fraction.max():.2f}"
    )
    print(
        "骨骼折叠帧数："
        f"{len(collapsed_frames)}"
    )
    print(
        "时间跳变数："
        f"{len(bad_transitions)}"
    )

    for stride in args.strides:
        (
            written,
            skipped,
            candidates,
            stride_root,
        ) = write_sequences(
            poses=poses,
            confidence=confidence,
            valid_frames=valid_frames,
            source_metadata=source_metadata,
            input_path=input_path,
            output_root=output_root,
            motion=motion,
            kind=args.kind,
            stride=stride,
            context=args.context,
            size=args.size,
            quality_filter=quality_filter,
        )

        print(
            f"s{stride}："
            f"写入 {written}，"
            f"跳过 {skipped}，"
            f"候选 {candidates}"
        )
        print(
            f"输出目录：{stride_root}"
        )


if __name__ == "__main__":
    main()