"""生成预测、真值和叠图的序列诊断动画。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from assemble_stickman_video import (
    write_gif,
    write_mp4,
)
from stickman.skeleton import (
    bone_length_error,
    mpjpe,
    render,
    render_overlay,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成火柴人序列诊断动画",
    )

    parser.add_argument(
        "manifest",
        type=Path,
        help="连续序列 manifest.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="批量推理输出目录",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
        help="包含真值 poses 的 .npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/v5_sequence_comparison"
        ),
        help="诊断输出目录",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="覆盖 manifest 中的输出 FPS",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="单个火柴人面板尺寸",
    )

    return parser.parse_args()


def load_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"manifest 不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    requests = manifest.get(
        "requests"
    )

    if (
        not isinstance(requests, list)
        or not requests
    ):
        raise ValueError(
            "manifest.requests "
            "必须是非空列表"
        )

    return manifest


def make_comparison_frame(
    prediction,
    target,
    request_id,
    target_frame,
    size,
):
    position_error = mpjpe(
        prediction,
        target,
        size=size,
    )
    bone_error = bone_length_error(
        prediction,
        target,
        size=size,
    )

    prediction_image = render(
        prediction,
        size=size,
    ).convert("RGB")

    target_image = render(
        target,
        size=size,
    ).convert("RGB")

    overlay_image = render_overlay(
        prediction,
        target,
        size=size,
    )

    header_height = 40

    canvas = Image.new(
        "RGB",
        (
            size * 3,
            size + header_height,
        ),
        (255, 255, 255),
    )

    canvas.paste(
        prediction_image,
        (0, header_height),
    )
    canvas.paste(
        target_image,
        (size, header_height),
    )
    canvas.paste(
        overlay_image,
        (size * 2, header_height),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (8, 4),
        "Prediction",
        fill=(0, 0, 0),
    )
    draw.text(
        (size + 8, 4),
        "Target",
        fill=(0, 0, 0),
    )
    draw.text(
        (size * 2 + 8, 4),
        "Overlay: pred red / target black",
        fill=(0, 0, 0),
    )

    draw.text(
        (8, 22),
        (
            f"{request_id} | "
            f"frame={target_frame} | "
            f"MPJPE={position_error:.3f}px | "
            f"bone={bone_error:.3f}px"
        ),
        fill=(0, 0, 0),
    )

    metrics = {
        "request_id": request_id,
        "target_frame": target_frame,
        "mpjpe_px": position_error,
        "bone_err_px": bone_error,
    }

    return canvas, metrics


def main():
    args = parse_args()

    if args.size <= 0:
        raise ValueError(
            "size 必须大于 0"
        )

    if not args.predictions.is_dir():
        raise FileNotFoundError(
            "预测目录不存在："
            f"{args.predictions}"
        )

    if not args.poses.is_file():
        raise FileNotFoundError(
            f"姿态文件不存在：{args.poses}"
        )

    manifest = load_manifest(
        args.manifest
    )

    fps = (
        float(args.fps)
        if args.fps is not None
        else float(
            manifest["output_fps"]
        )
    )

    if fps <= 0:
        raise ValueError(
            "fps 必须大于 0"
        )

    with np.load(
        args.poses
    ) as pose_data:
        target_poses = np.asarray(
            pose_data["poses"],
            dtype=np.float32,
        )

    frames = []
    metrics_rows = []
    target_frames = []

    for entry in manifest["requests"]:
        request_path = (
            args.manifest.parent
            / entry
        )

        with request_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            request = json.load(file)

        request_id = str(
            request["request_id"]
        )
        target_frame = int(
            request["target_frame"]
        )

        if not (
            0
            <= target_frame
            < len(target_poses)
        ):
            raise IndexError(
                "目标帧超出姿态序列："
                f"{target_frame}"
            )

        prediction_path = (
            args.predictions
            / request_id
            / "prediction_pose.npy"
        )

        if not prediction_path.is_file():
            raise FileNotFoundError(
                "预测姿态不存在："
                f"{prediction_path}"
            )

        prediction = np.asarray(
            np.load(prediction_path),
            dtype=np.float32,
        )
        target = target_poses[
            target_frame
        ]

        if prediction.shape != (17, 2):
            raise ValueError(
                "预测姿态形状错误："
                f"{prediction.shape}"
            )

        if target.shape != (17, 2):
            raise ValueError(
                "真值姿态形状错误："
                f"{target.shape}"
            )

        if not np.isfinite(
            prediction
        ).all():
            raise ValueError(
                f"{request_id} 的预测存在非有限值"
            )

        if not np.isfinite(
            target
        ).all():
            raise ValueError(
                f"目标帧 {target_frame} "
                "存在非有限值"
            )

        frame, metrics = (
            make_comparison_frame(
                prediction,
                target,
                request_id,
                target_frame,
                args.size,
            )
        )

        frames.append(
            frame
        )
        metrics_rows.append(
            metrics
        )
        target_frames.append(
            target_frame
        )

    if any(
        current <= previous
        for previous, current in zip(
            target_frames,
            target_frames[1:],
        )
    ):
        raise ValueError(
            "目标帧顺序不是严格递增"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mp4_path = (
        args.out / "comparison.mp4"
    )
    gif_path = (
        args.out / "comparison.gif"
    )
    metrics_path = (
        args.out / "metrics.csv"
    )
    result_path = (
        args.out / "comparison_result.json"
    )

    write_mp4(
        frames,
        mp4_path,
        fps,
    )
    gif_duration_ms = write_gif(
        frames,
        gif_path,
        fps,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "request_id",
                "target_frame",
                "mpjpe_px",
                "bone_err_px",
            ),
        )
        writer.writeheader()
        writer.writerows(
            metrics_rows
        )

    mean_mpjpe = float(
        np.mean(
            [
                row["mpjpe_px"]
                for row in metrics_rows
            ]
        )
    )
    mean_bone = float(
        np.mean(
            [
                row["bone_err_px"]
                for row in metrics_rows
            ]
        )
    )

    worst = max(
        metrics_rows,
        key=lambda row: row["mpjpe_px"],
    )

    result = {
        "schema_version": 1,
        "sequence_id": manifest.get(
            "sequence_id"
        ),
        "frame_count": len(frames),
        "fps": fps,
        "duration_seconds": (
            len(frames) / fps
        ),
        "gif_frame_duration_ms": (
            gif_duration_ms
        ),
        "panel_size": args.size,
        "video_size": [
            args.size * 3,
            args.size + 40,
        ],
        "mean_mpjpe_px": mean_mpjpe,
        "mean_bone_err_px": mean_bone,
        "worst_mpjpe": worst,
        "outputs": {
            "mp4": str(mp4_path),
            "gif": str(gif_path),
            "metrics": str(metrics_path),
        },
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("诊断序列生成完成")
    print("帧数：", len(frames))
    print("帧率：", fps)
    print(
        "平均 MPJPE：",
        f"{mean_mpjpe:.4f} px",
    )
    print(
        "平均骨长误差：",
        f"{mean_bone:.4f} px",
    )
    print(
        "最差帧：",
        worst["request_id"],
    )
    print(
        "最差 MPJPE：",
        f"{worst['mpjpe_px']:.4f} px",
    )
    print("MP4：", mp4_path)
    print("GIF：", gif_path)
    print("指标：", metrics_path)
    print("记录：", result_path)


if __name__ == "__main__":
    main()