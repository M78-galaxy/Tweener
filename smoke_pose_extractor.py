"""对比 MediaPipe 对真人帧和火柴人图的识别能力。"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from stickman.pose_extractor import (
    PoseExtractor,
)
from stickman.skeleton import (
    mpjpe,
    render,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=Path,
        default=Path(
            "data/videos/walk_val_001.mp4"
        ),
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--reference-poses",
        type=Path,
        default=Path(
            "data/poses/walk_val_001.npz"
        ),
    )
    parser.add_argument(
        "--stickman",
        type=Path,
        default=Path(
            "runs/v5_sequence_smoke/output/"
            "walk_s4_t0008/prediction.png"
        ),
    )
    parser.add_argument(
        "--stickman-pose",
        type=Path,
        default=Path(
            "runs/v5_sequence_smoke/output/"
            "walk_s4_t0008/"
            "prediction_pose.npy"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/"
            "pose_landmarker_full.task"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/"
            "v5_pose_extractor_smoke"
        ),
    )

    return parser.parse_args()


def read_video_frame(
    video_path,
    frame_index,
):
    if not video_path.is_file():
        raise FileNotFoundError(
            f"视频不存在：{video_path}"
        )

    if frame_index < 0:
        raise ValueError(
            "frame 必须大于或等于 0"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"无法打开视频：{video_path}"
        )

    try:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame_bgr = (
            capture.read()
        )
    finally:
        capture.release()

    if not ok or frame_bgr is None:
        raise RuntimeError(
            f"无法读取第 {frame_index} 帧"
        )

    return frame_bgr


def report_detection(
    name,
    detection,
    reference_pose,
    preview_path,
):
    if detection is None:
        print(f"{name}检测成功：False")
        return False

    pose, confidence = detection

    error = mpjpe(
        pose,
        reference_pose,
        size=256,
    )

    render(
        pose,
        size=256,
    ).save(
        preview_path
    )

    print(f"{name}检测成功：True")
    print(
        f"{name}姿态形状：",
        pose.shape,
    )
    print(
        f"{name}置信度最小值：",
        f"{confidence.min():.4f}",
    )
    print(
        f"{name}置信度平均值：",
        f"{confidence.mean():.4f}",
    )
    print(
        f"{name}相对参考 MPJPE：",
        f"{error:.4f} px",
    )
    print(
        f"{name}检测预览：",
        preview_path,
    )

    return True


def main():
    args = parse_args()

    if not args.reference_poses.is_file():
        raise FileNotFoundError(
            "参考姿态不存在："
            f"{args.reference_poses}"
        )

    if not args.stickman.is_file():
        raise FileNotFoundError(
            "火柴人图片不存在："
            f"{args.stickman}"
        )

    if not args.stickman_pose.is_file():
        raise FileNotFoundError(
            "火柴人参考姿态不存在："
            f"{args.stickman_pose}"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_bgr = read_video_frame(
        args.video,
        args.frame,
    )
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    with np.load(
        args.reference_poses
    ) as data:
        real_reference = np.asarray(
            data["poses"][args.frame],
            dtype=np.float32,
        )

    stickman_reference = np.asarray(
        np.load(args.stickman_pose),
        dtype=np.float32,
    )

    real_frame_path = (
        args.out / "real_frame.jpg"
    )

    if not cv2.imwrite(
        str(real_frame_path),
        frame_bgr,
    ):
        raise RuntimeError(
            "无法保存真人测试帧"
        )

    with PoseExtractor(
        args.model
    ) as extractor:
        real_detection = (
            extractor.detect_rgb(
                frame_rgb
            )
        )
        stickman_detection = (
            extractor.detect_path(
                args.stickman
            )
        )

    print("测试帧：", args.frame)
    print("真人原图：", real_frame_path)

    real_ok = report_detection(
        "真人",
        real_detection,
        real_reference,
        args.out / "real_detected.png",
    )

    stickman_ok = report_detection(
        "火柴人",
        stickman_detection,
        stickman_reference,
        args.out / "stickman_detected.png",
    )

    print()
    print("结论：")

    if real_ok and stickman_ok:
        print(
            "MediaPipe 同时识别了真人和火柴人。"
        )
    elif real_ok:
        print(
            "MediaPipe 能识别真人，"
            "但不能直接识别当前火柴人线稿。"
        )
    else:
        print(
            "单图模式连真人帧也未稳定识别，"
            "需要先检查单图提取配置。"
        )


if __name__ == "__main__":
    main()