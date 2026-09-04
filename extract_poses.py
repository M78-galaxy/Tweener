#!/usr/bin/env python3
"""从视频提取火柴人姿态。"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

from stickman.skeleton import render

POSE_CONNECTIONS = [
    # 脸
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),

    # 上半身
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),

    # 左手
    (15, 17), (15, 19), (15, 21), (17, 19),

    # 右手
    (16, 18), (16, 20), (16, 22), (18, 20),

    # 躯干
    (11, 23), (12, 24), (23, 24),

    # 左腿和脚
    (23, 25), (25, 27),
    (27, 29), (29, 31), (27, 31),

    # 右腿和脚
    (24, 26), (26, 28),
    (28, 30), (30, 32), (28, 32),
]

LEFT_LANDMARKS = {
    1, 2, 3, 7, 9,
    11, 13, 15, 17, 19, 21,
    23, 25, 27, 29, 31,
}

RIGHT_LANDMARKS = {
    4, 5, 6, 8, 10,
    12, 14, 16, 18, 20, 22,
    24, 26, 28, 30, 32,
}

# COCO-17 顺序对应的 MediaPipe 关键点索引。
MP_TO_COCO17 = [
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]

def draw_pose(
    frame_bgr,
    landmarks,
    frame_index: int,
    visibility_threshold: float = 0.5,
):
    """把 MediaPipe 的 33 个关键点画到原视频帧上。"""
    output = frame_bgr.copy()
    height, width = output.shape[:2]

    def visible(landmark) -> bool:
        return landmark.visibility >= visibility_threshold

    def point(landmark):
        return (
            round(landmark.x * width),
            round(landmark.y * height),
        )

    # 先画骨骼连线。
    for start, end in POSE_CONNECTIONS:
        a = landmarks[start]
        b = landmarks[end]

        if visible(a) and visible(b):
            cv2.line(
                output,
                point(a),
                point(b),
                color=(0, 255, 255),
                thickness=3,
                lineType=cv2.LINE_AA,
            )

    # 再画关键点。OpenCV 使用 BGR 颜色顺序。
    for index, landmark in enumerate(landmarks):
        if not visible(landmark):
            continue

        if index in LEFT_LANDMARKS:
            color = (255, 80, 40)       # 蓝色：人体左侧
        elif index in RIGHT_LANDMARKS:
            color = (40, 80, 255)       # 红色：人体右侧
        else:
            color = (40, 220, 40)       # 绿色：中间点

        cv2.circle(
            output,
            point(landmark),
            radius=5,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"frame {frame_index}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"frame {frame_index}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output

def mediapipe_to_coco17(
    landmarks,
    width: int,
    height: int,
):
    """把 MediaPipe-33 映射为 COCO-17。"""
    side = float(max(width, height))

    # 把长方形视频居中放入正方形，避免人体比例变形。
    pad_x = (side - width) / 2
    pad_y = (side - height) / 2

    pose = []
    confidence = []

    for mp_index in MP_TO_COCO17:
        landmark = landmarks[mp_index]

        # MediaPipe 的坐标原本是相对于视频宽高归一化的。
        # 先恢复成像素坐标，再映射到正方形画布。
        x = (landmark.x * width + pad_x) / side
        y = (landmark.y * height + pad_y) / side

        pose.append([x, y])

        visibility = float(landmark.visibility)
        presence = float(landmark.presence)

        # 两项中较低的一项作为该关节的置信度。
        confidence.append(min(visibility, presence))

    pose = np.asarray(pose, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)

    if pose.shape != (17, 2):
        raise RuntimeError(
            f"COCO-17 姿态形状错误：{pose.shape}"
        )

    return pose, confidence


def save_coco_preview(
    preview_poses,
    output_path: Path,
):
    """将四个 COCO-17 姿态画成 2×2 检查图。"""
    cell_size = 256

    sheet = Image.new(
        "L",
        (cell_size * 2, cell_size * 2),
        255,
    )

    preview_poses.sort(key=lambda item: item[0])

    for slot, item in enumerate(preview_poses):
        frame_index, pose, confidence = item

        cell = render(
            pose,
            size=cell_size,
        )

        draw = ImageDraw.Draw(cell)

        # 在每个格子上方写帧号和最低置信度。
        draw.rectangle(
            [0, 0, cell_size, 18],
            fill=255,
        )
        draw.text(
            (4, 3),
            (
                f"frame {frame_index}, "
                f"min conf {confidence.min():.2f}"
            ),
            fill=0,
        )

        column = slot % 2
        row = slot // 2

        sheet.paste(
            cell,
            (
                column * cell_size,
                row * cell_size,
            ),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(output_path)

def resize_with_letterbox(
    image,
    target_width: int = 480,
    target_height: int = 480,
):
    """等比例缩放，并用白色填充成固定尺寸。"""
    source_height, source_width = image.shape[:2]

    scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = max(
        1,
        round(source_width * scale),
    )
    resized_height = max(
        1,
        round(source_height * scale),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (target_height, target_width, 3),
        255,
        dtype=np.uint8,
    )

    offset_x = (
        target_width - resized_width
    ) // 2
    offset_y = (
        target_height - resized_height
    ) // 2

    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized

    return canvas

def detect_video(video_path: Path, model_path: Path) -> None:
    """逐帧检测人体姿态，并统计检测成功率。"""
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在：{model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"视频帧率无效：{fps}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    read_frames = 0
    detected_frames = 0
    missed_frames = []

    preview_indices = {
        0,
        reported_frames // 3,
        reported_frames * 2 // 3,
        reported_frames - 1,
    }
    preview_frames = []
    preview_coco_poses = []

    # 每个列表最终都应该包含read_frames个元素
    all_poses = []
    all_confidences = []
    all_detected = []

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if frame_bgr is None:
                    raise RuntimeError(f"第 {read_frames} 帧为空")

                # OpenCV 读取的是 BGR；MediaPipe 需要 RGB。
                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # VIDEO 模式要求时间戳单调递增，单位是毫秒。
                timestamp_ms = round(read_frames * 1000 / fps)

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]

                    if len(landmarks) != 33:
                        raise RuntimeError(
                            f"第 {read_frames} 帧得到 "
                            f"{len(landmarks)} 个关键点，预期 33 个"
                        )

                    detected_frames += 1

                    # 每一个检测成功帧都转换成 COCO-17。
                    coco_pose, coco_confidence = (
                        mediapipe_to_coco17(
                            landmarks,
                            width,
                            height,
                        )
                    )

                    all_poses.append(coco_pose)
                    all_confidences.append(coco_confidence)
                    all_detected.append(True)

                    # 只有指定的四帧需要生成预览图。
                    if read_frames in preview_indices:
                        annotated = draw_pose(
                            frame_bgr,
                            landmarks,
                            read_frames,
                        )

                        annotated = resize_with_letterbox(
                            annotated
                        )

                        preview_frames.append(
                            (read_frames, annotated)
                        )

                        preview_coco_poses.append(
                            (
                                read_frames,
                                coco_pose,
                                coco_confidence,
                            )
                        )

                else:
                    missed_frames.append(read_frames)

                    # 检测失败帧也保留位置，使用 NaN 表示坐标缺失。
                    all_poses.append(
                        np.full(
                            (17, 2),
                            np.nan,
                            dtype=np.float32,
                        )
                    )
                    all_confidences.append(
                        np.zeros(
                            17,
                            dtype=np.float32,
                        )
                    )
                    all_detected.append(False)

                read_frames += 1
    finally:
        cap.release()

    if read_frames == 0:
        raise RuntimeError("没有成功读取任何视频帧")

    detection_rate = detected_frames / read_frames * 100

    print(f"视频：{video_path}")
    print(f"尺寸：{width} x {height}")
    print(f"帧率：{fps:.2f} FPS")
    print(f"元数据帧数：{reported_frames}")
    print(f"实际读取帧数：{read_frames}")
    print(f"检测成功帧数：{detected_frames}")
    print(f"检测失败帧数：{len(missed_frames)}")
    print(f"检测成功率：{detection_rate:.2f}%")

    if missed_frames:
        print(f"前 20 个失败帧：{missed_frames[:20]}")
        
    poses_array = np.asarray(
        all_poses,
        dtype=np.float32,
    )
    confidence_array = np.asarray(
        all_confidences,
        dtype=np.float32,
    )
    detected_array = np.asarray(
        all_detected,
        dtype=np.bool_,
    )

    if poses_array.shape != (read_frames, 17, 2):
        raise RuntimeError(
            f"姿态数组形状错误：{poses_array.shape}"
        )

    if confidence_array.shape != (read_frames, 17):
        raise RuntimeError(
            f"置信度数组形状错误："
            f"{confidence_array.shape}"
        )

    pose_output = (
        Path("data/poses")
        / f"{video_path.stem}.npz"
    )
    pose_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        pose_output,
        poses=poses_array,
        confidence=confidence_array,
        detected=detected_array,
        fps=np.float32(fps),
        width=np.int32(width),
        height=np.int32(height),
        video=str(video_path),
    )

    print(f"姿态序列：{pose_output}")
    if len(preview_frames) == 4:
        preview_frames.sort(key=lambda item: item[0])
        images = [image for _, image in preview_frames]    
        top = np.hstack(images[:2])
        bottom = np.hstack(images[2:])
        grid = np.vstack([top, bottom])    
        preview_path = (
            Path("runs/previews")
            / f"{video_path.stem}_pose.jpg"
        )
        preview_path.parent.mkdir(parents=True, exist_ok=True)    
        if not cv2.imwrite(str(preview_path), grid):
            raise RuntimeError(f"无法保存预览图：{preview_path}")    
        print(f"预览图：{preview_path}")
    else:
        print(
            f"警告：只收集到 {len(preview_frames)} 张预览帧，"
            "没有生成 2×2 预览图"
        )    

    if len(preview_coco_poses) == 4:
        coco_preview_path = (
            Path("runs/previews")
            / f"{video_path.stem}_coco17.png"
        )
        
        save_coco_preview(
            preview_coco_poses,
            coco_preview_path,
        )

        print(
            f"COCO-17 预览图：{coco_preview_path}"
        )
    else:
        print(
            f"警告：只收集到 "
            f"{len(preview_coco_poses)} "
            "个 COCO-17 预览姿态"
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video",
        type=Path,
        help="输入视频路径",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_full.task"),
        help="MediaPipe Pose Landmarker 模型路径",
    )
    args = parser.parse_args()

    detect_video(args.video, args.model)


if __name__ == "__main__":
    main()