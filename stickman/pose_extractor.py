"""可复用的单张图片姿态提取接口。"""

from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image


# COCO-17 对应的 MediaPipe-33 关键点索引。
MP_TO_COCO17 = (
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
)


def mediapipe_to_coco17(
    landmarks,
    width,
    height,
):
    """把 MediaPipe-33 映射到正方形画布中的 COCO-17。"""
    if width <= 0 or height <= 0:
        raise ValueError(
            "图像宽高必须大于 0"
        )

    side = float(
        max(width, height)
    )
    pad_x = (
        side - width
    ) / 2.0
    pad_y = (
        side - height
    ) / 2.0

    pose = []
    confidence = []

    for index in MP_TO_COCO17:
        landmark = landmarks[index]

        x = (
            landmark.x * width
            + pad_x
        ) / side
        y = (
            landmark.y * height
            + pad_y
        ) / side

        pose.append(
            (x, y)
        )
        confidence.append(
            min(
                float(
                    landmark.visibility
                ),
                float(
                    landmark.presence
                ),
            )
        )

    pose = np.asarray(
        pose,
        dtype=np.float32,
    )
    confidence = np.asarray(
        confidence,
        dtype=np.float32,
    )

    if pose.shape != (17, 2):
        raise RuntimeError(
            "COCO-17 姿态形状错误："
            f"{pose.shape}"
        )

    if confidence.shape != (17,):
        raise RuntimeError(
            "COCO-17 置信度形状错误："
            f"{confidence.shape}"
        )

    if not np.isfinite(pose).all():
        raise RuntimeError(
            "COCO-17 姿态存在非有限值"
        )

    if not np.isfinite(
        confidence
    ).all():
        raise RuntimeError(
            "COCO-17 置信度存在非有限值"
        )

    return pose, confidence


class PoseExtractor:
    """使用 MediaPipe 从单张 RGB 图片提取 COCO-17。"""

    def __init__(
        self,
        model_path,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
    ):
        model_path = Path(
            model_path
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                "MediaPipe 模型不存在："
                f"{model_path}"
            )

        for name, value in (
            (
                "min_detection_confidence",
                min_detection_confidence,
            ),
            (
                "min_presence_confidence",
                min_presence_confidence,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} 必须位于 [0, 1]"
                )

        options = (
            mp.tasks.vision
            .PoseLandmarkerOptions(
                base_options=(
                    mp.tasks.BaseOptions(
                        model_asset_path=str(
                            model_path
                        ),
                    )
                ),
                running_mode=(
                    mp.tasks.vision
                    .RunningMode.IMAGE
                ),
                num_poses=1,
                min_pose_detection_confidence=(
                    min_detection_confidence
                ),
                min_pose_presence_confidence=(
                    min_presence_confidence
                ),
                output_segmentation_masks=False,
            )
        )

        self.model_path = model_path
        self.landmarker = (
            mp.tasks.vision
            .PoseLandmarker
            .create_from_options(
                options
            )
        )

    def close(self):
        """释放 MediaPipe 资源。"""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    def detect_rgb(
        self,
        image_rgb,
    ):
        """检测 RGB uint8 图像；失败时返回 None。"""
        if self.landmarker is None:
            raise RuntimeError(
                "PoseExtractor 已关闭"
            )

        image_rgb = np.asarray(
            image_rgb
        )

        if (
            image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
        ):
            raise ValueError(
                "RGB 图像形状必须是 "
                f"(H, W, 3)，实际为 "
                f"{image_rgb.shape}"
            )

        if image_rgb.dtype != np.uint8:
            raise ValueError(
                "RGB 图像 dtype 必须是 "
                f"uint8，实际为 "
                f"{image_rgb.dtype}"
            )

        image_rgb = np.ascontiguousarray(
            image_rgb
        )

        height, width = (
            image_rgb.shape[:2]
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=image_rgb,
        )

        result = self.landmarker.detect(
            mp_image
        )

        if not result.pose_landmarks:
            return None

        landmarks = (
            result.pose_landmarks[0]
        )

        if len(landmarks) != 33:
            raise RuntimeError(
                "MediaPipe 关键点数量错误："
                f"{len(landmarks)}"
            )

        return mediapipe_to_coco17(
            landmarks,
            width,
            height,
        )

    def detect_pil(
        self,
        image,
    ):
        """检测 PIL 图片；失败时返回 None。"""
        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image 必须是 PIL.Image"
            )

        image_rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

        return self.detect_rgb(
            image_rgb
        )

    def detect_path(
        self,
        image_path,
    ):
        """读取并检测一张图片。"""
        image_path = Path(
            image_path
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"图片不存在：{image_path}"
            )

        with Image.open(
            image_path
        ) as image:
            return self.detect_pil(
                image
            )