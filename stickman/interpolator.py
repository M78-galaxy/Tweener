"""火柴人插帧统一高层接口。"""

import numpy as np

from stickman.models import (
    SequencePredictor,
)
from stickman.skeleton import (
    CANVAS,
    render,
)


class StickmanInterpolator:
    """封装正式多帧 Transformer。"""

    def __init__(
        self,
        checkpoint,
        device=None,
        size=CANVAS,
    ):
        self.predictor = (
            SequencePredictor(
                checkpoint,
                device=device,
            )
        )

        self.size = int(size)

        if self.size <= 0:
            raise ValueError(
                "size 必须大于 0"
            )

    @property
    def epoch(self):
        return self.predictor.epoch

    @property
    def context(self):
        return self.predictor.context

    @property
    def strides(self):
        return self.predictor.strides

    @property
    def device(self):
        return self.predictor.device

    def predict_midpoint(
        self,
        left_poses,
        right_poses,
        left_offsets,
        right_offsets,
    ):
        """预测目标时刻的 COCO-17 姿态。"""
        pose = self.predictor.predict(
            left_poses,
            right_poses,
            left_offsets,
            right_offsets,
        )

        pose = np.asarray(
            pose,
            dtype=np.float32,
        )

        if pose.shape != (17, 2):
            raise RuntimeError(
                "预测姿态形状错误："
                f"{pose.shape}"
            )

        if not np.isfinite(pose).all():
            raise RuntimeError(
                "预测姿态存在 NaN 或无穷值"
            )

        return pose

    def render_midpoint(
        self,
        left_poses,
        right_poses,
        left_offsets,
        right_offsets,
    ):
        """返回预测姿态和渲染后的 PIL 图像。"""
        pose = self.predict_midpoint(
            left_poses,
            right_poses,
            left_offsets,
            right_offsets,
        )

        image = render(
            pose,
            size=self.size,
        )

        return pose, image