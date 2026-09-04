"""结构化火柴人插帧服务后端。"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from stickman.interpolator import (
    StickmanInterpolator,
)
from stickman.request import (
    parse_request,
)


@dataclass(frozen=True)
class StickmanResult:
    """一次结构化插帧的内存结果。"""

    schema_version: int
    request_id: str
    stride: int
    context: int
    pose: np.ndarray
    png: bytes
    checkpoint_epoch: int
    device: str
    size: int

    def metadata(self):
        """返回可以直接转成 JSON 的元数据。"""
        return {
            "schema_version": (
                self.schema_version
            ),
            "request_id": self.request_id,
            "stride": self.stride,
            "context": self.context,
            "pose": self.pose.tolist(),
            "pose_shape": list(
                self.pose.shape
            ),
            "pose_is_finite": bool(
                np.isfinite(
                    self.pose
                ).all()
            ),
            "png_bytes": len(self.png),
            "checkpoint_epoch": (
                self.checkpoint_epoch
            ),
            "device": self.device,
            "size": self.size,
        }


class StickmanBackend:
    """在内存中执行结构化火柴人插帧。"""

    def __init__(
        self,
        checkpoint,
        device=None,
        size=256,
    ):
        self.checkpoint = Path(
            checkpoint
        )
        self.size = int(size)

        self.interpolator = (
            StickmanInterpolator(
                checkpoint=self.checkpoint,
                device=device,
                size=self.size,
            )
        )

    @property
    def epoch(self):
        return self.interpolator.epoch

    @property
    def context(self):
        return self.interpolator.context

    @property
    def strides(self):
        return self.interpolator.strides

    @property
    def device(self):
        return self.interpolator.device

    def interpolate_midpoint(
        self,
        request: Mapping[str, Any],
    ):
        """预测一个请求的中心帧。"""
        sample = parse_request(
            request
        )

        if (
            sample["context"]
            != self.context
        ):
            raise ValueError(
                "请求与模型的 context "
                "不一致："
                f"请求={sample['context']}，"
                f"模型={self.context}"
            )

        if (
            sample["stride"]
            not in self.strides
        ):
            raise ValueError(
                "模型不支持 stride="
                f"{sample['stride']}；"
                f"支持 {self.strides}"
            )

        pose, image = (
            self.interpolator
            .render_midpoint(
                sample["left_poses"],
                sample["right_poses"],
                sample["left_offsets"],
                sample["right_offsets"],
            )
        )

        pose = np.asarray(
            pose,
            dtype=np.float32,
        ).copy()

        if pose.shape != (17, 2):
            raise RuntimeError(
                "后端预测姿态形状错误："
                f"{pose.shape}"
            )

        if not np.isfinite(
            pose
        ).all():
            raise RuntimeError(
                "后端预测姿态存在非有限值"
            )

        buffer = BytesIO()
        image.save(
            buffer,
            format="PNG",
        )
        png = buffer.getvalue()

        if not png.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise RuntimeError(
                "后端没有生成有效 PNG"
            )

        return StickmanResult(
            schema_version=(
                sample["schema_version"]
            ),
            request_id=(
                sample["request_id"]
            ),
            stride=sample["stride"],
            context=sample["context"],
            pose=pose,
            png=png,
            checkpoint_epoch=self.epoch,
            device=str(self.device),
            size=self.size,
        )

    def interpolate_many(
        self,
        requests: Iterable[
            Mapping[str, Any]
        ],
    ):
        """复用同一个模型处理多个请求。"""
        if isinstance(
            requests,
            Mapping,
        ):
            raise TypeError(
                "requests 必须是请求序列，"
                "不能是单个字典"
            )

        results = [
            self.interpolate_midpoint(
                request
            )
            for request in requests
        ]

        if not results:
            raise ValueError(
                "requests 不能为空"
            )

        request_ids = [
            result.request_id
            for result in results
        ]

        if len(set(request_ids)) != len(
            request_ids
        ):
            raise ValueError(
                "批量请求中存在重复 "
                "request_id"
            )

        return results