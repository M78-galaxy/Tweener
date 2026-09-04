"""从 checkpoint 加载多帧姿态 Transformer。"""

from pathlib import Path

import numpy as np
import torch

from stickman.dataset import normalize_endpoints
from stickman.models.seq import SequenceTransformer
from stickman.sequence_dataset import TIME_SCALE


class SequencePredictor:
    """多帧上下文姿态插帧推理器。"""

    def __init__(
        self,
        checkpoint,
        device=None,
    ):
        checkpoint = Path(checkpoint)

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"模型文件不存在：{checkpoint}"
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        saved = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        config = saved["config"]

        self.context = int(
            config["context"]
        )
        self.strides = tuple(
            int(value)
            for value in config["strides"]
        )

        self.model = SequenceTransformer(
            d_model=int(
                config["d_model"]
            ),
            nhead=int(
                config["nhead"]
            ),
            num_layers=int(
                config["num_layers"]
            ),
            dim_feedforward=int(
                config["dim_feedforward"]
            ),
            dropout=float(
                config["dropout"]
            ),
        ).to(self.device)

        self.model.load_state_dict(
            saved["model_state"]
        )
        self.model.eval()

        self.epoch = int(saved["epoch"])
        self.metrics = saved["metrics"]
        self.checkpoint = checkpoint

    @torch.no_grad()
    def predict(
        self,
        left_poses,
        right_poses,
        left_offsets,
        right_offsets,
    ):
        left = torch.as_tensor(
            np.asarray(left_poses),
            dtype=torch.float32,
            device=self.device,
        )
        right = torch.as_tensor(
            np.asarray(right_poses),
            dtype=torch.float32,
            device=self.device,
        )

        left_time = torch.as_tensor(
            np.asarray(left_offsets),
            dtype=torch.float32,
            device=self.device,
        )
        right_time = torch.as_tensor(
            np.asarray(right_offsets),
            dtype=torch.float32,
            device=self.device,
        )

        expected_pose_shape = (
            self.context,
            17,
            2,
        )
        expected_time_shape = (
            self.context,
        )

        if tuple(left.shape) != expected_pose_shape:
            raise ValueError(
                "left_poses 形状错误："
                f"期望 {expected_pose_shape}，"
                f"实际 {tuple(left.shape)}"
            )

        if tuple(right.shape) != expected_pose_shape:
            raise ValueError(
                "right_poses 形状错误："
                f"期望 {expected_pose_shape}，"
                f"实际 {tuple(right.shape)}"
            )

        if tuple(left_time.shape) != expected_time_shape:
            raise ValueError(
                "left_offsets 形状错误："
                f"期望 {expected_time_shape}，"
                f"实际 {tuple(left_time.shape)}"
            )

        if tuple(right_time.shape) != expected_time_shape:
            raise ValueError(
                "right_offsets 形状错误："
                f"期望 {expected_time_shape}，"
                f"实际 {tuple(right_time.shape)}"
            )

        if not torch.isfinite(left).all():
            raise ValueError(
                "left_poses 存在 NaN 或无穷值"
            )

        if not torch.isfinite(right).all():
            raise ValueError(
                "right_poses 存在 NaN 或无穷值"
            )

        if not torch.all(left_time < 0):
            raise ValueError(
                "左侧时间偏移必须全部小于 0"
            )

        if not torch.all(right_time > 0):
            raise ValueError(
                "右侧时间偏移必须全部大于 0"
            )

        left_stride = int(
            round(-left_time[-1].item())
        )
        right_stride = int(
            round(right_time[0].item())
        )

        if left_stride != right_stride:
            raise ValueError(
                "左右端点到目标帧的距离不一致："
                f"{left_stride} 和 {right_stride}"
            )

        if left_stride not in self.strides:
            raise ValueError(
                f"模型不支持 stride={left_stride}；"
                f"支持 {self.strides}"
            )

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

        right_normalized = (
            right - center
        ) / scale

        tokens = torch.cat(
            (
                left_normalized,
                right_normalized,
            ),
            dim=0,
        ).reshape(
            1,
            self.context * 2,
            34,
        )

        time_features = torch.cat(
            (
                left_time,
                right_time,
            )
        )

        time_features = (
            time_features / TIME_SCALE
        ).reshape(
            1,
            self.context * 2,
            1,
        )

        lerp_normalized = (
            left_normalized[-1]
            + right_normalized[0]
        ) * 0.5

        prediction_normalized = (
            self.model.predict_pose(
                tokens,
                time_features,
                lerp_normalized.unsqueeze(0),
            )[0]
        )

        prediction = (
            prediction_normalized * scale
            + center
        )

        return (
            prediction
            .cpu()
            .numpy()
            .astype(np.float32)
        )