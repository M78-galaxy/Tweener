"""从 checkpoint 加载姿态插帧 MLP。"""

from pathlib import Path

import numpy as np
import torch

from stickman.dataset import normalize_endpoints
from stickman.models.mlp import ResidualMLP


class MLPPredictor:
    """提供与传统基线相似的单样本预测接口。"""

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

        self.stride = int(config["stride"])
        self.bone_weight = float(
            config["bone_weight"]
        )

        self.model = ResidualMLP(
            hidden_dim=int(
                config["hidden_dim"]
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
        prev_pose,
        next_pose,
    ):
        prev = torch.as_tensor(
            np.asarray(prev_pose),
            dtype=torch.float32,
            device=self.device,
        )
        next_tensor = torch.as_tensor(
            np.asarray(next_pose),
            dtype=torch.float32,
            device=self.device,
        )

        if prev.shape != (17, 2):
            raise ValueError(
                "prev_pose 必须是 (17, 2)，"
                f"实际是 {tuple(prev.shape)}"
            )

        if next_tensor.shape != (17, 2):
            raise ValueError(
                "next_pose 必须是 (17, 2)，"
                f"实际是 {tuple(next_tensor.shape)}"
            )

        if not torch.isfinite(prev).all():
            raise ValueError(
                "prev_pose 存在 NaN 或无穷值"
            )

        if not torch.isfinite(
            next_tensor
        ).all():
            raise ValueError(
                "next_pose 存在 NaN 或无穷值"
            )

        (
            prev_normalized,
            next_normalized,
            center,
            scale,
        ) = normalize_endpoints(
            prev,
            next_tensor,
        )

        model_input = torch.cat(
            (
                prev_normalized.reshape(-1),
                next_normalized.reshape(-1),
            )
        ).unsqueeze(0)

        lerp_normalized = (
            prev_normalized
            + next_normalized
        ) * 0.5

        prediction_normalized = (
            self.model.predict_pose(
                model_input,
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