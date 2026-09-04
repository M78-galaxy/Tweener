"""用于姿态插帧的最小残差 MLP。"""

import torch
from torch import nn


INPUT_DIM = 68
OUTPUT_DIM = 34


class ResidualMLP(nn.Module):
    """根据前后姿态预测相对于 lerp 的残差。"""

    def __init__(
        self,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

        # 初始残差为零：
        # prediction = lerp + 0
        output_layer = self.network[-1]

        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, model_input):
        """返回展平后的姿态残差，形状为 (..., 34)。"""
        if model_input.shape[-1] != INPUT_DIM:
            raise ValueError(
                "模型输入最后一维必须是 "
                f"{INPUT_DIM}，实际是 "
                f"{model_input.shape[-1]}"
            )

        return self.network(model_input)

    def predict_pose(
        self,
        model_input,
        lerp_pose,
    ):
        """返回归一化坐标中的最终预测姿态。"""
        residual = self(model_input)

        residual = residual.reshape(
            *model_input.shape[:-1],
            17,
            2,
        )

        return lerp_pose + residual