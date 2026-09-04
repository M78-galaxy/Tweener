"""带连续时间嵌入的姿态序列 Transformer。"""

import torch
from torch import nn


POSE_DIM = 34
OUTPUT_DIM = 34


class SequenceTransformer(nn.Module):
    """使用左右多帧上下文预测中间姿态残差。"""

    def __init__(
        self,
        d_model=96,
        nhead=4,
        num_layers=2,
        dim_feedforward=192,
        dropout=0.1,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                "d_model 必须能被 nhead 整除"
            )

        self.pose_embedding = nn.Linear(
            POSE_DIM,
            d_model,
        )

        # 连续的 time-to-arrival embedding。
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 时间为 0 的目标查询 token。
        self.query_token = nn.Parameter(
            torch.empty(1, 1, d_model)
        )
        nn.init.normal_(
            self.query_token,
            mean=0.0,
            std=0.02,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=(
                    dim_feedforward
                ),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.output_norm = nn.LayerNorm(
            d_model
        )
        self.output_layer = nn.Linear(
            d_model,
            OUTPUT_DIM,
        )

        # 初始输出严格等于 lerp。
        nn.init.zeros_(
            self.output_layer.weight
        )
        nn.init.zeros_(
            self.output_layer.bias
        )

    def forward(
        self,
        tokens,
        time_features,
    ):
        """返回形状为 (batch, 34) 的残差。"""
        if tokens.ndim != 3:
            raise ValueError(
                "tokens 必须是三维张量"
            )

        if tokens.shape[-1] != POSE_DIM:
            raise ValueError(
                "tokens 最后一维必须是 "
                f"{POSE_DIM}，实际是 "
                f"{tokens.shape[-1]}"
            )

        expected_time_shape = (
            tokens.shape[0],
            tokens.shape[1],
            1,
        )

        if (
            tuple(time_features.shape)
            != expected_time_shape
        ):
            raise ValueError(
                "time_features 形状错误："
                f"期望 {expected_time_shape}，"
                f"实际 {tuple(time_features.shape)}"
            )

        context_tokens = (
            self.pose_embedding(tokens)
            + self.time_embedding(
                time_features
            )
        )

        batch_size = tokens.shape[0]

        query_time = tokens.new_zeros(
            batch_size,
            1,
            1,
        )

        query = self.query_token.expand(
            batch_size,
            -1,
            -1,
        )

        query = (
            query
            + self.time_embedding(
                query_time
            )
        )

        sequence = torch.cat(
            (
                query,
                context_tokens,
            ),
            dim=1,
        )

        encoded = self.encoder(sequence)

        query_output = encoded[:, 0]

        query_output = self.output_norm(
            query_output
        )

        return self.output_layer(
            query_output
        )

    def predict_pose(
        self,
        tokens,
        time_features,
        lerp_pose,
    ):
        """返回归一化坐标中的最终姿态。"""
        residual = self(
            tokens,
            time_features,
        )

        residual = residual.reshape(
            tokens.shape[0],
            17,
            2,
        )

        return lerp_pose + residual