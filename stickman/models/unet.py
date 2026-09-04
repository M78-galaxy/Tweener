"""火柴人栅格插帧小型 UNet。"""

import torch
from torch import nn
from torch.nn import functional as F


def group_count(channels):
    """选择能整除通道数的 GroupNorm 组数。"""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups

    return 1


class ConvBlock(nn.Module):
    """两层卷积、GroupNorm 和 SiLU。"""

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        groups = group_count(
            out_channels
        )

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                out_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                out_channels,
            ),
            nn.SiLU(),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class RasterUNet(nn.Module):
    """使用六帧图像与时间图预测目标火柴人。"""

    def __init__(
        self,
        in_channels=12,
        base_channels=16,
        dropout=0.1,
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels 必须大于 0"
            )

        if base_channels <= 0:
            raise ValueError(
                "base_channels 必须大于 0"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout 必须位于 [0, 1)"
            )

        self.in_channels = in_channels
        self.base_channels = base_channels

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.encoder1 = ConvBlock(
            in_channels,
            c1,
        )
        self.encoder2 = ConvBlock(
            c1,
            c2,
        )
        self.encoder3 = ConvBlock(
            c2,
            c3,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.bottleneck = nn.Sequential(
            ConvBlock(
                c3,
                c4,
            ),
            nn.Dropout2d(dropout),
        )

        self.decoder3 = ConvBlock(
            c4 + c3,
            c3,
        )
        self.decoder2 = ConvBlock(
            c3 + c2,
            c2,
        )
        self.decoder1 = ConvBlock(
            c2 + c1,
            c1,
        )

        self.output_layer = nn.Conv2d(
            c1,
            1,
            kernel_size=1,
        )

        # 初始残差为 0，
        # 初始预测近似等于 pixel lerp。
        nn.init.zeros_(
            self.output_layer.weight
        )
        nn.init.zeros_(
            self.output_layer.bias
        )

    def upsample(
        self,
        inputs,
        reference,
    ):
        return F.interpolate(
            inputs,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, model_input):
        """返回相对于 pixel lerp 的 logit 修正量。"""
        if model_input.ndim != 4:
            raise ValueError(
                "model_input 必须是四维张量"
            )

        if (
            model_input.shape[1]
            != self.in_channels
        ):
            raise ValueError(
                "输入通道错误："
                f"期望 {self.in_channels}，"
                f"实际 {model_input.shape[1]}"
            )

        height, width = (
            model_input.shape[-2:]
        )

        if height % 8 != 0:
            raise ValueError(
                "图像高度必须能被 8 整除"
            )

        if width % 8 != 0:
            raise ValueError(
                "图像宽度必须能被 8 整除"
            )

        encoder1 = self.encoder1(
            model_input
        )

        encoder2 = self.encoder2(
            self.pool(encoder1)
        )

        encoder3 = self.encoder3(
            self.pool(encoder2)
        )

        bottleneck = self.bottleneck(
            self.pool(encoder3)
        )

        decoder3 = self.upsample(
            bottleneck,
            encoder3,
        )
        decoder3 = torch.cat(
            (
                decoder3,
                encoder3,
            ),
            dim=1,
        )
        decoder3 = self.decoder3(
            decoder3
        )

        decoder2 = self.upsample(
            decoder3,
            encoder2,
        )
        decoder2 = torch.cat(
            (
                decoder2,
                encoder2,
            ),
            dim=1,
        )
        decoder2 = self.decoder2(
            decoder2
        )

        decoder1 = self.upsample(
            decoder2,
            encoder1,
        )
        decoder1 = torch.cat(
            (
                decoder1,
                encoder1,
            ),
            dim=1,
        )
        decoder1 = self.decoder1(
            decoder1
        )

        return self.output_layer(
            decoder1
        )

    def predict_logits(
        self,
        model_input,
        pixel_lerp,
        epsilon=1e-4,
    ):
        """返回加入 pixel lerp 先验后的最终 logits。"""
        expected_shape = (
            model_input.shape[0],
            1,
            model_input.shape[2],
            model_input.shape[3],
        )

        if (
            tuple(pixel_lerp.shape)
            != expected_shape
        ):
            raise ValueError(
                "pixel_lerp 形状错误："
                f"期望 {expected_shape}，"
                f"实际 {tuple(pixel_lerp.shape)}"
            )

        if not torch.isfinite(
            pixel_lerp
        ).all():
            raise ValueError(
                "pixel_lerp 存在非有限值"
            )

        base_logits = torch.logit(
            pixel_lerp.clamp(
                epsilon,
                1.0 - epsilon,
            )
        )

        residual_logits = self(
            model_input
        )

        return (
            base_logits
            + residual_logits
        )

    def predict_image(
        self,
        model_input,
        pixel_lerp,
    ):
        """返回范围为 [0, 1] 的前景概率图。"""
        logits = self.predict_logits(
            model_input,
            pixel_lerp,
        )

        return torch.sigmoid(logits)