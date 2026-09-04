"""火柴人稀疏栅格图训练损失。"""

import torch
from torch.nn import functional as F


def validate_raster_tensors(
    logits,
    target,
):
    if logits.shape != target.shape:
        raise ValueError(
            "logits 和 target 形状必须一致，"
            f"实际是 {tuple(logits.shape)} "
            f"和 {tuple(target.shape)}"
        )

    if logits.ndim != 4:
        raise ValueError(
            "logits 和 target 必须是 "
            "(batch, channel, height, width)"
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "当前只支持单通道图像"
        )

    if not torch.isfinite(logits).all():
        raise ValueError(
            "logits 存在 NaN 或无穷值"
        )

    if not torch.isfinite(target).all():
        raise ValueError(
            "target 存在 NaN 或无穷值"
        )

    if target.min().item() < 0.0:
        raise ValueError(
            "target 不能小于 0"
        )

    if target.max().item() > 1.0:
        raise ValueError(
            "target 不能大于 1"
        )


def spatial_sum(tensor):
    return tensor.flatten(1).sum(dim=1)


def weighted_bce_per_sample(
    logits,
    target,
    max_positive_weight=30.0,
    epsilon=1e-6,
):
    """对稀少的黑线前景提高权重。"""
    positive_count = spatial_sum(target)

    pixels_per_sample = (
        target[0].numel()
    )

    negative_count = (
        pixels_per_sample
        - positive_count
    )

    positive_weight = (
        negative_count
        / (positive_count + epsilon)
    ).clamp(
        min=1.0,
        max=max_positive_weight,
    )

    element_loss = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
    )

    pixel_weight = (
        1.0
        + target
        * (
            positive_weight[
                :,
                None,
                None,
                None,
            ]
            - 1.0
        )
    )

    weighted_loss = (
        element_loss
        * pixel_weight
    )

    return (
        weighted_loss
        .flatten(1)
        .mean(dim=1),
        positive_weight,
    )


def dice_loss_per_sample(
    probabilities,
    target,
    smooth=1.0,
):
    intersection = spatial_sum(
        probabilities * target
    )

    denominator = (
        spatial_sum(probabilities)
        + spatial_sum(target)
    )

    dice_score = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice_score


def dilate(
    image,
    radius,
):
    kernel_size = (
        radius * 2 + 1
    )

    return F.max_pool2d(
        image,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )


def line_distance_loss_per_sample(
    probabilities,
    target,
    radii=(1, 2, 4),
    epsilon=1e-6,
):
    """允许少量像素偏移的双向线条距离损失。

    prediction→target：
    惩罚落在目标线条邻域外的预测前景。

    target→prediction：
    惩罚预测邻域没有覆盖到的目标线条。
    """
    if not radii:
        raise ValueError(
            "radii 不能为空"
        )

    losses = []

    for radius in radii:
        if radius <= 0:
            raise ValueError(
                "距离半径必须大于 0"
            )

        target_region = dilate(
            target,
            radius,
        )

        prediction_region = dilate(
            probabilities,
            radius,
        )

        prediction_outside = (
            spatial_sum(
                probabilities
                * (1.0 - target_region)
            )
            / (
                spatial_sum(
                    probabilities
                )
                + epsilon
            )
        )

        target_missed = (
            spatial_sum(
                target
                * (
                    1.0
                    - prediction_region
                )
            )
            / (
                spatial_sum(target)
                + epsilon
            )
        )

        losses.append(
            (
                prediction_outside
                + target_missed
            ) * 0.5
        )

    return torch.stack(
        losses,
        dim=0,
    ).mean(dim=0)


def raster_loss(
    logits,
    target,
    bce_weight=1.0,
    dice_weight=1.0,
    distance_weight=0.5,
    max_positive_weight=30.0,
    distance_radii=(1, 2, 4),
    reduction="mean",
):
    """计算栅格火柴人综合损失。"""
    validate_raster_tensors(
        logits,
        target,
    )

    probabilities = torch.sigmoid(
        logits
    )

    (
        bce,
        positive_weight,
    ) = weighted_bce_per_sample(
        logits,
        target,
        max_positive_weight=(
            max_positive_weight
        ),
    )

    dice = dice_loss_per_sample(
        probabilities,
        target,
    )

    distance = (
        line_distance_loss_per_sample(
            probabilities,
            target,
            radii=distance_radii,
        )
    )

    total = (
        bce_weight * bce
        + dice_weight * dice
        + distance_weight * distance
    )

    foreground_fraction = (
        target
        .flatten(1)
        .mean(dim=1)
    )

    metrics = {
        "total": total,
        "bce": bce,
        "dice": dice,
        "distance": distance,
        "foreground_fraction": (
            foreground_fraction
        ),
        "positive_weight": (
            positive_weight
        ),
    }

    if reduction == "none":
        return metrics

    if reduction == "mean":
        return {
            name: values.mean()
            for name, values
            in metrics.items()
        }

    raise ValueError(
        "reduction 必须是 "
        "'mean' 或 'none'"
    )