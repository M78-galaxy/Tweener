"""姿态插帧训练损失。"""

import torch

from stickman.skeleton import LIMB_BONES


def limb_lengths(pose):
    """计算八段四肢骨骼长度。

    pose 的形状必须为 (..., 17, 2)。
    返回形状为 (..., 8)。
    """
    if pose.shape[-2:] != (17, 2):
        raise ValueError(
            "姿态最后两维必须是 (17, 2)，"
            f"实际是 {tuple(pose.shape[-2:])}"
        )

    bone_indices = torch.tensor(
        LIMB_BONES,
        dtype=torch.long,
        device=pose.device,
    )

    starts = pose[..., bone_indices[:, 0], :]
    ends = pose[..., bone_indices[:, 1], :]

    return torch.linalg.vector_norm(
        starts - ends,
        dim=-1,
    )


def pose_loss(
    prediction,
    target,
    bone_weight=0.5,
):
    """计算位置损失与骨长损失。

    所有计算都在以躯干长度归一化后的坐标中进行。
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction 和 target 形状必须一致，"
            f"实际是 {tuple(prediction.shape)} "
            f"和 {tuple(target.shape)}"
        )

    # 对应归一化坐标中的 MPJPE。
    position_loss = torch.linalg.vector_norm(
        prediction - target,
        dim=-1,
    ).mean()

    prediction_lengths = limb_lengths(prediction)
    target_lengths = limb_lengths(target)

    bone_loss = torch.abs(
        prediction_lengths - target_lengths
    ).mean()

    total_loss = (
        position_loss
        + bone_weight * bone_loss
    )

    return {
        "total": total_loss,
        "position": position_loss,
        "bone": bone_loss,
    }