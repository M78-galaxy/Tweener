"""结构化火柴人插帧请求的读取与校验。"""

import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def parse_request(
    data,
    default_request_id="request",
):
    """解析内存中的请求字典。"""
    if not isinstance(data, dict):
        raise TypeError(
            "请求必须是字典"
        )

    schema_version = int(
        data.get(
            "schema_version",
            SCHEMA_VERSION,
        )
    )

    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            "不支持的请求格式版本："
            f"{schema_version}"
        )

    request_id = str(
        data.get(
            "request_id",
            default_request_id,
        )
    ).strip()

    if not request_id:
        raise ValueError(
            "request_id 不能为空"
        )

    context = int(
        data["context"]
    )
    stride = int(
        data["stride"]
    )

    if context <= 0:
        raise ValueError(
            "context 必须大于 0"
        )

    if stride <= 0:
        raise ValueError(
            "stride 必须大于 0"
        )

    poses = data.get("poses")
    time_offsets = data.get(
        "time_offsets"
    )

    if not isinstance(poses, dict):
        raise ValueError(
            "poses 必须是字典"
        )

    if not isinstance(
        time_offsets,
        dict,
    ):
        raise ValueError(
            "time_offsets 必须是字典"
        )

    left_poses = np.asarray(
        poses["left"],
        dtype=np.float32,
    )
    right_poses = np.asarray(
        poses["right"],
        dtype=np.float32,
    )

    target_data = poses.get(
        "target"
    )

    target_pose = (
        None
        if target_data is None
        else np.asarray(
            target_data,
            dtype=np.float32,
        )
    )

    left_offsets = np.asarray(
        time_offsets["left"],
        dtype=np.int64,
    )
    right_offsets = np.asarray(
        time_offsets["right"],
        dtype=np.int64,
    )

    expected_pose_shape = (
        context,
        17,
        2,
    )
    expected_offset_shape = (
        context,
    )

    if left_poses.shape != expected_pose_shape:
        raise ValueError(
            "left poses 形状错误："
            f"期望 {expected_pose_shape}，"
            f"实际 {left_poses.shape}"
        )

    if right_poses.shape != expected_pose_shape:
        raise ValueError(
            "right poses 形状错误："
            f"期望 {expected_pose_shape}，"
            f"实际 {right_poses.shape}"
        )

    if (
        target_pose is not None
        and target_pose.shape != (17, 2)
    ):
        raise ValueError(
            "target pose 形状错误："
            f"{target_pose.shape}"
        )

    if (
        left_offsets.shape
        != expected_offset_shape
    ):
        raise ValueError(
            "left offsets 形状错误："
            f"{left_offsets.shape}"
        )

    if (
        right_offsets.shape
        != expected_offset_shape
    ):
        raise ValueError(
            "right offsets 形状错误："
            f"{right_offsets.shape}"
        )

    if not np.all(
        left_offsets < 0
    ):
        raise ValueError(
            "左侧时间偏移必须全部小于 0"
        )

    if not np.all(
        right_offsets > 0
    ):
        raise ValueError(
            "右侧时间偏移必须全部大于 0"
        )

    if left_offsets[-1] != -stride:
        raise ValueError(
            "左端点偏移与 stride 不一致："
            f"{left_offsets[-1]} 和 {-stride}"
        )

    if right_offsets[0] != stride:
        raise ValueError(
            "右端点偏移与 stride 不一致："
            f"{right_offsets[0]} 和 {stride}"
        )

    if not np.all(
        np.diff(left_offsets) > 0
    ):
        raise ValueError(
            "左侧时间偏移必须严格递增"
        )

    if not np.all(
        np.diff(right_offsets) > 0
    ):
        raise ValueError(
            "右侧时间偏移必须严格递增"
        )

    if not np.isfinite(
        left_poses
    ).all():
        raise ValueError(
            "left poses 存在非有限值"
        )

    if not np.isfinite(
        right_poses
    ).all():
        raise ValueError(
            "right poses 存在非有限值"
        )

    if (
        target_pose is not None
        and not np.isfinite(
            target_pose
        ).all()
    ):
        raise ValueError(
            "target pose 存在非有限值"
        )

    target_frame_data = data.get(
        "target_frame"
    )

    target_frame = (
        None
        if target_frame_data is None
        else int(target_frame_data)
    )

    if (
        target_frame is not None
        and target_frame < 0
    ):
        raise ValueError(
            "target_frame 不能小于 0"
        )

    sequence_id_data = data.get(
        "sequence_id"
    )
    sequence_id = (
        None
        if sequence_id_data is None
        else str(sequence_id_data)
    )

    return {
        "schema_version": schema_version,
        "request_id": request_id,
        "sequence_id": sequence_id,
        "target_frame": target_frame,
        "motion": data.get("motion"),
        "kind": data.get("kind"),
        "meta": data,
        "context": context,
        "stride": stride,
        "left_poses": left_poses,
        "target_pose": target_pose,
        "right_poses": right_poses,
        "left_offsets": left_offsets,
        "right_offsets": right_offsets,
    }


def load_request(path):
    """读取 JSON 文件并解析请求。"""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"请求文件不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    default_request_id = (
        path.parent.name
        if path.name == "meta.json"
        else path.stem
    )

    return parse_request(
        data,
        default_request_id=(
            default_request_id
        ),
    )