"""结构化火柴人服务后端烟雾测试。"""

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from server.backends.stickman import (
    StickmanBackend,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/formal_sequence/"
            "sequence_transformer_k3/"
            "20260816_182502/best.pt"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/v5_backend_smoke"
        ),
    )

    return parser.parse_args()


def load_json(path):
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def verify_result(
    result,
    reference_dir,
):
    reference_dir = Path(
        reference_dir
    )

    reference_pose = np.load(
        reference_dir
        / "prediction_pose.npy"
    )
    reference_png = (
        reference_dir
        / "prediction.png"
    ).read_bytes()

    pose_equal = np.array_equal(
        result.pose,
        reference_pose,
    )
    png_equal = (
        result.png == reference_png
    )

    if not pose_equal:
        difference = float(
            np.max(
                np.abs(
                    result.pose
                    - reference_pose
                )
            )
        )
        raise AssertionError(
            "后端与 CLI 姿态不一致，"
            f"最大差={difference}"
        )

    if not png_equal:
        raise AssertionError(
            "后端与 CLI PNG 字节不一致"
        )

    with Image.open(
        BytesIO(result.png)
    ) as image:
        image.load()
        image_size = image.size
        image_format = image.format

    if image_size != (
        result.size,
        result.size,
    ):
        raise AssertionError(
            "PNG 尺寸错误："
            f"{image_size}"
        )

    if image_format != "PNG":
        raise AssertionError(
            "返回图片不是 PNG"
        )

    return {
        "pose_equal": pose_equal,
        "png_equal": png_equal,
        "png_size": image_size,
        "png_bytes": len(result.png),
    }


def main():
    args = parse_args()

    backend = StickmanBackend(
        checkpoint=args.checkpoint,
        device=args.device,
        size=256,
    )

    model_id_before = id(
        backend.interpolator
        .predictor.model
    )

    # 单请求：直接把 JSON 读成内存字典。
    single_request_path = Path(
        "runs/v5_minimal_request/"
        "request.json"
    )
    single_request = load_json(
        single_request_path
    )

    single_result = (
        backend.interpolate_midpoint(
            single_request
        )
    )

    single_check = verify_result(
        single_result,
        "runs/v5_shared_request_smoke",
    )

    # 批量请求：仍然只传内存字典。
    manifest_path = Path(
        "runs/v5_batch_smoke/"
        "manifest.json"
    )
    manifest = load_json(
        manifest_path
    )

    batch_requests = [
        load_json(
            manifest_path.parent
            / relative_path
        )
        for relative_path
        in manifest["requests"]
    ]

    batch_results = (
        backend.interpolate_many(
            batch_requests
        )
    )

    batch_checks = []

    for result in batch_results:
        stride_name = (
            f"s{result.stride}"
        )
        reference_dir = (
            Path(
                "runs/v5_shared_batch_smoke"
            )
            / result.request_id
        )

        check = verify_result(
            result,
            reference_dir,
        )

        batch_checks.append(
            {
                "request_id": (
                    result.request_id
                ),
                "stride": result.stride,
                **check,
            }
        )

        print(
            stride_name,
            result.request_id,
            "姿态一致=",
            check["pose_equal"],
            "PNG一致=",
            check["png_equal"],
        )

    model_id_after = id(
        backend.interpolator
        .predictor.model
    )

    model_reused = (
        model_id_before
        == model_id_after
    )

    if not model_reused:
        raise AssertionError(
            "批量处理中模型实例发生变化"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        args.out
        / "single_prediction.png"
    ).write_bytes(
        single_result.png
    )

    np.save(
        args.out
        / "single_prediction_pose.npy",
        single_result.pose,
    )

    summary = {
        "checkpoint": str(
            args.checkpoint
        ),
        "epoch": backend.epoch,
        "device": str(
            backend.device
        ),
        "context": backend.context,
        "strides": list(
            backend.strides
        ),
        "model_reused": model_reused,
        "single": {
            **single_result.metadata(),
            **single_check,
        },
        "batch_count": len(
            batch_results
        ),
        "batch": batch_checks,
    }

    summary_path = (
        args.out / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("后端烟雾测试通过")
    print("设备：", backend.device)
    print("epoch：", backend.epoch)
    print("context：", backend.context)
    print("strides：", backend.strides)
    print(
        "单请求姿态一致：",
        single_check["pose_equal"],
    )
    print(
        "单请求 PNG 一致：",
        single_check["png_equal"],
    )
    print(
        "批量请求数量：",
        len(batch_results),
    )
    print(
        "模型实例复用：",
        model_reused,
    )
    print("输出目录：", args.out)
    print("测试记录：", summary_path)


if __name__ == "__main__":
    main()