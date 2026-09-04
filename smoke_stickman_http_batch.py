"""Tweener 火柴人批量 HTTP 接口烟雾测试。"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from smoke_stickman_http import (
    get_json,
    post_json,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default=(
            "http://127.0.0.1:8765"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "runs/v5_batch_smoke/"
            "manifest.json"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "runs/"
            "v5_shared_batch_smoke"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/v5_http_batch_smoke"
        ),
    )

    return parser.parse_args()


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    args = parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"manifest 不存在："
            f"{args.manifest}"
        )

    if not args.reference.is_dir():
        raise FileNotFoundError(
            "参考目录不存在："
            f"{args.reference}"
        )

    base_url = args.url.rstrip(
        "/"
    )

    health = get_json(
        base_url + "/health"
    )

    manifest = load_json(
        args.manifest
    )

    requests = [
        load_json(
            args.manifest.parent
            / relative_path
        )
        for relative_path
        in manifest["requests"]
    ]

    payload = {
        "schema_version": 1,
        "batch_id": (
            manifest["batch_id"]
        ),
        "requests": requests,
    }

    response = post_json(
        (
            base_url
            + "/v1/stickman/"
            "interpolate-batch"
        ),
        payload,
    )

    results = response.get(
        "results"
    )

    if not isinstance(
        results,
        list,
    ):
        raise AssertionError(
            "HTTP 批量响应没有 results 列表"
        )

    if len(results) != len(
        requests
    ):
        raise AssertionError(
            "批量结果数量错误："
            f"{len(results)}"
        )

    expected_ids = [
        request["request_id"]
        for request in requests
    ]
    actual_ids = [
        result["request_id"]
        for result in results
    ]

    if actual_ids != expected_ids:
        raise AssertionError(
            "批量结果顺序与请求不一致"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks = []
    saved_results = []

    for result in results:
        request_id = result[
            "request_id"
        ]

        pose = np.asarray(
            result["pose"],
            dtype=np.float32,
        )
        png = base64.b64decode(
            result["png_base64"],
            validate=True,
        )

        if pose.shape != (17, 2):
            raise AssertionError(
                f"{request_id} 姿态形状错误："
                f"{pose.shape}"
            )

        if not np.isfinite(
            pose
        ).all():
            raise AssertionError(
                f"{request_id} 存在非有限值"
            )

        if not png.startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise AssertionError(
                f"{request_id} PNG 无效"
            )

        reference_dir = (
            args.reference
            / request_id
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
            pose,
            reference_pose,
        )
        png_equal = (
            png == reference_png
        )

        if not pose_equal:
            raise AssertionError(
                f"{request_id} "
                "姿态与参考不一致"
            )

        if not png_equal:
            raise AssertionError(
                f"{request_id} "
                "PNG 与参考不一致"
            )

        output_dir = (
            args.out / request_id
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_dir
            / "prediction_pose.npy",
            pose,
        )
        (
            output_dir
            / "prediction.png"
        ).write_bytes(
            png
        )

        checks.append(
            {
                "request_id": request_id,
                "stride": result["stride"],
                "pose_equal": pose_equal,
                "png_equal": png_equal,
                "png_bytes": len(png),
            }
        )

        saved_result = dict(result)
        saved_result["png_base64"] = (
            f"<省略 "
            f"{len(result['png_base64'])} "
            "个 Base64 字符>"
        )
        saved_results.append(
            saved_result
        )

        print(
            request_id,
            "stride=",
            result["stride"],
            "姿态一致=",
            pose_equal,
            "PNG一致=",
            png_equal,
        )

    record = {
        "health": health,
        "schema_version": response[
            "schema_version"
        ],
        "batch_id": response[
            "batch_id"
        ],
        "request_count": response[
            "request_count"
        ],
        "checkpoint_epoch": response[
            "checkpoint_epoch"
        ],
        "device": response["device"],
        "context": response["context"],
        "strides": response["strides"],
        "checks": checks,
        "results": saved_results,
    }

    record_path = (
        args.out / "response.json"
    )

    record_path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("批量 HTTP 烟雾测试通过")
    print(
        "batch id：",
        response["batch_id"],
    )
    print(
        "请求数量：",
        response["request_count"],
    )
    print(
        "设备：",
        response["device"],
    )
    print(
        "epoch：",
        response["checkpoint_epoch"],
    )
    print(
        "结果顺序一致：",
        actual_ids == expected_ids,
    )
    print("输出目录：", args.out)
    print("响应记录：", record_path)


if __name__ == "__main__":
    main()