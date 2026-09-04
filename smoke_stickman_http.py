"""Tweener 火柴人 HTTP 服务烟雾测试。"""

import argparse
import base64
import json
from pathlib import Path
from urllib.request import (
    Request,
    urlopen,
)

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default=(
            "http://127.0.0.1:8765"
        ),
    )
    parser.add_argument(
        "--request",
        type=Path,
        default=Path(
            "runs/v5_minimal_request/"
            "request.json"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "runs/"
            "v5_shared_request_smoke"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/v5_http_smoke"
        ),
    )

    return parser.parse_args()


def get_json(url):
    request = Request(
        url,
        method="GET",
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:
        status = response.status
        content_type = (
            response.headers
            .get_content_type()
        )
        body = response.read()

    if status != 200:
        raise RuntimeError(
            f"GET {url} 返回 {status}"
        )

    if content_type != (
        "application/json"
    ):
        raise RuntimeError(
            "GET 返回类型错误："
            f"{content_type}"
        )

    return json.loads(
        body.decode("utf-8")
    )


def post_json(
    url,
    payload,
):
    body = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": (
                "application/json"
            ),
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        status = response.status
        content_type = (
            response.headers
            .get_content_type()
        )
        response_body = (
            response.read()
        )

    if status != 200:
        raise RuntimeError(
            f"POST {url} 返回 {status}"
        )

    if content_type != (
        "application/json"
    ):
        raise RuntimeError(
            "POST 返回类型错误："
            f"{content_type}"
        )

    return json.loads(
        response_body.decode(
            "utf-8"
        )
    )


def main():
    args = parse_args()

    if not args.request.is_file():
        raise FileNotFoundError(
            f"请求文件不存在："
            f"{args.request}"
        )

    if not args.reference.is_dir():
        raise FileNotFoundError(
            "参考输出不存在："
            f"{args.reference}"
        )

    base_url = args.url.rstrip(
        "/"
    )

    health = get_json(
        base_url + "/health"
    )

    if health.get("status") != "ok":
        raise RuntimeError(
            "服务健康检查失败"
        )

    request_data = json.loads(
        args.request.read_text(
            encoding="utf-8"
        )
    )

    result = post_json(
        (
            base_url
            + "/v1/stickman/interpolate"
        ),
        request_data,
    )

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
            "HTTP 姿态形状错误："
            f"{pose.shape}"
        )

    if not np.isfinite(
        pose
    ).all():
        raise AssertionError(
            "HTTP 姿态存在非有限值"
        )

    if not png.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        raise AssertionError(
            "HTTP 返回的 PNG 无效"
        )

    reference_pose = np.load(
        args.reference
        / "prediction_pose.npy"
    )
    reference_png = (
        args.reference
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
            "HTTP 与 CLI 姿态不一致"
        )

    if not png_equal:
        raise AssertionError(
            "HTTP 与 CLI PNG 不一致"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        args.out
        / "prediction_pose.npy",
        pose,
    )

    (
        args.out
        / "prediction.png"
    ).write_bytes(
        png
    )

    record = dict(result)
    record["png_base64"] = (
        f"<省略 {len(result['png_base64'])} "
        "个 Base64 字符>"
    )
    record["health"] = health
    record["pose_equal_cli"] = (
        pose_equal
    )
    record["png_equal_cli"] = (
        png_equal
    )

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

    print("HTTP 烟雾测试通过")
    print("服务状态：", health["status"])
    print(
        "设备：",
        health["device"],
    )
    print(
        "epoch：",
        health["checkpoint_epoch"],
    )
    print(
        "request id：",
        result["request_id"],
    )
    print("姿态形状：", pose.shape)
    print(
        "姿态全部有限：",
        bool(np.isfinite(pose).all()),
    )
    print("PNG 字节数：", len(png))
    print(
        "姿态与 CLI 一致：",
        pose_equal,
    )
    print(
        "PNG 与 CLI 一致：",
        png_equal,
    )
    print("输出目录：", args.out)
    print("响应记录：", record_path)


if __name__ == "__main__":
    main()