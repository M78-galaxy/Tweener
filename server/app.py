"""Tweener 结构化火柴人本机推理服务。"""

import argparse
import base64
import json
import traceback
from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
)
from pathlib import Path
from urllib.parse import urlsplit

from server.backends.stickman import (
    StickmanBackend,
)


MAX_BODY_BYTES = (
    16 * 1024 * 1024
)

def result_to_payload(result):
    """把内存结果转换为 HTTP JSON。"""
    payload = result.metadata()
    payload["png_base64"] = (
        base64.b64encode(
            result.png
        ).decode("ascii")
    )
    return payload

def make_handler(backend):
    """创建绑定到指定后端的请求处理器。"""

    class StickmanRequestHandler(
        BaseHTTPRequestHandler
    ):
        server_version = (
            "TweenerStickman/0.1"
        )

        def send_bytes(
            self,
            status,
            content_type,
            body,
        ):
            self.send_response(
                status
            )
            self.send_header(
                "Content-Type",
                content_type,
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.send_header(
                "Cache-Control",
                "no-store",
            )
            self.end_headers()
            self.wfile.write(
                body
            )

        def send_json(
            self,
            status,
            payload,
        ):
            body = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")

            self.send_bytes(
                status,
                (
                    "application/json; "
                    "charset=utf-8"
                ),
                body,
            )

        def request_path(self):
            return urlsplit(
                self.path
            ).path

        def read_json(self):
            content_length = (
                self.headers.get(
                    "Content-Length"
                )
            )

            if content_length is None:
                raise ValueError(
                    "缺少 Content-Length"
                )

            try:
                content_length = int(
                    content_length
                )
            except ValueError as error:
                raise ValueError(
                    "Content-Length 无效"
                ) from error

            if content_length <= 0:
                raise ValueError(
                    "请求正文不能为空"
                )

            if (
                content_length
                > MAX_BODY_BYTES
            ):
                raise ValueError(
                    "请求正文超过 "
                    f"{MAX_BODY_BYTES} 字节"
                )

            body = self.rfile.read(
                content_length
            )

            try:
                text = body.decode(
                    "utf-8"
                )
            except UnicodeDecodeError as error:
                raise ValueError(
                    "请求正文不是 UTF-8"
                ) from error

            try:
                payload = json.loads(
                    text
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    "请求正文不是有效 JSON"
                ) from error

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "JSON 顶层必须是字典"
                )

            return payload

        def do_GET(self):
            path = self.request_path()

            if path == "/health":
                self.send_json(
                    200,
                    {
                        "status": "ok",
                        "backend": "stickman",
                        "checkpoint": str(
                            backend.checkpoint
                        ),
                        "checkpoint_epoch": (
                            backend.epoch
                        ),
                        "device": str(
                            backend.device
                        ),
                        "context": (
                            backend.context
                        ),
                        "strides": list(
                            backend.strides
                        ),
                        "size": backend.size,
                    },
                )
                return

            self.send_json(
                404,
                {
                    "error": "not_found",
                    "message": (
                        f"没有这个接口：{path}"
                    ),
                },
            )

        def do_POST(self):
            path = self.request_path()

            single_path = (
                "/v1/stickman/interpolate"
            )
            batch_path = (
                "/v1/stickman/"
                "interpolate-batch"
            )

            if path not in (
                single_path,
                batch_path,
            ):
                self.send_json(
                    404,
                    {
                        "error": "not_found",
                        "message": (
                            "没有这个接口："
                            f"{path}"
                        ),
                    },
                )
                return

            try:
                payload = self.read_json()

                if path == single_path:
                    result = (
                        backend
                        .interpolate_midpoint(
                            payload
                        )
                    )
                    response = (
                        result_to_payload(
                            result
                        )
                    )

                else:
                    schema_version = int(
                        payload.get(
                            "schema_version",
                            1,
                        )
                    )

                    if schema_version != 1:
                        raise ValueError(
                            "不支持的批量请求"
                            "格式版本："
                            f"{schema_version}"
                        )

                    batch_id = str(
                        payload.get(
                            "batch_id",
                            "batch",
                        )
                    ).strip()

                    if not batch_id:
                        raise ValueError(
                            "batch_id 不能为空"
                        )

                    requests = payload.get(
                        "requests"
                    )

                    if (
                        not isinstance(
                            requests,
                            list,
                        )
                        or not requests
                    ):
                        raise ValueError(
                            "requests 必须是"
                            "非空列表"
                        )

                    results = (
                        backend
                        .interpolate_many(
                            requests
                        )
                    )

                    response = {
                        "schema_version": 1,
                        "batch_id": batch_id,
                        "request_count": len(
                            results
                        ),
                        "checkpoint_epoch": (
                            backend.epoch
                        ),
                        "device": str(
                            backend.device
                        ),
                        "context": (
                            backend.context
                        ),
                        "strides": list(
                            backend.strides
                        ),
                        "results": [
                            result_to_payload(
                                result
                            )
                            for result
                            in results
                        ],
                    }

                self.send_json(
                    200,
                    response,
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                self.send_json(
                    400,
                    {
                        "error": (
                            "invalid_request"
                        ),
                        "message": str(
                            error
                        ),
                    },
                )

            except Exception:
                traceback.print_exc()

                self.send_json(
                    500,
                    {
                        "error": (
                            "internal_error"
                        ),
                        "message": (
                            "服务器内部错误"
                        ),
                    },
                )
    return StickmanRequestHandler


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "启动 Tweener 火柴人"
            "本机推理服务"
        ),
    )

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
        default=None,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "默认仅监听本机，"
            "不要随意改成 0.0.0.0"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 1 <= args.port <= 65535:
        raise ValueError(
            "port 必须位于 "
            "1 到 65535"
        )

    backend = StickmanBackend(
        checkpoint=args.checkpoint,
        device=args.device,
        size=args.size,
    )

    handler = make_handler(
        backend
    )

    server = HTTPServer(
        (args.host, args.port),
        handler,
    )

    print("Tweener 火柴人服务已启动")
    print(
        "地址：",
        f"http://{args.host}:{args.port}",
    )
    print("设备：", backend.device)
    print("epoch：", backend.epoch)
    print("context：", backend.context)
    print("strides：", backend.strides)
    print("停止服务：Ctrl+C")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("正在停止服务……")
    finally:
        server.server_close()

    print("服务已停止")


if __name__ == "__main__":
    main()