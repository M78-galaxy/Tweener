"""批量执行火柴人多帧插帧请求。"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from infer_stickman import (
    run_request,
)
from stickman.request import (
    load_request,
)
from stickman.interpolator import (
    StickmanInterpolator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "使用同一个 Transformer "
            "批量处理多个插帧请求"
        ),
    )

    parser.add_argument(
        "manifest",
        type=Path,
        help="批量请求 manifest.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Transformer checkpoint",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="推理设备；不指定时自动选择",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/v5_batch_inference"
        ),
        help="批量输出根目录",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="输出图像尺寸",
    )

    return parser.parse_args()


def load_manifest(manifest_path):
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"批量清单不存在：{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    schema_version = int(
        manifest.get("schema_version", 1)
    )

    if schema_version != 1:
        raise ValueError(
            "不支持的批量清单版本："
            f"{schema_version}"
        )

    batch_id = str(
        manifest.get(
            "batch_id",
            manifest_path.stem,
        )
    )

    request_entries = manifest.get(
        "requests"
    )

    if (
        not isinstance(request_entries, list)
        or not request_entries
    ):
        raise ValueError(
            "manifest.requests "
            "必须是非空列表"
        )

    request_paths = []

    for entry in request_entries:
        if not isinstance(entry, str):
            raise ValueError(
                "manifest.requests "
                "中的每一项必须是字符串"
            )

        relative_path = Path(entry)

        if relative_path.is_absolute():
            raise ValueError(
                "请求路径必须相对于 "
                f"manifest：{entry}"
            )

        request_path = (
            manifest_path.parent
            / relative_path
        )

        if not request_path.is_file():
            raise FileNotFoundError(
                f"请求文件不存在：{request_path}"
            )

        request_paths.append(
            request_path
        )

    if len(set(request_paths)) != len(
        request_paths
    ):
        raise ValueError(
            "manifest 中存在重复请求路径"
        )

    return {
        "schema_version": schema_version,
        "batch_id": batch_id,
        "request_paths": request_paths,
    }


def validate_request_id(request_id):
    if (
        not request_id
        or Path(request_id).name
        != request_id
        or request_id in (".", "..")
    ):
        raise ValueError(
            "request_id 不能包含路径："
            f"{request_id!r}"
        )


def main():
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint 不存在："
            f"{args.checkpoint}"
        )

    if args.size <= 0:
        raise ValueError(
            "size 必须大于 0"
        )

    batch = load_manifest(
        args.manifest
    )

    # 整个批次只在这里加载一次模型。
    interpolator = StickmanInterpolator(
        checkpoint=args.checkpoint,
        device=args.device,
        size=args.size,
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    request_ids = set()
    request_paths = batch[
        "request_paths"
    ]

    for index, request_path in enumerate(
        request_paths,
        start=1,
    ):
        sample = load_request(
            request_path
        )
        request_id = sample["request_id"]

        validate_request_id(
            request_id
        )

        if request_id in request_ids:
            raise ValueError(
                "批次中存在重复 request_id："
                f"{request_id}"
            )

        request_ids.add(
            request_id
        )

        output_dir = (
            args.out / request_id
        )

        request_args = SimpleNamespace(
            meta=request_path,
            checkpoint=args.checkpoint,
            device=args.device,
            out=output_dir,
            size=args.size,
        )

        print()
        print(
            f"===== 请求 "
            f"{index}/{len(request_paths)} "
            f"====="
        )

        result = run_request(
            request_args,
            interpolator=interpolator,
        )

        results.append(
            result
        )

    batch_result = {
        "schema_version": (
            batch["schema_version"]
        ),
        "batch_id": batch["batch_id"],
        "manifest": str(args.manifest),
        "checkpoint": str(
            args.checkpoint
        ),
        "checkpoint_epoch": (
            interpolator.epoch
        ),
        "device": str(
            interpolator.device
        ),
        "size": args.size,
        "request_count": len(results),
        "request_ids": [
            result["request_id"]
            for result in results
        ],
        "results": results,
    }

    batch_result_path = (
        args.out / "batch_result.json"
    )

    with batch_result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            batch_result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("批量推理完成")
    print(
        "batch id：",
        batch["batch_id"],
    )
    print("请求数量：", len(results))
    print(
        "设备：",
        interpolator.device,
    )
    print(
        "checkpoint epoch：",
        interpolator.epoch,
    )
    print("输出目录：", args.out)
    print(
        "批量记录：",
        batch_result_path,
    )


if __name__ == "__main__":
    main()