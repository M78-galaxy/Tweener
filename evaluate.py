#!/usr/bin/env python3
"""在自建三连帧上评测各方法。"""

import argparse
import csv
import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stickman.baselines import (
    angle_lerp,
    lerp,
    zero_velocity,
)
from stickman.models import MLPPredictor
from stickman.skeleton import (
    bone_length_error,
    mpjpe,
    render,
    render_overlay,
)


# 不依赖 checkpoint 的静态方法。
METHODS = {
    "zero_velocity": lambda meta: zero_velocity(
        meta["poses"]["prev"],
        meta["poses"]["next"],
    ),
    "lerp": lambda meta: lerp(
        meta["poses"]["prev"],
        meta["poses"]["next"],
    ),
    "angle_lerp": lambda meta: angle_lerp(
        meta["params"]["prev"],
        meta["params"]["next"],
    ),
}


def load_clips(root):
    clips = []

    for directory in sorted(
        Path(root).iterdir()
    ):
        meta_path = directory / "meta.json"

        if not meta_path.exists():
            continue

        with meta_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            meta = json.load(file)

        meta["poses"] = {
            name: np.asarray(
                value,
                dtype=np.float32,
            )
            for name, value
            in meta["poses"].items()
        }

        # 合成数据有 params，真人数据没有。
        if "params" in meta:
            meta["params"] = {
                name: np.asarray(value)
                for name, value
                in meta["params"].items()
            }

        meta["dir"] = directory
        clips.append(meta)

    return clips


def label(image, text):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [0, 0, image.width, 14],
        fill=(240, 240, 240),
    )
    draw.text(
        (3, 2),
        text,
        fill=(0, 0, 0),
    )

    return image


def make_grid(
    clips,
    method,
    predictions,
    size,
    path,
    max_rows=6,
):
    """生成 prev、预测、真值、next 和叠图。"""
    step = max(
        1,
        len(clips) // max_rows,
    )

    picked = clips[::step][:max_rows]

    columns = 5
    rows = len(picked)

    sheet = Image.new(
        "RGB",
        (
            columns * size,
            rows * size,
        ),
        (255, 255, 255),
    )

    for row, clip in enumerate(picked):
        prediction = predictions[
            clip["dir"].name
        ]

        cells = [
            label(
                render(
                    clip["poses"]["prev"],
                    size,
                ),
                "prev",
            ),
            label(
                render(
                    prediction,
                    size,
                ),
                f"pred:{method}",
            ),
            label(
                render(
                    clip["poses"]["gt"],
                    size,
                ),
                "ground truth",
            ),
            label(
                render(
                    clip["poses"]["next"],
                    size,
                ),
                "next",
            ),
            label(
                render_overlay(
                    prediction,
                    clip["poses"]["gt"],
                    size,
                ),
                clip["kind"],
            ),
        ]

        for column, cell in enumerate(cells):
            sheet.paste(
                cell,
                (
                    column * size,
                    row * size,
                ),
            )

    sheet.save(path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/triplets",
    )
    parser.add_argument(
        "--runs",
        default="runs",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[
            "zero_velocity",
            "lerp",
        ],
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="MLP 的 best.pt；使用 mlp 方法时必须提供",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    clips = load_clips(args.data)

    if not clips:
        raise SystemExit(
            f"{args.data} 里没有三连帧数据"
        )

    sizes = {
        int(clip["size"])
        for clip in clips
    }
    strides = {
        int(clip["stride"])
        for clip in clips
    }

    if len(sizes) != 1:
        raise SystemExit(
            f"数据中存在多个画布尺寸：{sizes}"
        )

    if len(strides) != 1:
        raise SystemExit(
            f"数据中存在多个 stride：{strides}"
        )

    size = next(iter(sizes))
    data_stride = next(iter(strides))

    methods = dict(METHODS)
    predictor = None

    if args.checkpoint is not None:
        predictor = MLPPredictor(
            args.checkpoint,
            device=args.device,
        )

        if predictor.stride != data_stride:
            raise SystemExit(
                "checkpoint stride 与数据不一致："
                f"模型是 {predictor.stride}，"
                f"数据是 {data_stride}"
            )

        methods["mlp"] = lambda meta: (
            predictor.predict(
                meta["poses"]["prev"],
                meta["poses"]["next"],
            )
        )

    if (
        "mlp" in args.methods
        and predictor is None
    ):
        raise SystemExit(
            "使用 mlp 方法时必须提供 "
            "--checkpoint"
        )

    unknown_methods = (
        set(args.methods) - set(methods)
    )

    if unknown_methods:
        raise SystemExit(
            "未知方法："
            f"{sorted(unknown_methods)}；"
            "可用方法："
            f"{sorted(methods)}"
        )

    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        Path(args.runs) / timestamp
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    config = {
        "data": args.data,
        "methods": args.methods,
        "n_clips": len(clips),
        "stride": data_stride,
        "size": size,
        "checkpoint": args.checkpoint,
        "device": (
            str(predictor.device)
            if predictor is not None
            else None
        ),
        "model_epoch": (
            predictor.epoch
            if predictor is not None
            else None
        ),
    }

    with (
        run_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    metric_rows = []

    for method in args.methods:
        function = methods[method]
        predictions = {}

        for clip in clips:
            prediction = np.asarray(
                function(clip),
                dtype=np.float32,
            )

            if prediction.shape != (17, 2):
                raise RuntimeError(
                    f"{method} 对 "
                    f"{clip['dir'].name} "
                    "输出了错误形状："
                    f"{prediction.shape}"
                )

            if not np.isfinite(
                prediction
            ).all():
                raise RuntimeError(
                    f"{method} 对 "
                    f"{clip['dir'].name} "
                    "输出了 NaN 或无穷值"
                )

            predictions[
                clip["dir"].name
            ] = prediction

            metric_rows.append(
                {
                    "clip": clip["dir"].name,
                    "motion": clip["motion"],
                    "kind": clip["kind"],
                    "method": method,
                    "mpjpe_px": round(
                        mpjpe(
                            prediction,
                            clip["poses"]["gt"],
                            size,
                        ),
                        2,
                    ),
                    "bone_err_px": round(
                        bone_length_error(
                            prediction,
                            clip["poses"]["gt"],
                            size,
                        ),
                        2,
                    ),
                }
            )

        make_grid(
            clips,
            method,
            predictions,
            size,
            run_dir / f"grid_{method}.png",
            args.rows,
        )

    with (
        run_dir / "metrics.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                metric_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    aggregated = defaultdict(list)

    for row in metric_rows:
        aggregated[
            (
                row["kind"],
                row["method"],
            )
        ].append(
            (
                row["mpjpe_px"],
                row["bone_err_px"],
            )
        )

    lines = [
        (
            f"{'运动类型':<20}"
            f"{'方法':<16}"
            f"{'MPJPE(px)':>12}"
            f"{'骨长误差(px)':>16}"
        ),
        "-" * 66,
    ]

    kinds = sorted(
        {
            kind
            for kind, _
            in aggregated
        }
    )

    for kind in kinds:
        for method in args.methods:
            values = aggregated.get(
                (kind, method)
            )

            if not values:
                continue

            mean = np.asarray(
                values
            ).mean(axis=0)

            lines.append(
                f"{kind:<20}"
                f"{method:<16}"
                f"{mean[0]:>12.2f}"
                f"{mean[1]:>16.2f}"
            )

        lines.append("")

    summary = "\n".join(lines)

    (
        run_dir / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print(f"→ {run_dir}")


if __name__ == "__main__":
    main()