#!/usr/bin/env python3
"""可视化 Transformer 相对 MLP 的最好与最差案例。"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stickman.models import (
    MLPPredictor,
    SequencePredictor,
)
from stickman.skeleton import (
    render,
    render_overlay,
)


STRIDES = (2, 4, 8)
REQUIRED_METHODS = {
    "lerp",
    "mlp",
    "transformer",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-run",
        required=True,
        help="evaluate_sequence.py 的输出目录",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
    )

    return parser.parse_args()


def add_label(image, text):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [0, 0, image.width, 18],
        fill=(240, 240, 240),
    )
    draw.text(
        (3, 3),
        text,
        fill=(0, 0, 0),
    )

    return image


def load_results(metrics_path):
    samples = {}

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            clip = row["clip"]
            method = row["method"]

            if clip not in samples:
                samples[clip] = {
                    "clip": clip,
                    "action": row["action"],
                    "motion": row["motion"],
                    "kind": row["kind"],
                    "stride": int(
                        row["stride"]
                    ),
                    "metrics": {},
                }

            samples[clip]["metrics"][
                method
            ] = {
                "mpjpe_px": float(
                    row["mpjpe_px"]
                ),
                "bone_err_px": float(
                    row["bone_err_px"]
                ),
            }

    for clip, sample in samples.items():
        methods = set(
            sample["metrics"]
        )

        if methods != REQUIRED_METHODS:
            raise RuntimeError(
                f"{clip} 的方法不完整："
                f"{sorted(methods)}"
            )

    return list(samples.values())


def select_best_and_worst(samples):
    grouped = defaultdict(list)

    for sample in samples:
        grouped[
            (
                sample["stride"],
                sample["kind"],
            )
        ].append(sample)

    selected = {
        stride: []
        for stride in STRIDES
    }

    for (
        stride,
        kind,
    ), group in sorted(grouped.items()):
        # 负数代表 Transformer 比 MLP 更好。
        ordered = sorted(
            group,
            key=lambda sample: (
                sample["metrics"][
                    "transformer"
                ]["mpjpe_px"]
                - sample["metrics"][
                    "mlp"
                ]["mpjpe_px"]
            ),
        )

        best = dict(ordered[0])
        best["case"] = "best_vs_mlp"

        worst = dict(ordered[-1])
        worst["case"] = "worst_vs_mlp"

        selected[stride].extend(
            (best, worst)
        )

    return selected


def load_meta(
    data_base,
    context,
    sample,
):
    meta_path = (
        Path(data_base)
        / f"{sample['action']}_val_001"
        / f"k{context}"
        / f"s{sample['stride']}"
        / sample["clip"]
        / "meta.json"
    )

    if not meta_path.is_file():
        raise FileNotFoundError(
            f"找不到样本：{meta_path}"
        )

    with meta_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def make_grid(
    samples,
    data_base,
    context,
    sequence_predictor,
    mlp_predictors,
    output_path,
):
    rows = len(samples)
    columns = 6
    size = 256

    sheet = Image.new(
        "RGB",
        (
            columns * size,
            rows * size,
        ),
        (255, 255, 255),
    )

    for row_index, sample in enumerate(
        samples
    ):
        meta = load_meta(
            data_base,
            context,
            sample,
        )

        left = np.asarray(
            meta["poses"]["left"],
            dtype=np.float32,
        )
        right = np.asarray(
            meta["poses"]["right"],
            dtype=np.float32,
        )
        target = np.asarray(
            meta["poses"]["target"],
            dtype=np.float32,
        )

        stride = sample["stride"]

        predictions = {
            "lerp": (
                left[-1] + right[0]
            ) * 0.5,
            "mlp": (
                mlp_predictors[
                    stride
                ].predict(
                    left[-1],
                    right[0],
                )
            ),
            "transformer": (
                sequence_predictor.predict(
                    left,
                    right,
                    meta[
                        "time_offsets"
                    ]["left"],
                    meta[
                        "time_offsets"
                    ]["right"],
                )
            ),
        }

        metrics = sample["metrics"]

        row_name = (
            f"{sample['kind']} "
            f"{sample['case']}"
        )

        cells = [
            add_label(
                render(
                    left[-1],
                    size=size,
                ),
                f"{row_name} | prev",
            ),
            add_label(
                render_overlay(
                    predictions["lerp"],
                    target,
                    size=size,
                ),
                (
                    "LERP "
                    f"mp={metrics['lerp']['mpjpe_px']:.2f} "
                    f"bone={metrics['lerp']['bone_err_px']:.2f}"
                ),
            ),
            add_label(
                render_overlay(
                    predictions["mlp"],
                    target,
                    size=size,
                ),
                (
                    "MLP "
                    f"mp={metrics['mlp']['mpjpe_px']:.2f} "
                    f"bone={metrics['mlp']['bone_err_px']:.2f}"
                ),
            ),
            add_label(
                render_overlay(
                    predictions[
                        "transformer"
                    ],
                    target,
                    size=size,
                ),
                (
                    "Transformer "
                    f"mp={metrics['transformer']['mpjpe_px']:.2f} "
                    f"bone={metrics['transformer']['bone_err_px']:.2f}"
                ),
            ),
            add_label(
                render(
                    target,
                    size=size,
                ),
                "ground truth",
            ),
            add_label(
                render(
                    right[0],
                    size=size,
                ),
                "next",
            ),
        ]

        for column, cell in enumerate(
            cells
        ):
            sheet.paste(
                cell,
                (
                    column * size,
                    row_index * size,
                ),
            )

    sheet.save(output_path)


def write_selection(path, selected):
    rows = []

    for stride in STRIDES:
        for sample in selected[stride]:
            metrics = sample["metrics"]

            rows.append(
                {
                    "clip": sample["clip"],
                    "action": sample["action"],
                    "kind": sample["kind"],
                    "stride": stride,
                    "case": sample["case"],
                    "lerp_mpjpe": (
                        metrics["lerp"][
                            "mpjpe_px"
                        ]
                    ),
                    "mlp_mpjpe": (
                        metrics["mlp"][
                            "mpjpe_px"
                        ]
                    ),
                    "transformer_mpjpe": (
                        metrics[
                            "transformer"
                        ]["mpjpe_px"]
                    ),
                    "transformer_minus_mlp": (
                        metrics[
                            "transformer"
                        ]["mpjpe_px"]
                        - metrics["mlp"][
                            "mpjpe_px"
                        ]
                    ),
                    "lerp_bone": (
                        metrics["lerp"][
                            "bone_err_px"
                        ]
                    ),
                    "mlp_bone": (
                        metrics["mlp"][
                            "bone_err_px"
                        ]
                    ),
                    "transformer_bone": (
                        metrics[
                            "transformer"
                        ]["bone_err_px"]
                    ),
                }
            )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    eval_run = Path(args.eval_run)

    config_path = (
        eval_run / "config.json"
    )
    metrics_path = (
        eval_run / "metrics.csv"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"找不到：{config_path}"
        )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"找不到：{metrics_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    context = int(config["context"])
    data_base = config["data_base"]

    sequence_predictor = (
        SequencePredictor(
            config[
                "sequence_checkpoint"
            ],
            device=args.device,
        )
    )

    mlp_checkpoints = (
        config["mlp_checkpoints"]
    )

    mlp_predictors = {
        stride: MLPPredictor(
            mlp_checkpoints[str(stride)],
            device=args.device,
        )
        for stride in STRIDES
    }

    samples = load_results(
        metrics_path
    )

    selected = select_best_and_worst(
        samples
    )

    for stride in STRIDES:
        output_path = (
            eval_run
            / (
                f"grid_s{stride}"
                "_best_worst.png"
            )
        )

        make_grid(
            selected[stride],
            data_base,
            context,
            sequence_predictor,
            mlp_predictors,
            output_path,
        )

        print(
            f"s{stride}："
            f"{len(selected[stride])} 行 → "
            f"{output_path}"
        )

    selection_path = (
        eval_run / "selection.csv"
    )

    write_selection(
        selection_path,
        selected,
    )
 
    print(
        f"选择明细：{selection_path}"
    )



if __name__ == "__main__":
    main()
