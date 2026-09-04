#!/usr/bin/env python3
"""从 Raster 正式评测结果中选择诊断案例。"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from summarize_raster_eval import (
    METHODS,
    STRIDES,
    load_rows,
    validate_rows,
)


OUTPUT_METRICS = (
    "soft_dice",
    "soft_iou",
    "hard_dice",
    "hard_iou",
    "mae",
    "line_distance",
)

SELECTION_TYPES = (
    "transformer_over_raster",
    "raster_over_transformer",
    "last_over_best",
    "best_over_last",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "metrics",
        help="正式评测生成的 metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
    )

    return parser.parse_args()


def build_samples(rows):
    grouped = defaultdict(dict)

    for row in rows:
        grouped[
            row["meta_path"]
        ][row["method"]] = row

    samples = []

    for meta_path, method_rows in (
        grouped.items()
    ):
        missing = (
            set(METHODS)
            - set(method_rows)
        )

        if missing:
            raise RuntimeError(
                f"{meta_path} 缺少方法："
                f"{sorted(missing)}"
            )

        reference = method_rows[
            METHODS[0]
        ]

        samples.append(
            {
                "meta_path": meta_path,
                "clip": reference["clip"],
                "motion": reference["motion"],
                "kind": reference["kind"],
                "stride": (
                    reference["stride"]
                ),
                "context": (
                    reference["context"]
                ),
                "methods": method_rows,
            }
        )

    samples.sort(
        key=lambda sample: (
            sample["stride"],
            sample["kind"],
            sample["clip"],
        )
    )

    return samples


def hard_dice(
    sample,
    method,
):
    return sample[
        "methods"
    ][method]["hard_dice"]


def hard_dice_delta(
    sample,
    method_a,
    method_b,
):
    return (
        hard_dice(
            sample,
            method_a,
        )
        - hard_dice(
            sample,
            method_b,
        )
    )


def select_largest_delta(
    samples,
    method_a,
    method_b,
):
    if not samples:
        raise RuntimeError(
            "没有可供选择的样本"
        )

    return max(
        samples,
        key=lambda sample: (
            hard_dice_delta(
                sample,
                method_a,
                method_b,
            ),
            sample["clip"],
        ),
    )


def select_worst(
    samples,
    method,
):
    if not samples:
        raise RuntimeError(
            "没有可供选择的样本"
        )

    return min(
        samples,
        key=lambda sample: (
            hard_dice(
                sample,
                method,
            ),
            sample["clip"],
        ),
    )


def make_selection_row(
    selection_type,
    sample,
    method_a,
    method_b=None,
):
    if method_b is None:
        delta = ""
        actual_advantage = ""
    else:
        delta = hard_dice_delta(
            sample,
            method_a,
            method_b,
        )

        actual_advantage = (
            delta > 0.0
        )

    row = {
        "selection_type": (
            selection_type
        ),
        "stride": sample["stride"],
        "kind": sample["kind"],
        "clip": sample["clip"],
        "motion": sample["motion"],
        "context": sample["context"],
        "method_a": method_a,
        "method_b": (
            ""
            if method_b is None
            else method_b
        ),
        "delta_hard_dice": delta,
        "actual_advantage": (
            actual_advantage
        ),
        "meta_path": (
            sample["meta_path"]
        ),
    }

    for method in METHODS:
        method_values = sample[
            "methods"
        ][method]

        for metric in OUTPUT_METRICS:
            row[
                f"{method}_{metric}"
            ] = method_values[
                metric
            ]

    return row


def select_cases(samples):
    selections = []

    kinds = sorted(
        {
            sample["kind"]
            for sample in samples
        }
    )

    for stride in STRIDES:
        stride_samples = [
            sample
            for sample in samples
            if sample["stride"] == stride
        ]

        comparison_specs = (
            (
                "transformer_over_raster",
                "transformer",
                "raster_last",
            ),
            (
                "raster_over_transformer",
                "raster_last",
                "transformer",
            ),
            (
                "last_over_best",
                "raster_last",
                "raster_best",
            ),
            (
                "best_over_last",
                "raster_best",
                "raster_last",
            ),
        )

        for (
            selection_type,
            method_a,
            method_b,
        ) in comparison_specs:
            sample = (
                select_largest_delta(
                    stride_samples,
                    method_a,
                    method_b,
                )
            )

            selections.append(
                make_selection_row(
                    selection_type,
                    sample,
                    method_a,
                    method_b,
                )
            )

        for kind in kinds:
            kind_samples = [
                sample
                for sample in stride_samples
                if sample["kind"] == kind
            ]

            if not kind_samples:
                continue

            sample = select_worst(
                kind_samples,
                "raster_last",
            )

            selections.append(
                make_selection_row(
                    "raster_last_worst",
                    sample,
                    "raster_last",
                )
            )

    return selections


def fieldnames():
    names = [
        "selection_type",
        "stride",
        "kind",
        "clip",
        "motion",
        "context",
        "method_a",
        "method_b",
        "delta_hard_dice",
        "actual_advantage",
        "meta_path",
    ]

    for method in METHODS:
        for metric in OUTPUT_METRICS:
            names.append(
                f"{method}_{metric}"
            )

    return names


def write_csv(
    path,
    rows,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames(),
        )

        writer.writeheader()
        writer.writerows(rows)


def build_text(selections):
    lines = []

    lines.append(
        "Raster 正式评测诊断案例"
    )
    lines.append("=" * 100)
    lines.append(
        "delta_hard_dice = "
        "method_a - method_b"
    )
    lines.append("")

    for row in selections:
        if (
            row["delta_hard_dice"]
            == ""
        ):
            comparison = (
                "hard Dice="
                f"{row['raster_last_hard_dice']:.4f}"
            )
        else:
            comparison = (
                f"{row['method_a']} - "
                f"{row['method_b']} = "
                f"{row['delta_hard_dice']:+.4f}"
            )

        lines.append(
            f"s{row['stride']} | "
            f"{row['selection_type']:<28} | "
            f"{row['kind']:<20} | "
            f"{comparison} | "
            f"{row['clip']}"
        )

    return "\n".join(lines)


def main():
    args = parse_args()

    metrics_path = Path(
        args.metrics
    )

    rows = load_rows(
        metrics_path
    )

    validation = validate_rows(
        rows
    )

    samples = build_samples(
        rows
    )

    if (
        len(samples)
        != validation["samples"]
    ):
        raise RuntimeError(
            "样本重组数量错误："
            f"{len(samples)}"
        )

    selections = select_cases(
        samples
    )

    if args.out_dir is None:
        output_dir = (
            metrics_path.parent
        )
    else:
        output_dir = Path(
            args.out_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir / "selection.csv"
    )

    text_path = (
        output_dir / "selection.txt"
    )

    write_csv(
        csv_path,
        selections,
    )

    text = build_text(
        selections
    )

    text_path.write_text(
        text + "\n",
        encoding="utf-8",
    )

    unique_clips = {
        row["clip"]
        for row in selections
    }

    print("案例选择完成")
    print(
        f"选择记录：{len(selections)}"
    )
    print(
        f"唯一案例：{len(unique_clips)}"
    )
    print(f"CSV：{csv_path}")
    print(f"文本：{text_path}")
    print("")
    print(text)


if __name__ == "__main__":
    main()