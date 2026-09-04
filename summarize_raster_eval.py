#!/usr/bin/env python3
"""校验并汇总 Raster 正式评测 CSV。"""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


METHODS = (
    "pixel_lerp",
    "pose_lerp",
    "transformer",
    "raster_best",
    "raster_last",
)

STRIDES = (2, 4, 8)

METRIC_NAMES = (
    "soft_dice",
    "soft_iou",
    "hard_dice",
    "hard_iou",
    "mae",
    "line_distance",
)

REQUIRED_COLUMNS = (
    "clip",
    "motion",
    "kind",
    "stride",
    "context",
    "method",
    *METRIC_NAMES,
    "meta_path",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "metrics",
        help="evaluate_raster.py 生成的 metrics.csv",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="summary.txt 输出位置",
    )

    return parser.parse_args()


def load_rows(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"指标文件不存在：{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "CSV 没有表头"
            )

        missing = (
            set(REQUIRED_COLUMNS)
            - set(reader.fieldnames)
        )

        if missing:
            raise RuntimeError(
                "CSV 缺少字段："
                f"{sorted(missing)}"
            )

        rows = []

        for csv_row in reader:
            row = {
                "clip": csv_row["clip"],
                "motion": csv_row["motion"],
                "kind": csv_row["kind"],
                "stride": int(
                    csv_row["stride"]
                ),
                "context": int(
                    csv_row["context"]
                ),
                "method": csv_row["method"],
                "meta_path": (
                    csv_row["meta_path"]
                ),
            }

            for name in METRIC_NAMES:
                value = float(
                    csv_row[name]
                )

                if not math.isfinite(value):
                    raise RuntimeError(
                        f"{row['clip']} 的 "
                        f"{row['method']}/{name} "
                        "不是有限值"
                    )

                row[name] = value

            rows.append(row)

    if not rows:
        raise RuntimeError(
            "CSV 中没有评测数据"
        )

    return rows


def validate_rows(rows):
    method_counts = Counter(
        row["method"]
        for row in rows
    )

    if set(method_counts) != set(METHODS):
        raise RuntimeError(
            "方法集合错误："
            f"{sorted(method_counts)}"
        )

    method_sample_sets = {}

    for method in METHODS:
        method_rows = [
            row
            for row in rows
            if row["method"] == method
        ]

        sample_paths = [
            row["meta_path"]
            for row in method_rows
        ]

        if (
            len(sample_paths)
            != len(set(sample_paths))
        ):
            raise RuntimeError(
                f"{method} 存在重复样本"
            )

        method_sample_sets[method] = set(
            sample_paths
        )

    reference_samples = (
        method_sample_sets[
            METHODS[0]
        ]
    )

    for method in METHODS[1:]:
        if (
            method_sample_sets[method]
            != reference_samples
        ):
            missing = (
                reference_samples
                - method_sample_sets[method]
            )

            extra = (
                method_sample_sets[method]
                - reference_samples
            )

            raise RuntimeError(
                f"{method} 样本覆盖不一致："
                f"缺少 {len(missing)}，"
                f"多出 {len(extra)}"
            )

    expected_rows = (
        len(reference_samples)
        * len(METHODS)
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            "总行数错误："
            f"期望 {expected_rows}，"
            f"实际 {len(rows)}"
        )

    if {
        row["stride"]
        for row in rows
    } != set(STRIDES):
        raise RuntimeError(
            "stride 集合不是 2、4、8"
        )

    contexts = {
        row["context"]
        for row in rows
    }

    if len(contexts) != 1:
        raise RuntimeError(
            "CSV 中存在多个 context："
            f"{sorted(contexts)}"
        )

    descriptors = {}

    for row in rows:
        descriptor = (
            row["clip"],
            row["motion"],
            row["kind"],
            row["stride"],
            row["context"],
        )

        descriptors.setdefault(
            row["meta_path"],
            set(),
        ).add(descriptor)

    inconsistent = [
        meta_path
        for meta_path, values
        in descriptors.items()
        if len(values) != 1
    ]

    if inconsistent:
        raise RuntimeError(
            "同一样本在不同方法中的元数据不一致："
            f"{inconsistent[:5]}"
        )

    for row in rows:
        for name in (
            "soft_dice",
            "soft_iou",
            "hard_dice",
            "hard_iou",
            "mae",
            "line_distance",
        ):
            value = row[name]

            if not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"{row['clip']} 的 "
                    f"{row['method']}/{name} "
                    f"超出 [0, 1]：{value}"
                )

    return {
        "samples": len(
            reference_samples
        ),
        "rows": len(rows),
        "context": next(
            iter(contexts)
        ),
        "method_counts": (
            method_counts
        ),
    }


def select_rows(
    rows,
    method,
    stride=None,
    kind=None,
):
    return [
        row
        for row in rows
        if row["method"] == method
        and (
            stride is None
            or row["stride"] == stride
        )
        and (
            kind is None
            or row["kind"] == kind
        )
    ]


def mean_metrics(
    rows,
    method,
    stride=None,
    kind=None,
):
    selected = select_rows(
        rows,
        method,
        stride=stride,
        kind=kind,
    )

    if not selected:
        raise RuntimeError(
            "聚合时没有找到数据："
            f"method={method}，"
            f"stride={stride}，"
            f"kind={kind}"
        )

    result = {
        "count": len(selected),
    }

    for name in METRIC_NAMES:
        result[name] = (
            sum(
                row[name]
                for row in selected
            )
            / len(selected)
        )

    return result


def macro_metrics(
    rows,
    method,
):
    by_stride = [
        mean_metrics(
            rows,
            method,
            stride=stride,
        )
        for stride in STRIDES
    ]

    result = {}

    for name in METRIC_NAMES:
        result[name] = (
            sum(
                values[name]
                for values in by_stride
            )
            / len(by_stride)
        )

    return result


def append_header(lines):
    lines.append(
        f"{'方法':<18}"
        f"{'样本':>7}"
        f"{'sDice':>10}"
        f"{'sIoU':>10}"
        f"{'hDice':>10}"
        f"{'hIoU':>10}"
        f"{'MAE':>11}"
        f"{'距离':>10}"
    )

    lines.append("-" * 86)


def append_metric_line(
    lines,
    method,
    values,
    count=None,
):
    if count is None:
        count_text = "-"
    else:
        count_text = str(count)

    lines.append(
        f"{method:<18}"
        f"{count_text:>7}"
        f"{values['soft_dice']:>10.4f}"
        f"{values['soft_iou']:>10.4f}"
        f"{values['hard_dice']:>10.4f}"
        f"{values['hard_iou']:>10.4f}"
        f"{values['mae']:>11.6f}"
        f"{values['line_distance']:>10.4f}"
    )


def build_summary(
    rows,
    validation,
):
    lines = []

    lines.append(
        "火柴人坐标—栅格统一正式评测"
    )
    lines.append("=" * 86)
    lines.append(
        f"唯一验证样本："
        f"{validation['samples']}"
    )
    lines.append(
        f"评测记录："
        f"{validation['rows']}"
    )
    lines.append(
        f"上下文：左右各 "
        f"{validation['context']} 帧"
    )
    lines.append(
        "方法："
        + ", ".join(METHODS)
    )
    lines.append("")
    lines.append(
        "指标方向："
        "Dice/IoU 越高越好；"
        "MAE/距离越低越好。"
    )
    lines.append("")

    lines.append(
        "==== 按 stride 汇总 ===="
    )

    for stride in STRIDES:
        lines.append("")
        lines.append(f"---- s{stride} ----")
        append_header(lines)

        for method in METHODS:
            values = mean_metrics(
                rows,
                method,
                stride=stride,
            )

            append_metric_line(
                lines,
                method,
                values,
                count=values["count"],
            )

    lines.append("")
    lines.append(
        "==== 全部样本自然加权 ===="
    )
    append_header(lines)

    for method in METHODS:
        values = mean_metrics(
            rows,
            method,
        )

        append_metric_line(
            lines,
            method,
            values,
            count=values["count"],
        )

    lines.append("")
    lines.append(
        "==== 三个 stride 宏平均 ===="
    )
    append_header(lines)

    for method in METHODS:
        values = macro_metrics(
            rows,
            method,
        )

        append_metric_line(
            lines,
            method,
            values,
            count=None,
        )

    lines.append("")
    lines.append(
        "==== 按运动类型和 stride ===="
    )

    kinds = sorted(
        {
            row["kind"]
            for row in rows
        }
    )

    for kind in kinds:
        for stride in STRIDES:
            matching = [
                row
                for row in rows
                if row["kind"] == kind
                and row["stride"] == stride
            ]

            if not matching:
                continue

            lines.append("")
            lines.append(
                f"---- {kind} / "
                f"s{stride} ----"
            )
            append_header(lines)

            for method in METHODS:
                values = mean_metrics(
                    rows,
                    method,
                    stride=stride,
                    kind=kind,
                )

                append_metric_line(
                    lines,
                    method,
                    values,
                    count=values["count"],
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

    summary = build_summary(
        rows,
        validation,
    )

    if args.out is None:
        output_path = (
            metrics_path.parent
            / "summary.txt"
        )
    else:
        output_path = Path(
            args.out
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        summary + "\n",
        encoding="utf-8",
    )

    print("数据校验通过")
    print(
        f"唯一验证样本："
        f"{validation['samples']}"
    )
    print(
        f"评测记录："
        f"{validation['rows']}"
    )
    print(
        "方法样本数："
        + ", ".join(
            (
                f"{method}="
                f"{validation['method_counts'][method]}"
            )
            for method in METHODS
        )
    )
    print(f"汇总文件：{output_path}")
    print("")
    print(summary)


if __name__ == "__main__":
    main()