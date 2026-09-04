#!/usr/bin/env python3
"""统一评测 LERP、MLP 和多帧 Transformer。"""

import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np

from stickman.models import (
    MLPPredictor,
    SequencePredictor,
)
from stickman.skeleton import (
    bone_length_error,
    mpjpe,
)


ACTIONS = (
    "walk",
    "wave",
    "swing",
    "squat",
    "lean",
)

STRIDES = (2, 4, 8)

METHODS = (
    "lerp",
    "mlp",
    "transformer",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-base",
        default="data/val_sequences",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--sequence-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--mlp-s2",
        required=True,
    )
    parser.add_argument(
        "--mlp-s4",
        required=True,
    )
    parser.add_argument(
        "--mlp-s8",
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
    )
    parser.add_argument(
        "--runs",
        default="runs/sequence_eval",
    )

    return parser.parse_args()


def load_sequences(
    base,
    context,
):
    clips = []

    for action in ACTIONS:
        for stride in STRIDES:
            root = (
                Path(base)
                / f"{action}_val_001"
                / f"k{context}"
                / f"s{stride}"
            )

            if not root.is_dir():
                raise FileNotFoundError(
                    f"序列目录不存在：{root}"
                )

            meta_paths = sorted(
                root.glob("*/meta.json")
            )

            if not meta_paths:
                raise RuntimeError(
                    f"{root} 中没有 meta.json"
                )

            for meta_path in meta_paths:
                with meta_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    meta = json.load(file)

                if int(meta["stride"]) != stride:
                    raise RuntimeError(
                        f"{meta_path} 的 stride 错误"
                    )

                if int(meta["context"]) != context:
                    raise RuntimeError(
                        f"{meta_path} 的 context 错误"
                    )

                left = np.asarray(
                    meta["poses"]["left"],
                    dtype=np.float32,
                )
                target = np.asarray(
                    meta["poses"]["target"],
                    dtype=np.float32,
                )
                right = np.asarray(
                    meta["poses"]["right"],
                    dtype=np.float32,
                )

                expected_context_shape = (
                    context,
                    17,
                    2,
                )

                if (
                    left.shape
                    != expected_context_shape
                ):
                    raise RuntimeError(
                        f"{meta_path} 的 left "
                        f"形状错误：{left.shape}"
                    )

                if (
                    right.shape
                    != expected_context_shape
                ):
                    raise RuntimeError(
                        f"{meta_path} 的 right "
                        f"形状错误：{right.shape}"
                    )

                if target.shape != (17, 2):
                    raise RuntimeError(
                        f"{meta_path} 的 target "
                        f"形状错误：{target.shape}"
                    )

                if not np.isfinite(left).all():
                    raise RuntimeError(
                        f"{meta_path} 的 left "
                        "存在非有限值"
                    )

                if not np.isfinite(right).all():
                    raise RuntimeError(
                        f"{meta_path} 的 right "
                        "存在非有限值"
                    )

                if not np.isfinite(target).all():
                    raise RuntimeError(
                        f"{meta_path} 的 target "
                        "存在非有限值"
                    )

                clips.append(
                    {
                        "clip": (
                            meta_path.parent.name
                        ),
                        "action": action,
                        "motion": meta["motion"],
                        "kind": meta["kind"],
                        "stride": stride,
                        "size": int(meta["size"]),
                        "left": left,
                        "target": target,
                        "right": right,
                        "left_offsets": (
                            meta[
                                "time_offsets"
                            ]["left"]
                        ),
                        "right_offsets": (
                            meta[
                                "time_offsets"
                            ]["right"]
                        ),
                    }
                )

    return clips


def mean_metrics(
    rows,
    method,
    stride=None,
    kind=None,
):
    selected = [
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

    if not selected:
        raise RuntimeError(
            "聚合时没有找到匹配样本："
            f"method={method}，"
            f"stride={stride}，"
            f"kind={kind}"
        )

    return {
        "count": len(selected),
        "mpjpe_px": float(
            np.mean(
                [
                    row["mpjpe_px"]
                    for row in selected
                ]
            )
        ),
        "bone_err_px": float(
            np.mean(
                [
                    row["bone_err_px"]
                    for row in selected
                ]
            )
        ),
    }


def improvement_percent(
    baseline,
    value,
):
    if baseline == 0:
        return float("nan")

    return (
        (baseline - value)
        / baseline
        * 100.0
    )


def build_summary(rows):
    lines = []

    lines.append("==== 按 stride 汇总 ====")
    lines.append(
        f"{'stride':<8}"
        f"{'方法':<16}"
        f"{'样本':>8}"
        f"{'MPJPE(px)':>14}"
        f"{'骨长(px)':>14}"
        f"{'MP改善':>12}"
        f"{'骨长改善':>12}"
    )
    lines.append("-" * 84)

    for stride in STRIDES:
        lerp_values = mean_metrics(
            rows,
            "lerp",
            stride=stride,
        )

        for method in METHODS:
            values = mean_metrics(
                rows,
                method,
                stride=stride,
            )

            mp_improvement = (
                improvement_percent(
                    lerp_values["mpjpe_px"],
                    values["mpjpe_px"],
                )
            )
            bone_improvement = (
                improvement_percent(
                    lerp_values["bone_err_px"],
                    values["bone_err_px"],
                )
            )

            lines.append(
                f"s{stride:<7}"
                f"{method:<16}"
                f"{values['count']:>8}"
                f"{values['mpjpe_px']:>14.4f}"
                f"{values['bone_err_px']:>14.4f}"
                f"{mp_improvement:>11.2f}%"
                f"{bone_improvement:>11.2f}%"
            )

        lines.append("")

    lines.append("==== 全部样本自然加权 ====")

    overall = {
        method: mean_metrics(
            rows,
            method,
        )
        for method in METHODS
    }

    overall_lerp = overall["lerp"]

    for method in METHODS:
        values = overall[method]

        mp_improvement = improvement_percent(
            overall_lerp["mpjpe_px"],
            values["mpjpe_px"],
        )
        bone_improvement = (
            improvement_percent(
                overall_lerp["bone_err_px"],
                values["bone_err_px"],
            )
        )

        lines.append(
            f"{method:<16}"
            f"n={values['count']:<6} "
            f"MPJPE={values['mpjpe_px']:.4f}px "
            f"骨长={values['bone_err_px']:.4f}px "
            f"MP改善={mp_improvement:.2f}% "
            f"骨长改善={bone_improvement:.2f}%"
        )

    lines.append("")
    lines.append("==== 三个 stride 宏平均 ====")

    macro = {}

    for method in METHODS:
        stride_values = [
            mean_metrics(
                rows,
                method,
                stride=stride,
            )
            for stride in STRIDES
        ]

        macro[method] = {
            "mpjpe_px": float(
                np.mean(
                    [
                        value["mpjpe_px"]
                        for value
                        in stride_values
                    ]
                )
            ),
            "bone_err_px": float(
                np.mean(
                    [
                        value["bone_err_px"]
                        for value
                        in stride_values
                    ]
                )
            ),
        }

    macro_lerp = macro["lerp"]

    for method in METHODS:
        values = macro[method]

        mp_improvement = improvement_percent(
            macro_lerp["mpjpe_px"],
            values["mpjpe_px"],
        )
        bone_improvement = (
            improvement_percent(
                macro_lerp["bone_err_px"],
                values["bone_err_px"],
            )
        )

        lines.append(
            f"{method:<16}"
            f"MPJPE={values['mpjpe_px']:.4f}px "
            f"骨长={values['bone_err_px']:.4f}px "
            f"MP改善={mp_improvement:.2f}% "
            f"骨长改善={bone_improvement:.2f}%"
        )

    lines.append("")
    lines.append("==== 按运动类型和 stride ====")
    lines.append(
        f"{'运动类型':<22}"
        f"{'stride':<8}"
        f"{'方法':<16}"
        f"{'样本':>8}"
        f"{'MPJPE(px)':>14}"
        f"{'骨长(px)':>14}"
    )
    lines.append("-" * 82)

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

            for method in METHODS:
                values = mean_metrics(
                    rows,
                    method,
                    stride=stride,
                    kind=kind,
                )

                lines.append(
                    f"{kind:<22}"
                    f"s{stride:<7}"
                    f"{method:<16}"
                    f"{values['count']:>8}"
                    f"{values['mpjpe_px']:>14.4f}"
                    f"{values['bone_err_px']:>14.4f}"
                )

            lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()

    sequence_predictor = (
        SequencePredictor(
            args.sequence_checkpoint,
            device=args.device,
        )
    )

    if (
        sequence_predictor.context
        != args.context
    ):
        raise SystemExit(
            "Transformer context 与数据不一致："
            f"模型是 {sequence_predictor.context}，"
            f"数据是 {args.context}"
        )

    mlp_checkpoints = {
        2: args.mlp_s2,
        4: args.mlp_s4,
        8: args.mlp_s8,
    }

    mlp_predictors = {
        stride: MLPPredictor(
            checkpoint,
            device=args.device,
        )
        for stride, checkpoint
        in mlp_checkpoints.items()
    }

    for stride, predictor in (
        mlp_predictors.items()
    ):
        if predictor.stride != stride:
            raise SystemExit(
                f"s{stride} MLP checkpoint "
                f"实际是 s{predictor.stride}"
            )

    clips = load_sequences(
        args.data_base,
        args.context,
    )

    timestamp = (
        datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_dir = (
        Path(args.runs) / timestamp
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"验证样本：{len(clips)}"
    )
    print(
        "Transformer epoch："
        f"{sequence_predictor.epoch}"
    )
    print(
        "MLP epoch："
        + ", ".join(
            (
                f"s{stride}="
                f"{mlp_predictors[stride].epoch}"
            )
            for stride in STRIDES
        )
    )
    print(
        f"设备：{sequence_predictor.device}"
    )
    print(
        f"输出目录：{run_dir}"
    )

    metric_rows = []

    for index, clip in enumerate(
        clips,
        start=1,
    ):
        left = clip["left"]
        right = clip["right"]
        target = clip["target"]
        stride = clip["stride"]
        size = clip["size"]

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
                    clip["left_offsets"],
                    clip["right_offsets"],
                )
            ),
        }

        for method in METHODS:
            prediction = np.asarray(
                predictions[method],
                dtype=np.float32,
            )

            if prediction.shape != (17, 2):
                raise RuntimeError(
                    f"{clip['clip']} 的 "
                    f"{method} 输出形状错误："
                    f"{prediction.shape}"
                )

            if not np.isfinite(
                prediction
            ).all():
                raise RuntimeError(
                    f"{clip['clip']} 的 "
                    f"{method} 输出非有限值"
                )

            metric_rows.append(
                {
                    "clip": clip["clip"],
                    "action": clip["action"],
                    "motion": clip["motion"],
                    "kind": clip["kind"],
                    "stride": stride,
                    "method": method,
                    "mpjpe_px": mpjpe(
                        prediction,
                        target,
                        size,
                    ),
                    "bone_err_px": (
                        bone_length_error(
                            prediction,
                            target,
                            size,
                        )
                    ),
                }
            )

        if (
            index % 100 == 0
            or index == len(clips)
        ):
            print(
                f"已评测：{index}/"
                f"{len(clips)}"
            )

    metrics_path = (
        run_dir / "metrics.csv"
    )

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "clip",
            "action",
            "motion",
            "kind",
            "stride",
            "method",
            "mpjpe_px",
            "bone_err_px",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = build_summary(
        metric_rows
    )

    (
        run_dir / "summary.txt"
    ).write_text(
        summary,
        encoding="utf-8",
    )

    config = {
        "data_base": args.data_base,
        "context": args.context,
        "samples": len(clips),
        "methods": list(METHODS),
        "sequence_checkpoint": (
            args.sequence_checkpoint
        ),
        "sequence_epoch": (
            sequence_predictor.epoch
        ),
        "mlp_checkpoints": (
            mlp_checkpoints
        ),
        "mlp_epochs": {
            stride: (
                mlp_predictors[
                    stride
                ].epoch
            )
            for stride in STRIDES
        },
        "device": str(
            sequence_predictor.device
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

    print()
    print(summary)
    print()
    print(f"→ {run_dir}")


if __name__ == "__main__":
    main()