#!/usr/bin/env python3
"""可视化 Raster 正式评测选择案例。"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from evaluate_raster import (
    make_roots,
    resolve_device,
    load_raster_checkpoint,
    validate_raster_pair,
)
from stickman.models import (
    SequencePredictor,
)
from stickman.raster_dataset import (
    RasterSequenceDataset,
    pose_to_foreground,
)
from visualize_raster_checkpoint import (
    add_label,
    make_overlay,
    mask_to_image,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "selection",
        help="select_raster_cases.py 生成的 selection.csv",
    )
    parser.add_argument(
        "--raster-best",
        required=True,
    )
    parser.add_argument(
        "--raster-last",
        required=True,
    )
    parser.add_argument(
        "--sequence-checkpoint",
        required=True,
    )
    parser.add_argument(
        "--data-base",
        default="data/val_sequences",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        default=None,
    )

    return parser.parse_args()


def load_selection(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"selection.csv 不存在：{path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise RuntimeError(
            "selection.csv 中没有案例"
        )

    required = {
        "selection_type",
        "stride",
        "kind",
        "clip",
        "meta_path",
        "transformer_hard_dice",
        "raster_best_hard_dice",
        "raster_last_hard_dice",
        "raster_best_soft_dice",
        "raster_last_soft_dice",
    }

    missing = (
        required - set(rows[0])
    )

    if missing:
        raise RuntimeError(
            "selection.csv 缺少字段："
            f"{sorted(missing)}"
        )

    return rows


def make_index(dataset):
    result = {}

    for index, record in enumerate(
        dataset.records
    ):
        meta_path, _ = record

        key = str(
            meta_path.resolve()
        )

        if key in result:
            raise RuntimeError(
                f"重复 meta_path：{key}"
            )

        result[key] = index

    return result


def short_reason(row):
    names = {
        "transformer_over_raster": (
            "T>R"
        ),
        "raster_over_transformer": (
            "R>T"
        ),
        "last_over_best": (
            "last>best"
        ),
        "best_over_last": (
            "best>last"
        ),
        "raster_last_worst": (
            "last worst"
        ),
    }

    selection_type = row[
        "selection_type"
    ]

    label = names.get(
        selection_type,
        selection_type,
    )

    delta_text = row.get(
        "delta_hard_dice",
        "",
    )

    if delta_text:
        label += (
            f" {float(delta_text):+.3f}"
        )

    return (
        f"{label} | {row['kind']}"
    )


def render_case(
    row,
    dataset,
    index_by_meta,
    raster_best,
    raster_last,
    sequence_predictor,
    size,
    device,
):
    meta_key = str(
        Path(
            row["meta_path"]
        ).resolve()
    )

    if meta_key not in index_by_meta:
        raise RuntimeError(
            "验证 Dataset 中找不到："
            f"{row['meta_path']}"
        )

    sample = dataset[
        index_by_meta[meta_key]
    ]

    if sample["clip"] != row["clip"]:
        raise RuntimeError(
            "selection 与 Dataset "
            "的 clip 不一致："
            f"{row['clip']} 和 "
            f"{sample['clip']}"
        )

    model_input = (
        sample["input"]
        .unsqueeze(0)
        .to(device)
    )

    pixel_lerp = (
        sample["pixel_lerp"]
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        raster_best_prediction = (
            raster_best["model"]
            .predict_image(
                model_input,
                pixel_lerp,
            )[0]
        )

        raster_last_prediction = (
            raster_last["model"]
            .predict_image(
                model_input,
                pixel_lerp,
            )[0]
        )

    transformer_pose = (
        sequence_predictor.predict(
            sample["left_poses"].numpy(),
            sample["right_poses"].numpy(),
            sample["left_offsets"].numpy(),
            sample["right_offsets"].numpy(),
        )
    )

    transformer_image = (
        pose_to_foreground(
            transformer_pose,
            size=size,
        )
        .unsqueeze(0)
    )

    best_binary = (
        raster_best_prediction
        >= 0.5
    ).float()

    last_binary = (
        raster_last_prediction
        >= 0.5
    ).float()

    return {
        "sample": sample,
        "transformer": (
            transformer_image.cpu()
        ),
        "raster_best": (
            raster_best_prediction
            .detach()
            .cpu()
        ),
        "raster_last": (
            raster_last_prediction
            .detach()
            .cpu()
        ),
        "best_binary": (
            best_binary.detach().cpu()
        ),
        "last_binary": (
            last_binary.detach().cpu()
        ),
    }

def row_metric(
    row,
    method,
    metric,
):
    return float(
        row[
            f"{method}_{metric}"
        ]
    )


def make_route_cells(
    row,
    rendered,
):
    sample = rendered["sample"]
    target = sample["target"]

    return [
        add_label(
            mask_to_image(
                sample["prev_image"]
            ),
            short_reason(row),
        ),
        add_label(
            mask_to_image(
                sample["next_image"]
            ),
            "next",
        ),
        add_label(
            mask_to_image(
                sample["pixel_lerp"]
            ),
            (
                "pixel "
                f"hD={row_metric(row, 'pixel_lerp', 'hard_dice'):.3f}"
            ),
        ),
        add_label(
            mask_to_image(
                sample["pose_lerp"]
            ),
            (
                "pose "
                f"hD={row_metric(row, 'pose_lerp', 'hard_dice'):.3f}"
            ),
        ),
        add_label(
            mask_to_image(
                rendered["transformer"]
            ),
            (
                "Transformer "
                f"hD={row_metric(row, 'transformer', 'hard_dice'):.3f}"
            ),
        ),
        add_label(
            mask_to_image(
                rendered["raster_last"]
            ),
            (
                "Raster probability "
                f"sD={row_metric(row, 'raster_last', 'soft_dice'):.3f}"
            ),
        ),
        add_label(
            mask_to_image(
                rendered["last_binary"]
            ),
            (
                "Raster threshold "
                f"hD={row_metric(row, 'raster_last', 'hard_dice'):.3f}"
            ),
        ),
        add_label(
            make_overlay(
                rendered["transformer"],
                target,
            ),
            "Transformer overlay",
        ),
        add_label(
            make_overlay(
                rendered["raster_last"],
                target,
            ),
            "Raster overlay",
        ),
        add_label(
            mask_to_image(target),
            "target",
        ),
    ]


def make_checkpoint_cells(
    row,
    rendered,
):
    sample = rendered["sample"]
    target = sample["target"]

    return [
        add_label(
            mask_to_image(
                sample["prev_image"]
            ),
            short_reason(row),
        ),
        add_label(
            mask_to_image(
                sample["next_image"]
            ),
            "next",
        ),
        add_label(
            mask_to_image(
                rendered["raster_best"]
            ),
            (
                "best probability "
                f"sD={row_metric(row, 'raster_best', 'soft_dice'):.3f}"
            ),
        ),
        add_label(
            mask_to_image(
                rendered["best_binary"]
            ),
            (
                "best threshold "
                f"hD={row_metric(row, 'raster_best', 'hard_dice'):.3f}"
            ),
        ),
        add_label(
            mask_to_image(
                rendered["raster_last"]
            ),
            (
                "last probability "
                f"sD={row_metric(row, 'raster_last', 'soft_dice'):.3f}"
            ),
        ),
        add_label(
            mask_to_image(
                rendered["last_binary"]
            ),
            (
                "last threshold "
                f"hD={row_metric(row, 'raster_last', 'hard_dice'):.3f}"
            ),
        ),
        add_label(
            make_overlay(
                rendered["raster_best"],
                target,
            ),
            "best overlay",
        ),
        add_label(
            make_overlay(
                rendered["raster_last"],
                target,
            ),
            "last overlay",
        ),
        add_label(
            mask_to_image(target),
            "target",
        ),
    ]


def save_grid(
    rows,
    size,
    output_path,
):
    if not rows:
        raise RuntimeError(
            "网格中没有任何行"
        )

    columns = len(rows[0])

    for cells in rows:
        if len(cells) != columns:
            raise RuntimeError(
                "网格列数不一致"
            )

    sheet = Image.new(
        "RGB",
        (
            columns * size,
            len(rows) * size,
        ),
        (255, 255, 255),
    )

    for row_index, cells in enumerate(
        rows
    ):
        for column, image in enumerate(
            cells
        ):
            sheet.paste(
                image,
                (
                    column * size,
                    row_index * size,
                ),
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(output_path)

def main():
    args = parse_args()

    selection_path = Path(
        args.selection
    )

    rows = load_selection(
        selection_path
    )

    device = resolve_device(
        args.device
    )

    raster_best = (
        load_raster_checkpoint(
            args.raster_best,
            device,
        )
    )

    raster_last = (
        load_raster_checkpoint(
            args.raster_last,
            device,
        )
    )

    validate_raster_pair(
        raster_best,
        raster_last,
    )

    config = raster_best["config"]

    context = int(
        config["context"]
    )
    size = int(
        config["size"]
    )

    sequence_predictor = (
        SequencePredictor(
            args.sequence_checkpoint,
            device=str(device),
        )
    )

    if (
        sequence_predictor.context
        != context
    ):
        raise RuntimeError(
            "Transformer 和 Raster "
            "context 不一致"
        )

    roots = make_roots(
        args.data_base,
        context,
    )

    dataset = RasterSequenceDataset(
        roots,
        size=size,
    )

    index_by_meta = make_index(
        dataset
    )

    if args.out_dir is None:
        output_dir = (
            selection_path.parent
            / "selection_grids"
        )
    else:
        output_dir = Path(
            args.out_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = {}
    output_paths = []

    for stride in (2, 4, 8):
        stride_rows = [
            row
            for row in rows
            if int(row["stride"]) == stride
        ]

        if len(stride_rows) != 9:
            raise RuntimeError(
                f"s{stride} 应有 9 条选择记录，"
                f"实际是 {len(stride_rows)}"
            )

        route_rows = []
        checkpoint_rows = []

        for row in stride_rows:
            meta_path = row[
                "meta_path"
            ]

            if meta_path not in cache:
                cache[meta_path] = (
                    render_case(
                        row=row,
                        dataset=dataset,
                        index_by_meta=(
                            index_by_meta
                        ),
                        raster_best=(
                            raster_best
                        ),
                        raster_last=(
                            raster_last
                        ),
                        sequence_predictor=(
                            sequence_predictor
                        ),
                        size=size,
                        device=device,
                    )
                )

            rendered = cache[
                meta_path
            ]

            route_rows.append(
                make_route_cells(
                    row,
                    rendered,
                )
            )

            checkpoint_rows.append(
                make_checkpoint_cells(
                    row,
                    rendered,
                )
            )

        route_path = (
            output_dir
            / f"route_comparison_s{stride}.png"
        )

        checkpoint_path = (
            output_dir
            / (
                "checkpoint_comparison_"
                f"s{stride}.png"
            )
        )

        save_grid(
            route_rows,
            size,
            route_path,
        )

        save_grid(
            checkpoint_rows,
            size,
            checkpoint_path,
        )

        output_paths.extend(
            (
                route_path,
                checkpoint_path,
            )
        )

    print(f"设备：{device}")
    print(
        f"selection 记录：{len(rows)}"
    )
    print(
        f"唯一渲染案例：{len(cache)}"
    )
    print(f"输出目录：{output_dir}")

    for output_path in output_paths:
        print(f"生成：{output_path}")


if __name__ == "__main__":
    main()