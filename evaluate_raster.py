#!/usr/bin/env python3
"""统一评测火柴人坐标与栅格插帧方法。"""

import argparse
import csv
import datetime
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stickman.models import (
    RasterUNet,
    SequencePredictor,
)
from stickman.raster_dataset import (
    RasterSequenceDataset,
    pose_to_foreground,
)
from stickman.raster_losses import (
    line_distance_loss_per_sample,
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
    "pixel_lerp",
    "pose_lerp",
    "transformer",
    "raster_best",
    "raster_last",
)

METRIC_NAMES = (
    "soft_dice",
    "soft_iou",
    "hard_dice",
    "hard_iou",
    "mae",
    "line_distance",
)

def spatial_sum(tensor):
    return (
        tensor
        .flatten(1)
        .sum(dim=1)
    )


def validate_images(
    prediction,
    target,
):
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction 和 target "
            "形状必须一致："
            f"{tuple(prediction.shape)} "
            f"和 {tuple(target.shape)}"
        )

    if prediction.ndim != 4:
        raise ValueError(
            "图像必须是 "
            "(batch, channel, height, width)"
        )

    if prediction.shape[1] != 1:
        raise ValueError(
            "当前只支持单通道图像"
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise ValueError(
            "prediction 存在 NaN 或无穷值"
        )

    if not torch.isfinite(
        target
    ).all():
        raise ValueError(
            "target 存在 NaN 或无穷值"
        )

    if prediction.min().item() < 0.0:
        raise ValueError(
            "prediction 不能小于 0"
        )

    if prediction.max().item() > 1.0:
        raise ValueError(
            "prediction 不能大于 1"
        )

    if target.min().item() < 0.0:
        raise ValueError(
            "target 不能小于 0"
        )

    if target.max().item() > 1.0:
        raise ValueError(
            "target 不能大于 1"
        )


def dice_and_iou(
    prediction,
    target,
    smooth=1.0,
):
    intersection = spatial_sum(
        prediction * target
    )

    dice_denominator = (
        spatial_sum(prediction)
        + spatial_sum(target)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        dice_denominator + smooth
    )

    union = spatial_sum(
        prediction
        + target
        - prediction * target
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return dice, iou


def compute_image_metrics(
    prediction,
    target,
    threshold,
):
    """返回每个样本的统一图像指标。"""
    validate_images(
        prediction,
        target,
    )

    soft_dice, soft_iou = (
        dice_and_iou(
            prediction,
            target,
        )
    )

    hard_prediction = (
        prediction >= threshold
    ).to(
        dtype=prediction.dtype
    )

    hard_target = (
        target >= threshold
    ).to(
        dtype=target.dtype
    )

    hard_dice, hard_iou = (
        dice_and_iou(
            hard_prediction,
            hard_target,
        )
    )

    mae = (
        torch.abs(
            prediction - target
        )
        .flatten(1)
        .mean(dim=1)
    )

    line_distance = (
        line_distance_loss_per_sample(
            prediction,
            target,
            radii=(1, 2, 4),
        )
    )

    metrics = {
        "soft_dice": soft_dice,
        "soft_iou": soft_iou,
        "hard_dice": hard_dice,
        "hard_iou": hard_iou,
        "mae": mae,
        "line_distance": line_distance,
    }

    for name in METRIC_NAMES:
        values = metrics[name]

        if values.ndim != 1:
            raise RuntimeError(
                f"{name} 必须返回每样本一值，"
                f"实际形状是 {tuple(values.shape)}"
            )

        if not torch.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"{name} 存在 NaN 或无穷值"
            )

    return metrics


def mean_metric_scalars(metrics):
    return {
        name: float(
            metrics[name]
            .mean()
            .item()
        )
        for name in METRIC_NAMES
    }


def print_metric_row(
    name,
    metrics,
):
    values = mean_metric_scalars(
        metrics
    )

    print(
        f"{name:<16}"
        f"soft Dice="
        f"{values['soft_dice']:.4f} | "
        f"soft IoU="
        f"{values['soft_iou']:.4f} | "
        f"hard Dice="
        f"{values['hard_dice']:.4f} | "
        f"hard IoU="
        f"{values['hard_iou']:.4f} | "
        f"MAE="
        f"{values['mae']:.6f} | "
        f"距离="
        f"{values['line_distance']:.4f}"
    )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-base",
        default="data/val_sequences",
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
        "--device",
        choices=("cpu", "cuda"),
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--runs",
        default="runs/raster_eval",
    )

    return parser.parse_args()


def resolve_device(requested):
    if requested is None:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        requested == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "要求使用 CUDA，"
            "但当前环境 CUDA 不可用"
        )

    return torch.device(requested)


def make_roots(
    data_base,
    context,
):
    return [
        (
            Path(data_base)
            / f"{action}_val_001"
            / f"k{context}"
            / f"s{stride}"
        )
        for action in ACTIONS
        for stride in STRIDES
    ]


def load_raster_checkpoint(
    checkpoint_path,
    device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Raster checkpoint 不存在："
            f"{checkpoint_path}"
        )

    saved = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "epoch",
        "model_state",
        "metrics",
        "config",
    }

    missing_keys = (
        required_keys - saved.keys()
    )

    if missing_keys:
        raise RuntimeError(
            f"{checkpoint_path} 缺少字段："
            f"{sorted(missing_keys)}"
        )

    config = saved["config"]

    if config.get("model") != "RasterUNet":
        raise RuntimeError(
            f"{checkpoint_path} 不是 "
            "RasterUNet checkpoint"
        )

    model = RasterUNet(
        in_channels=int(
            config["in_channels"]
        ),
        base_channels=int(
            config["base_channels"]
        ),
        dropout=float(
            config["dropout"]
        ),
    ).to(device)

    model.load_state_dict(
        saved["model_state"]
    )
    model.eval()

    return {
        "path": checkpoint_path,
        "model": model,
        "epoch": int(saved["epoch"]),
        "metrics": saved["metrics"],
        "config": config,
    }


def validate_raster_pair(
    best,
    last,
):
    fields = (
        "context",
        "size",
        "in_channels",
        "base_channels",
        "dropout",
        "strides",
    )

    for field in fields:
        best_value = best["config"][field]
        last_value = last["config"][field]

        if best_value != last_value:
            raise RuntimeError(
                "best 和 last 配置不一致："
                f"{field} 分别为 "
                f"{best_value} 和 {last_value}"
            )

def make_loader(
    dataset,
    batch_size,
    num_workers,
    device,
):
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": (
            device.type == "cuda"
        ),
    }

    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2

    return DataLoader(**options)


def render_transformer_batch(
    sequence_predictor,
    batch,
    size,
    device,
):
    """逐个预测坐标，再渲染成统一前景图。"""
    batch_size = (
        batch["left_poses"].shape[0]
    )

    rendered = []

    for index in range(batch_size):
        prediction_pose = (
            sequence_predictor.predict(
                batch["left_poses"][
                    index
                ].numpy(),
                batch["right_poses"][
                    index
                ].numpy(),
                batch["left_offsets"][
                    index
                ].numpy(),
                batch["right_offsets"][
                    index
                ].numpy(),
            )
        )

        prediction_image = (
            pose_to_foreground(
                prediction_pose,
                size=size,
            )
            .unsqueeze(0)
        )

        rendered.append(
            prediction_image
        )

    return (
        torch.stack(
            rendered,
            dim=0,
        )
        .to(
            device,
            non_blocking=True,
        )
    )


def make_metric_row(
    batch,
    index,
    method,
    metrics,
):
    row = {
        "clip": batch["clip"][index],
        "motion": batch["motion"][index],
        "kind": batch["kind"][index],
        "stride": int(
            batch["stride"][
                index
            ].item()
        ),
        "context": int(
            batch["context"][
                index
            ].item()
        ),
        "method": method,
        "meta_path": (
            batch["meta_path"][index]
        ),
    }

    for name in METRIC_NAMES:
        value = float(
            metrics[name][
                index
            ].item()
        )

        if not math.isfinite(value):
            raise RuntimeError(
                f"{row['clip']} 的 "
                f"{method}/{name} "
                "不是有限值"
            )

        row[name] = value

    return row


def evaluate_all(
    loader,
    dataset_size,
    raster_best,
    raster_last,
    sequence_predictor,
    size,
    threshold,
    device,
):
    rows = []
    processed = 0
    next_report = 100

    best_model = raster_best["model"]
    last_model = raster_last["model"]

    best_model.eval()
    last_model.eval()

    for batch in loader:
        model_input = (
            batch["input"]
            .to(
                device,
                non_blocking=True,
            )
        )

        pixel_lerp = (
            batch["pixel_lerp"]
            .to(
                device,
                non_blocking=True,
            )
        )

        pose_lerp = (
            batch["pose_lerp"]
            .to(
                device,
                non_blocking=True,
            )
        )

        target = (
            batch["target"]
            .to(
                device,
                non_blocking=True,
            )
        )

        with torch.no_grad():
            transformer = (
                render_transformer_batch(
                    sequence_predictor,
                    batch,
                    size,
                    device,
                )
            )

            raster_best_prediction = (
                best_model.predict_image(
                    model_input,
                    pixel_lerp,
                )
            )

            raster_last_prediction = (
                last_model.predict_image(
                    model_input,
                    pixel_lerp,
                )
            )

        predictions = {
            "pixel_lerp": pixel_lerp,
            "pose_lerp": pose_lerp,
            "transformer": transformer,
            "raster_best": (
                raster_best_prediction
            ),
            "raster_last": (
                raster_last_prediction
            ),
        }

        batch_size = target.shape[0]

        for method in METHODS:
            prediction = predictions[
                method
            ]

            metrics = compute_image_metrics(
                prediction,
                target,
                threshold,
            )

            for index in range(
                batch_size
            ):
                rows.append(
                    make_metric_row(
                        batch,
                        index,
                        method,
                        metrics,
                    )
                )

        processed += batch_size

        if (
            processed >= next_report
            or processed == dataset_size
        ):
            print(
                "评测进度："
                f"{processed}/{dataset_size}"
            )

            while (
                next_report <= processed
            ):
                next_report += 100

    expected_rows = (
        dataset_size * len(METHODS)
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            "评测行数错误："
            f"期望 {expected_rows}，"
            f"实际 {len(rows)}"
        )

    return rows


def write_metrics_csv(
    path,
    rows,
):
    fieldnames = [
        "clip",
        "motion",
        "kind",
        "stride",
        "context",
        "method",
        *METRIC_NAMES,
        "meta_path",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_eval_config(
    path,
    args,
    device,
    dataset_size,
    raster_best,
    raster_last,
    sequence_predictor,
):
    config = {
        "data_base": args.data_base,
        "raster_best": str(
            raster_best["path"]
        ),
        "raster_last": str(
            raster_last["path"]
        ),
        "sequence_checkpoint": str(
            sequence_predictor.checkpoint
        ),
        "raster_best_epoch": (
            raster_best["epoch"]
        ),
        "raster_last_epoch": (
            raster_last["epoch"]
        ),
        "sequence_epoch": (
            sequence_predictor.epoch
        ),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "threshold": args.threshold,
        "dataset_size": dataset_size,
        "methods": list(METHODS),
        "metrics": list(METRIC_NAMES),
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size 必须大于 0"
        )

    if args.num_workers < 0:
        raise ValueError(
            "num-workers 不能小于 0"
        )

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "threshold 必须位于 (0, 1)"
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

    raster_config = (
        raster_best["config"]
    )

    context = int(
        raster_config["context"]
    )
    size = int(
        raster_config["size"]
    )
    raster_strides = tuple(
        int(value)
        for value
        in raster_config["strides"]
    )

    if raster_strides != STRIDES:
        raise RuntimeError(
            "Raster checkpoint 的 stride "
            f"是 {raster_strides}，"
            f"评测要求 {STRIDES}"
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
            "Transformer 与 Raster "
            "context 不一致："
            f"{sequence_predictor.context} "
            f"和 {context}"
        )

    if (
        tuple(sequence_predictor.strides)
        != STRIDES
    ):
        raise RuntimeError(
            "Transformer 支持的 stride "
            f"是 {sequence_predictor.strides}，"
            f"评测要求 {STRIDES}"
        )

    roots = make_roots(
        args.data_base,
        context,
    )

    dataset = RasterSequenceDataset(
        roots,
        size=size,
    )

    stride_counts = {
        stride: dataset.strides.count(
            stride
        )
        for stride in STRIDES
    }

    print(f"设备：{device}")
    print(f"图像尺寸：{size} x {size}")
    print(f"上下文：左右各 {context} 帧")
    print(f"验证样本：{len(dataset)}")
    print(f"stride 样本：{stride_counts}")
    print(
        "Transformer epoch："
        f"{sequence_predictor.epoch}"
    )
    print(
        "Raster best epoch："
        f"{raster_best['epoch']}"
    )
    print(
        "Raster last epoch："
        f"{raster_last['epoch']}"
    )
    print(
        "评测方法："
        + ", ".join(METHODS)
    )
    loader = make_loader(
        dataset,
        args.batch_size,
        args.num_workers,
        device,
    )

    timestamp = (
        datetime.datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_dir = (
        Path(args.runs)
        / timestamp
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(f"输出目录：{run_dir}")
    print("")
    print("开始完整评测……")

    rows = evaluate_all(
        loader=loader,
        dataset_size=len(dataset),
        raster_best=raster_best,
        raster_last=raster_last,
        sequence_predictor=(
            sequence_predictor
        ),
        size=size,
        threshold=args.threshold,
        device=device,
    )

    metrics_path = (
        run_dir / "metrics.csv"
    )

    config_path = (
        run_dir / "config.json"
    )

    write_metrics_csv(
        metrics_path,
        rows,
    )

    write_eval_config(
        config_path,
        args,
        device,
        len(dataset),
        raster_best,
        raster_last,
        sequence_predictor,
    )

    print("")
    print("完整评测完成")
    print(f"评测行数：{len(rows)}")
    print(f"指标文件：{metrics_path}")
    print(f"配置文件：{config_path}")


if __name__ == "__main__":
    main()