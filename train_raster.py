#!/usr/bin/env python3
"""训练多帧火柴人栅格插帧 UNet。"""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)

from stickman.models import RasterUNet
from stickman.raster_dataset import (
    RasterSequenceDataset,
)
from stickman.raster_losses import (
    raster_loss,
)


ACTIONS = (
    "walk",
    "wave",
    "swing",
    "squat",
    "lean",
)

STRIDES = (2, 4, 8)

METRIC_NAMES = (
    "total",
    "bce",
    "dice_loss",
    "distance",
    "soft_dice",
    "soft_iou",
    "mae",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--context",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-positive-weight",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-base",
        default="data/formal_sequences",
    )
    parser.add_argument(
        "--val-base",
        default="data/val_sequences",
    )
    parser.add_argument(
        "--runs",
        default="runs",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_roots(
    base,
    split,
    context,
):
    return [
        (
            Path(base)
            / f"{action}_{split}_001"
            / f"k{context}"
            / f"s{stride}"
        )
        for action in ACTIONS
        for stride in STRIDES
    ]


def make_train_loader(
    dataset,
    batch_size,
    num_workers,
    device,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=(
            dataset.stride_sampling_weights()
        ),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": (
            device.type == "cuda"
        ),
    }

    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2

    return DataLoader(**options)


def make_val_loader(
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


def spatial_sum(tensor):
    return tensor.flatten(1).sum(dim=1)


def compute_metrics(
    logits,
    target,
    loss_parts,
):
    probabilities = torch.sigmoid(
        logits
    )

    intersection = spatial_sum(
        probabilities * target
    )

    union = spatial_sum(
        probabilities
        + target
        - probabilities * target
    )

    soft_iou = (
        intersection + 1.0
    ) / (
        union + 1.0
    )

    mae = (
        torch.abs(
            probabilities - target
        )
        .flatten(1)
        .mean(dim=1)
    )

    return {
        "total": loss_parts["total"],
        "bce": loss_parts["bce"],
        "dice_loss": loss_parts["dice"],
        "distance": (
            loss_parts["distance"]
        ),
        "soft_dice": (
            1.0 - loss_parts["dice"]
        ),
        "soft_iou": soft_iou,
        "mae": mae,
    }


def empty_accumulator():
    return {
        "count": 0,
        **{
            name: 0.0
            for name in METRIC_NAMES
        },
    }


def update_accumulator(
    accumulator,
    metrics,
    mask=None,
):
    if mask is None:
        count = metrics[
            "total"
        ].shape[0]
    else:
        count = int(
            mask.sum().item()
        )

    if count == 0:
        return

    accumulator["count"] += count

    for name in METRIC_NAMES:
        values = metrics[name]

        if mask is not None:
            values = values[mask]

        accumulator[name] += (
            values.detach().sum().item()
        )


def finalize_accumulator(
    accumulator,
):
    count = accumulator["count"]

    if count == 0:
        raise RuntimeError(
            "某个 stride 没有样本"
        )

    return {
        name: (
            accumulator[name] / count
        )
        for name in METRIC_NAMES
    }


def run_epoch(
    model,
    loader,
    device,
    args,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    overall_accumulator = (
        empty_accumulator()
    )

    stride_accumulators = {
        stride: empty_accumulator()
        for stride in STRIDES
    }

    for batch in loader:
        model_input = batch["input"].to(
            device,
            non_blocking=True,
        )
        pixel_lerp = batch[
            "pixel_lerp"
        ].to(
            device,
            non_blocking=True,
        )
        target = batch["target"].to(
            device,
            non_blocking=True,
        )
        strides = batch["stride"].to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):
            logits = model.predict_logits(
                model_input,
                pixel_lerp,
            )

            loss_parts = raster_loss(
                logits,
                target,
                bce_weight=(
                    args.bce_weight
                ),
                dice_weight=(
                    args.dice_weight
                ),
                distance_weight=(
                    args.distance_weight
                ),
                max_positive_weight=(
                    args.max_positive_weight
                ),
                distance_radii=(
                    1,
                    2,
                    4,
                ),
                reduction="none",
            )

            batch_loss = (
                loss_parts["total"]
                .mean()
            )

            if training:
                batch_loss.backward()

                clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        with torch.no_grad():
            metrics = compute_metrics(
                logits,
                target,
                loss_parts,
            )

        update_accumulator(
            overall_accumulator,
            metrics,
        )

        for stride in STRIDES:
            mask = strides == stride

            update_accumulator(
                stride_accumulators[
                    stride
                ],
                metrics,
                mask=mask,
            )

    overall = finalize_accumulator(
        overall_accumulator
    )

    by_stride = {
        stride: finalize_accumulator(
            stride_accumulators[stride]
        )
        for stride in STRIDES
    }

    macro_total = sum(
        by_stride[stride]["total"]
        for stride in STRIDES
    ) / len(STRIDES)

    return {
        "overall": overall,
        "by_stride": by_stride,
        "macro_total": macro_total,
    }


def flatten_metrics(
    prefix,
    metrics,
):
    result = {
        f"{prefix}_macro_total": (
            metrics["macro_total"]
        )
    }

    for name, value in (
        metrics["overall"].items()
    ):
        result[
            f"{prefix}_overall_{name}"
        ] = value

    for stride in STRIDES:
        for name, value in (
            metrics["by_stride"][
                stride
            ].items()
        ):
            result[
                f"{prefix}_s{stride}_{name}"
            ] = value

    return result


def write_history(path, history):
    if not history:
        return

    fieldnames = list(
        history[0].keys()
    )

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
        writer.writerows(history)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    config,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": (
                model.state_dict()
            ),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def print_initial(metrics):
    print(
        "初始验证（近似 pixel lerp）："
    )

    for stride in STRIDES:
        values = metrics[
            "by_stride"
        ][stride]

        print(
            f"  s{stride} | "
            f"loss="
            f"{values['total']:.4f} | "
            f"Dice="
            f"{values['soft_dice']:.4f} | "
            f"IoU="
            f"{values['soft_iou']:.4f} | "
            f"MAE="
            f"{values['mae']:.4f}"
        )

    print(
        "  stride 宏平均 loss："
        f"{metrics['macro_total']:.4f}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_roots = make_roots(
        args.train_base,
        "train",
        args.context,
    )
    val_roots = make_roots(
        args.val_base,
        "val",
        args.context,
    )

    train_dataset = (
        RasterSequenceDataset(
            train_roots,
            size=args.size,
        )
    )
    val_dataset = (
        RasterSequenceDataset(
            val_roots,
            size=args.size,
        )
    )

    train_loader = make_train_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        device,
        args.seed,
    )
    val_loader = make_val_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )

    model = RasterUNet(
        in_channels=args.context * 4,
        base_channels=(
            args.base_channels
        ),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        Path(args.runs)
        / f"raster_unet_k{args.context}"
        / timestamp
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    config = {
        **vars(args),
        "device": str(device),
        "model": "RasterUNet",
        "in_channels": (
            args.context * 4
        ),
        "train_samples": (
            len(train_dataset)
        ),
        "val_samples": (
            len(val_dataset)
        ),
        "strides": list(STRIDES),
        "distance_radii": [
            1,
            2,
            4,
        ],
        "train_roots": [
            str(path)
            for path in train_roots
        ],
        "val_roots": [
            str(path)
            for path in val_roots
        ],
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

    print(f"设备：{device}")
    print(
        f"上下文：左右各 {args.context} 帧"
    )
    print(
        f"图像尺寸：{args.size} x "
        f"{args.size}"
    )
    print(
        f"训练样本：{len(train_dataset)}"
    )
    print(
        f"验证样本：{len(val_dataset)}"
    )
    print(
        "训练采样：s2、s4、s8 等概率"
    )
    print(f"输出目录：{run_dir}")

    initial_val = run_epoch(
        model,
        val_loader,
        device,
        args,
    )

    print_initial(initial_val)

    best_val = initial_val[
        "macro_total"
    ]
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    save_checkpoint(
        run_dir / "best.pt",
        model,
        optimizer,
        epoch=0,
        metrics=initial_val,
        config=config,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args,
        )

        scheduler.step(
            val_metrics["macro_total"]
        )

        learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "learning_rate": (
                learning_rate
            ),
            **flatten_metrics(
                "train",
                train_metrics,
            ),
            **flatten_metrics(
                "val",
                val_metrics,
            ),
        }

        history.append(row)

        write_history(
            run_dir / "history.csv",
            history,
        )

        improved = (
            val_metrics["macro_total"]
            < best_val - 1e-7
        )

        if improved:
            best_val = (
                val_metrics["macro_total"]
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            config,
        )

        if (
            epoch == 1
            or epoch % args.print_every == 0
            or epoch == args.epochs
        ):
            s2 = val_metrics[
                "by_stride"
            ][2]
            s4 = val_metrics[
                "by_stride"
            ][4]
            s8 = val_metrics[
                "by_stride"
            ][8]

            print(
                f"epoch {epoch:03d} | "
                f"train="
                f"{train_metrics['macro_total']:.4f} | "
                f"val="
                f"{val_metrics['macro_total']:.4f} | "
                f"Dice "
                f"s2={s2['soft_dice']:.4f} "
                f"s4={s4['soft_dice']:.4f} "
                f"s8={s8['soft_dice']:.4f} | "
                f"lr={learning_rate:.2e}"
            )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "提前停止：验证宏平均损失连续 "
                f"{args.patience} 个 epoch "
                "没有改善"
            )
            break

    print(f"最佳 epoch：{best_epoch}")
    print(
        "最佳验证宏平均损失："
        f"{best_val:.6f}"
    )
    print(
        f"最佳模型：{run_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()