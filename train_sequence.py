#!/usr/bin/env python3
"""训练多帧上下文姿态插帧 Transformer。"""

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

from stickman.losses import limb_lengths
from stickman.models import SequenceTransformer
from stickman.sequence_dataset import (
    SequencePoseDataset,
)
from stickman.skeleton import CANVAS


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
    "position",
    "bone",
    "mpjpe_px",
    "bone_px",
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
        default=64,
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
        "--bone-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--context",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def make_val_loader(
    dataset,
    batch_size,
    device,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def per_sample_metrics(
    prediction,
    target,
    scale,
    bone_weight,
):
    joint_error = torch.linalg.vector_norm(
        prediction - target,
        dim=-1,
    )

    position = joint_error.mean(dim=-1)

    prediction_bones = limb_lengths(
        prediction
    )
    target_bones = limb_lengths(
        target
    )

    bone_difference = torch.abs(
        prediction_bones - target_bones
    )

    bone = bone_difference.mean(dim=-1)

    total = position + bone_weight * bone

    mpjpe_px = (
        joint_error
        * scale[:, None]
        * CANVAS
    ).mean(dim=-1)

    bone_px = (
        bone_difference
        * scale[:, None]
        * CANVAS
    ).mean(dim=-1)

    return {
        "total": total,
        "position": position,
        "bone": bone,
        "mpjpe_px": mpjpe_px,
        "bone_px": bone_px,
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
        count = metrics["total"].shape[0]
    else:
        count = int(mask.sum().item())

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


def finalize_accumulator(accumulator):
    count = accumulator["count"]

    if count == 0:
        raise RuntimeError(
            "某个 stride 没有验证样本"
        )

    return {
        name: accumulator[name] / count
        for name in METRIC_NAMES
    }


def run_epoch(
    model,
    loader,
    device,
    bone_weight,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    overall_accumulator = empty_accumulator()

    stride_accumulators = {
        stride: empty_accumulator()
        for stride in STRIDES
    }

    for batch in loader:
        tokens = batch["tokens"].to(
            device,
            non_blocking=True,
        )
        time_features = batch["time"].to(
            device,
            non_blocking=True,
        )
        lerp_pose = batch["lerp"].to(
            device,
            non_blocking=True,
        )
        gt_pose = batch["gt"].to(
            device,
            non_blocking=True,
        )
        scale = batch["scale"].to(
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

        with torch.set_grad_enabled(training):
            prediction = model.predict_pose(
                tokens,
                time_features,
                lerp_pose,
            )

            metrics = per_sample_metrics(
                prediction,
                gt_pose,
                scale,
                bone_weight,
            )

            loss = metrics["total"].mean()

            if training:
                loss.backward()

                clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        update_accumulator(
            overall_accumulator,
            metrics,
        )

        for stride in STRIDES:
            mask = strides == stride

            update_accumulator(
                stride_accumulators[stride],
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

    # 三个 stride 各占三分之一。
    macro_total = sum(
        by_stride[stride]["total"]
        for stride in STRIDES
    ) / len(STRIDES)

    return {
        "overall": overall,
        "by_stride": by_stride,
        "macro_total": macro_total,
    }


def flatten_metrics(prefix, metrics):
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


def print_initial_metrics(metrics):
    print("初始验证（模型严格等于 LERP）：")

    for stride in STRIDES:
        values = metrics[
            "by_stride"
        ][stride]

        print(
            f"  s{stride} | "
            f"MPJPE="
            f"{values['mpjpe_px']:.4f}px | "
            f"骨长="
            f"{values['bone_px']:.4f}px | "
            f"loss="
            f"{values['total']:.6f}"
        )

    print(
        "  stride 宏平均 loss："
        f"{metrics['macro_total']:.6f}"
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

    train_dataset = SequencePoseDataset(
        train_roots
    )
    val_dataset = SequencePoseDataset(
        val_roots
    )

    train_loader = make_train_loader(
        train_dataset,
        args.batch_size,
        device,
        args.seed,
    )
    val_loader = make_val_loader(
        val_dataset,
        args.batch_size,
        device,
    )

    model = SequenceTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=(
            args.dim_feedforward
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
        / f"sequence_transformer_k{args.context}"
        / timestamp
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    config = {
        **vars(args),
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_roots": [
            str(path)
            for path in train_roots
        ],
        "val_roots": [
            str(path)
            for path in val_roots
        ],
        "strides": list(STRIDES),
        "model": "SequenceTransformer",
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
        model=model,
        loader=val_loader,
        device=device,
        bone_weight=args.bone_weight,
    )

    print_initial_metrics(initial_val)

    best_val = initial_val["macro_total"]
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
            model=model,
            loader=train_loader,
            device=device,
            bone_weight=args.bone_weight,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            bone_weight=args.bone_weight,
        )

        scheduler.step(
            val_metrics["macro_total"]
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
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
                epoch=epoch,
                metrics=val_metrics,
                config=config,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            metrics=val_metrics,
            config=config,
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
                f"{train_metrics['macro_total']:.6f} | "
                f"val="
                f"{val_metrics['macro_total']:.6f} | "
                f"MPJPE "
                f"s2={s2['mpjpe_px']:.4f}px "
                f"s4={s4['mpjpe_px']:.4f}px "
                f"s8={s8['mpjpe_px']:.4f}px | "
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