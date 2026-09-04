#!/usr/bin/env python3
"""训练火柴人姿态插帧残差 MLP。"""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from stickman.dataset import PoseTripletDataset
from stickman.losses import limb_lengths, pose_loss
from stickman.models import ResidualMLP
from stickman.skeleton import CANVAS


ACTIONS = (
    "walk",
    "wave",
    "swing",
    "squat",
    "lean",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stride",
        type=int,
        required=True,
        choices=(2, 4, 8),
    )
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
        default=1e-3,
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
        "--hidden-dim",
        type=int,
        default=128,
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


def make_roots(base, split, stride):
    return [
        Path(base)
        / f"{action}_{split}_001"
        / f"s{stride}"
        for action in ACTIONS
    ]


def make_loader(
    dataset,
    batch_size,
    shuffle,
    device,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def run_epoch(
    model,
    loader,
    device,
    bone_weight,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)

    totals = {
        "total": 0.0,
        "position": 0.0,
        "bone": 0.0,
        "mpjpe_px": 0.0,
        "bone_px": 0.0,
    }
    sample_count = 0

    for batch in loader:
        model_input = batch["input"].to(
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

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = model.predict_pose(
                model_input,
                lerp_pose,
            )

            losses = pose_loss(
                prediction,
                gt_pose,
                bone_weight=bone_weight,
            )

            if training:
                losses["total"].backward()

                clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        with torch.no_grad():
            joint_error = torch.linalg.vector_norm(
                prediction - gt_pose,
                dim=-1,
            )

            mpjpe_px = (
                joint_error
                * scale[:, None]
                * CANVAS
            ).mean()

            prediction_bones = limb_lengths(
                prediction
            )
            target_bones = limb_lengths(
                gt_pose
            )

            bone_px = (
                torch.abs(
                    prediction_bones
                    - target_bones
                )
                * scale[:, None]
                * CANVAS
            ).mean()

        batch_size = model_input.shape[0]
        sample_count += batch_size

        totals["total"] += (
            losses["total"].detach().item()
            * batch_size
        )
        totals["position"] += (
            losses["position"].detach().item()
            * batch_size
        )
        totals["bone"] += (
            losses["bone"].detach().item()
            * batch_size
        )
        totals["mpjpe_px"] += (
            mpjpe_px.item() * batch_size
        )
        totals["bone_px"] += (
            bone_px.item() * batch_size
        )

    return {
        name: value / sample_count
        for name, value in totals.items()
    }


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
            "model_state": model.state_dict(),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def write_history(path, history):
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_total",
        "train_position",
        "train_bone",
        "train_mpjpe_px",
        "train_bone_px",
        "val_total",
        "val_position",
        "val_bone",
        "val_mpjpe_px",
        "val_bone_px",
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
        writer.writerows(history)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_roots = make_roots(
        "data/formal_triplets",
        "train",
        args.stride,
    )
    val_roots = make_roots(
        "data/val_triplets",
        "val",
        args.stride,
    )

    train_dataset = PoseTripletDataset(
        train_roots
    )
    val_dataset = PoseTripletDataset(
        val_roots
    )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        device,
        args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        False,
        device,
        args.seed + 1,
    )

    model = ResidualMLP(
        hidden_dim=args.hidden_dim,
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
        / f"mlp_s{args.stride}"
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
    print(f"stride：{args.stride}")
    print(
        f"训练样本：{len(train_dataset)}"
    )
    print(
        f"验证样本：{len(val_dataset)}"
    )
    print(f"输出目录：{run_dir}")

    # 零初始化模型应当严格等于 lerp。
    initial_val = run_epoch(
        model=model,
        loader=val_loader,
        device=device,
        bone_weight=args.bone_weight,
    )

    print(
        "初始验证（等于 lerp）："
        f"MPJPE={initial_val['mpjpe_px']:.4f}px，"
        f"骨长={initial_val['bone_px']:.4f}px"
    )

    best_val = initial_val["total"]
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
            val_metrics["total"]
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            **{
                f"train_{name}": value
                for name, value
                in train_metrics.items()
            },
            **{
                f"val_{name}": value
                for name, value
                in val_metrics.items()
            },
        }
        history.append(row)

        write_history(
            run_dir / "history.csv",
            history,
        )

        improved = (
            val_metrics["total"]
            < best_val - 1e-7
        )

        if improved:
            best_val = val_metrics["total"]
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
            print(
                f"epoch {epoch:03d} | "
                f"train={train_metrics['total']:.6f} | "
                f"val={val_metrics['total']:.6f} | "
                f"val MPJPE="
                f"{val_metrics['mpjpe_px']:.4f}px | "
                f"val bone="
                f"{val_metrics['bone_px']:.4f}px | "
                f"lr={learning_rate:.2e}"
            )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "提前停止：验证损失连续 "
                f"{args.patience} 个 epoch "
                "没有改善"
            )
            break

    print(f"最佳 epoch：{best_epoch}")
    print(f"最佳验证损失：{best_val:.6f}")
    print(
        f"最佳模型：{run_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()