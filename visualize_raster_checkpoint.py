#!/usr/bin/env python3
"""可视化 RasterUNet checkpoint 的预测结果。"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from stickman.models import RasterUNet
from stickman.raster_dataset import (
    RasterSequenceDataset,
)


ACTIONS = (
    "walk",
    "wave",
    "swing",
    "squat",
    "lean",
)

STRIDES = (2, 4, 8)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
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
        "--out",
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


def mask_to_image(mask):
    mask = (
        mask.detach()
        .float()
        .cpu()
    )

    if mask.ndim == 3:
        mask = mask.squeeze(0)

    pixels = (
        (1.0 - mask)
        .clamp(0.0, 1.0)
        .mul(255)
        .to(torch.uint8)
        .numpy()
    )

    return Image.fromarray(
        pixels,
        mode="L",
    ).convert("RGB")


def make_overlay(
    prediction,
    target,
):
    prediction = (
        prediction.detach()
        .cpu()
        .squeeze()
        .numpy()
    )
    target = (
        target.detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    prediction_mask = (
        prediction >= 0.5
    )
    target_mask = target >= 0.5

    height, width = target.shape

    image = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    target_only = (
        target_mask
        & ~prediction_mask
    )
    prediction_only = (
        prediction_mask
        & ~target_mask
    )
    overlap = (
        prediction_mask
        & target_mask
    )

    # 黑：只有 target。
    image[target_only] = (
        0,
        0,
        0,
    )

    # 红：只有预测。
    image[prediction_only] = (
        220,
        40,
        40,
    )

    # 紫：预测和 target 重合。
    image[overlap] = (
        100,
        20,
        130,
    )

    return Image.fromarray(
        image,
        mode="RGB",
    )


def soft_metrics(
    prediction,
    target,
):
    intersection = (
        prediction * target
    ).sum()

    dice = (
        2.0 * intersection + 1.0
    ) / (
        prediction.sum()
        + target.sum()
        + 1.0
    )

    union = (
        prediction
        + target
        - prediction * target
    ).sum()

    iou = (
        intersection + 1.0
    ) / (
        union + 1.0
    )

    return (
        float(dice.item()),
        float(iou.item()),
    )


def main():
    args = parse_args()

    checkpoint = Path(
        args.checkpoint
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint 不存在：{checkpoint}"
        )

    if args.device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            args.device
        )

    saved = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    config = saved["config"]

    context = int(
        config["context"]
    )
    size = int(
        config["size"]
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

    if args.out is None:
        output_root = (
            checkpoint.parent
        )
    else:
        output_root = Path(
            args.out
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"checkpoint epoch：{saved['epoch']}"
    )
    print(f"设备：{device}")
    print(f"输出目录：{output_root}")

    for stride in STRIDES:
        rows = []

        for action in ACTIONS:
            root = (
                Path(args.data_base)
                / f"{action}_val_001"
                / f"k{context}"
                / f"s{stride}"
            )

            dataset = (
                RasterSequenceDataset(
                    [root],
                    size=size,
                )
            )

            # 使用排序后的中位样本，
            # 不手工挑最好看的案例。
            index = len(dataset) // 2
            sample = dataset[index]

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
            target = (
                sample["target"]
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                prediction = (
                    model.predict_image(
                        model_input,
                        pixel_lerp,
                    )
                )

            dice, iou = soft_metrics(
                prediction,
                target,
            )

            binary_prediction = (
                prediction >= 0.5
            ).float()

            cells = [
                add_label(
                    mask_to_image(
                        sample[
                            "prev_image"
                        ]
                    ),
                    (
                        f"{sample['kind']} "
                        "| prev"
                    ),
                ),
                add_label(
                    mask_to_image(
                        sample[
                            "next_image"
                        ]
                    ),
                    "next",
                ),
                add_label(
                    mask_to_image(
                        sample[
                            "pixel_lerp"
                        ]
                    ),
                    "pixel lerp",
                ),
                add_label(
                    mask_to_image(
                        sample[
                            "pose_lerp"
                        ]
                    ),
                    "pose lerp",
                ),
                add_label(
                    mask_to_image(
                        prediction[0]
                    ),
                    (
                        "UNet probability "
                        f"D={dice:.3f}"
                    ),
                ),
                add_label(
                    mask_to_image(
                        binary_prediction[0]
                    ),
                    "UNet threshold 0.5",
                ),
                add_label(
                    make_overlay(
                        prediction[0],
                        target[0],
                    ),
                    (
                        "overlay "
                        "black=GT red=pred"
                    ),
                ),
                add_label(
                    mask_to_image(
                        target[0]
                    ),
                    "target",
                ),
            ]

            rows.append(cells)

            print(
                f"s{stride} "
                f"{sample['kind']:<20} "
                f"Dice={dice:.4f} "
                f"IoU={iou:.4f} "
                f"{sample['clip']}"
            )

        columns = 8

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

        output_path = (
            output_root
            / f"preview_s{stride}.png"
        )

        sheet.save(output_path)

        print(
            f"s{stride} 预览图："
            f"{output_path}"
        )


if __name__ == "__main__":
    main()