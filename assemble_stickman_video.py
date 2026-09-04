"""把批量火柴人预测结果装配成 MP4 和 GIF。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "按照 manifest 顺序，"
            "把预测图片装配成动画"
        ),
    )

    parser.add_argument(
        "manifest",
        type=Path,
        help="序列 manifest.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="批量推理输出目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "runs/v5_sequence_video"
        ),
        help="动画输出目录",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "输出帧率；不指定时读取 "
            "manifest.output_fps"
        ),
    )

    return parser.parse_args()


def load_frames(
    manifest_path,
    predictions_root,
):
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest 不存在："
            f"{manifest_path}"
        )

    if not predictions_root.is_dir():
        raise FileNotFoundError(
            f"预测目录不存在："
            f"{predictions_root}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    request_entries = manifest.get(
        "requests"
    )

    if (
        not isinstance(request_entries, list)
        or not request_entries
    ):
        raise ValueError(
            "manifest.requests "
            "必须是非空列表"
        )

    request_ids = []
    frames = []
    expected_size = None

    for entry in request_entries:
        request_path = (
            manifest_path.parent
            / entry
        )

        if not request_path.is_file():
            raise FileNotFoundError(
                f"请求文件不存在："
                f"{request_path}"
            )

        with request_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            request = json.load(file)

        request_id = str(
            request["request_id"]
        )

        image_path = (
            predictions_root
            / request_id
            / "prediction.png"
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"预测图不存在："
                f"{image_path}"
            )

        with Image.open(
            image_path
        ) as image:
            frame = image.convert(
                "RGB"
            ).copy()

        if expected_size is None:
            expected_size = frame.size
        elif frame.size != expected_size:
            raise ValueError(
                "预测图尺寸不一致："
                f"期望 {expected_size}，"
                f"实际 {frame.size}"
            )

        request_ids.append(
            request_id
        )
        frames.append(
            frame
        )

    if len(set(request_ids)) != len(
        request_ids
    ):
        raise ValueError(
            "序列中存在重复 request_id"
        )

    return manifest, request_ids, frames


def write_mp4(
    frames,
    output_path,
    fps,
):
    width, height = frames[0].size

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建 MP4：{output_path}"
        )

    try:
        for frame in frames:
            rgb = np.asarray(
                frame,
                dtype=np.uint8,
            )
            bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )
            writer.write(
                bgr
            )
    finally:
        writer.release()

    if (
        not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"MP4 输出失败：{output_path}"
        )


def write_gif(
    frames,
    output_path,
    fps,
):
    duration_ms = max(
        1,
        round(1000.0 / fps),
    )

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )

    if (
        not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"GIF 输出失败：{output_path}"
        )

    return duration_ms


def main():
    args = parse_args()

    (
        manifest,
        request_ids,
        frames,
    ) = load_frames(
        args.manifest,
        args.predictions,
    )

    fps = (
        float(args.fps)
        if args.fps is not None
        else float(
            manifest["output_fps"]
        )
    )

    if fps <= 0:
        raise ValueError(
            "fps 必须大于 0"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mp4_path = (
        args.out / "prediction.mp4"
    )
    gif_path = (
        args.out / "prediction.gif"
    )
    result_path = (
        args.out / "sequence_result.json"
    )

    write_mp4(
        frames,
        mp4_path,
        fps,
    )

    gif_duration_ms = write_gif(
        frames,
        gif_path,
        fps,
    )

    width, height = frames[0].size

    result = {
        "schema_version": 1,
        "batch_id": manifest.get(
            "batch_id"
        ),
        "sequence_id": manifest.get(
            "sequence_id"
        ),
        "manifest": str(
            args.manifest
        ),
        "predictions": str(
            args.predictions
        ),
        "frame_count": len(frames),
        "fps": fps,
        "duration_seconds": (
            len(frames) / fps
        ),
        "gif_frame_duration_ms": (
            gif_duration_ms
        ),
        "width": width,
        "height": height,
        "request_ids": request_ids,
        "outputs": {
            "mp4": str(mp4_path),
            "gif": str(gif_path),
        },
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("序列装配完成")
    print("序列：", result["sequence_id"])
    print("帧数：", len(frames))
    print("帧率：", fps)
    print(
        "时长：",
        f"{result['duration_seconds']:.2f} s",
    )
    print(
        "尺寸：",
        f"{width} x {height}",
    )
    print("MP4：", mp4_path)
    print("GIF：", gif_path)
    print("记录：", result_path)


if __name__ == "__main__":
    main()