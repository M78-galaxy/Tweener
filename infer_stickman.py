"""火柴人多帧插帧命令行工具。"""

import argparse
import json
from pathlib import Path

import numpy as np
from stickman.interpolator import StickmanInterpolator
from stickman.request import load_request
from stickman.skeleton import render

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用多帧 Transformer 预测火柴人中间帧",
    )

    parser.add_argument(
        "meta",
        type=Path,
        help="包含左右上下文姿态的 JSON 请求文件",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Transformer checkpoint",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="推理设备；不指定时自动选择",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/v5_inference"),
        help="输出目录",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="输出图像尺寸",
    )

    return parser.parse_args()

def run_request(
    args,
    interpolator=None,
):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint 不存在：{args.checkpoint}"
        )

    if args.size <= 0:
        raise ValueError(
            "size 必须大于 0"
        )

    sample = load_request(
    args.meta
    )

    if interpolator is None:
        interpolator = StickmanInterpolator(
            checkpoint=args.checkpoint,
            device=args.device,
            size=args.size,
        )

    if sample["context"] != interpolator.context:
        raise ValueError(
            "样本与模型的 context 不一致："
            f"样本={sample['context']}，"
            f"模型={interpolator.context}"
        )

    if sample["stride"] not in interpolator.strides:
        raise ValueError(
            f"模型不支持 stride={sample['stride']}；"
            f"支持 {interpolator.strides}"
        )

    prediction_pose, prediction_image = (
        interpolator.render_midpoint(
            sample["left_poses"],
            sample["right_poses"],
            sample["left_offsets"],
            sample["right_offsets"],
        )
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        args.out / "prediction.png"
    )

    target_path = None

    if sample["target_pose"] is not None:
        target_path = (
            args.out / "target.png"
        )

    pose_path = (
        args.out / "prediction_pose.npy"
    )
    result_path = (
        args.out / "result.json"
    )

    prediction_image.save(
        prediction_path
    )
    if target_path is not None:
        target_image = render(
            sample["target_pose"],
            size=args.size,
        )
        target_image.save(
            target_path
        )
    np.save(
        pose_path,
        prediction_pose,
    )

    result = {
        "schema_version": (
            sample["schema_version"]
        ),
        "input_request": str(args.meta),
        "request_id": sample["request_id"],
        "motion": sample["motion"],
        "kind": sample["kind"],
        "stride": sample["stride"],
        "context": sample["context"],
        "size": args.size,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": interpolator.epoch,
        "device": str(interpolator.device),
        "supported_strides": list(
            interpolator.strides
        ),
        "time_offsets": {
            "left": (
                sample["left_offsets"].tolist()
            ),
            "right": (
                sample["right_offsets"].tolist()
            ),
        },
        "prediction_shape": list(
            prediction_pose.shape
        ),
        "prediction_is_finite": bool(
            np.isfinite(
                prediction_pose
            ).all()
        ),
        "has_target": (
            sample["target_pose"] is not None
        ),
        "outputs": {
            "prediction": str(
                prediction_path
            ),
        "target": (
            None
            if target_path is None
            else str(target_path)
        ),
            "prediction_pose": str(
                pose_path
            ),
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

    print("推理完成")
    print("设备：", interpolator.device)
    print("checkpoint epoch：", interpolator.epoch)
    print(
        "request id：",
        result["request_id"],
    )
    print("stride：", sample["stride"])
    print("context：", sample["context"])
    print("支持 stride：", interpolator.strides)
    print("预测形状：", prediction_pose.shape)
    print(
        "全部有限：",
        result["prediction_is_finite"],
    )
    print("预测图：", prediction_path)
    if target_path is None:
        print("真值图：无")
    else:
        print("真值图：", target_path)
    print("预测姿态：", pose_path)
    print("运行记录：", result_path)

    return result

def main():
    args = parse_args()
    run_request(args)

if __name__ == "__main__":
    main()