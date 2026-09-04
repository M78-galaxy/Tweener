#!/usr/bin/env python3
"""造火柴人三连帧评测集。

    python make_triplets.py --stride 3 --out data/triplets

stride 控制抽帧间隔，也就是「两张关键帧之间隔多远」。
stride 越大越接近日式动画一拍二/一拍三的情况，也越难。
建议先跑 stride=2 看看基线有多强，再跑 stride=4 看它在哪崩。

每个片段目录下：
    prev.png / gt.png / next.png   —— 渲染出来的三张图
    meta.json                      —— 关节坐标、姿态参数、运动类型标注
"""

import argparse, json
from pathlib import Path

import numpy as np

from stickman.motion import MOTIONS, MOTION_KIND, fk
from stickman.skeleton import render, CANVAS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/triplets")
    ap.add_argument("--stride", type=int, default=6,
                    help="prev 与 gt、gt 与 next 之间隔几帧")
    ap.add_argument("--frames", type=int, default=24, help="每个动作的稠密帧数")
    ap.add_argument("--size", type=int, default=CANVAS)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for name, make in MOTIONS.items():
        seq = make(n=args.frames)
        poses = np.array([fk(p) for p in seq])

        for i in range(0, len(seq) - 2 * args.stride):
            a, m, b = i, i + args.stride, i + 2 * args.stride
            clip = out_root / f"{name}_{i:03d}"
            clip.mkdir(exist_ok=True)

            for tag, idx in (("prev", a), ("gt", m), ("next", b)):
                render(poses[idx], size=args.size).save(clip / f"{tag}.png")

            json.dump({
                "motion": name,
                "kind": MOTION_KIND[name],
                "stride": args.stride,
                "size": args.size,
                # 坐标是 v0 的「免费真值」。等你想加难度，就把它删掉，
                # 逼自己从 PNG 里检测关键点 —— 那就是 AnimeInbet 里
                # 「栅格线稿 → 向量图」那一步的玩具版。
                "poses": {t: poses[k].tolist()
                          for t, k in (("prev", a), ("gt", m), ("next", b))},
                "params": {t: seq[k].tolist()
                           for t, k in (("prev", a), ("gt", m), ("next", b))},
            }, open(clip / "meta.json", "w"), indent=1)
            n_written += 1

    print(f"写了 {n_written} 组三连帧到 {out_root}/  (stride={args.stride})")


if __name__ == "__main__":
    main()
