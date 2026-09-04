"""
火柴人骨架的唯一真相来源。

关节顺序一旦定下就不要改 —— 后面所有数据、模型权重、评测结果都依赖它。
这里用 COCO-17 的顺序，理由是 MediaPipe / RTMPose / OpenPose 都能直接映射过来，
将来你想换成真人视频提取的姿态，不用改一行下游代码。
"""

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- 关节定义

JOINTS = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]
N_JOINTS = len(JOINTS)
IDX = {name: i for i, name in enumerate(JOINTS)}

# 画线时用哪些连接。脸部关键点不画线，头部单独画个圆。
BONES = [
    (5, 6),    # 肩
    (11, 12),  # 胯
    (5, 11), (6, 12),   # 躯干两侧
    (5, 7), (7, 9),     # 左臂
    (6, 8), (8, 10),    # 右臂
    (11, 13), (13, 15), # 左腿
    (12, 14), (14, 16), # 右腿
]

# 评测「肢体有没有被插短」时要检查的骨头（骨长应当守恒）
LIMB_BONES = [(5, 7), (7, 9), (6, 8), (8, 10),
              (11, 13), (13, 15), (12, 14), (14, 16)]


# ---------------------------------------------------------------- 渲染

CANVAS = 256          # 输出图边长（像素）
LINE_WIDTH = 3


def render(pose, size=CANVAS, line_width=LINE_WIDTH, color=0, bg=255,
           img=None):
    """把 (17, 2) 的归一化坐标画成火柴人。

    pose: ndarray (17, 2)，坐标在 [0, 1] 区间，原点左上，y 向下。
    img:  传入已有的 PIL Image 则叠加绘制（用来做 pred/gt 叠图）。
    返回 PIL Image (mode 'L' 灰度 或 'RGB'，取决于传入的 img)。
    """
    pose = np.asarray(pose, dtype=float)
    assert pose.shape == (N_JOINTS, 2), f"期望 (17,2)，拿到 {pose.shape}"

    if img is None:
        img = Image.new("L", (size, size), bg)
    draw = ImageDraw.Draw(img)

    px = pose * size

    for a, b in BONES:
        draw.line([tuple(px[a]), tuple(px[b])], fill=color, width=line_width)

    # 头：以鼻子为中心，半径取肩宽的一半，看起来比例正常
    shoulder_w = np.linalg.norm(px[IDX["left_shoulder"]] - px[IDX["right_shoulder"]])
    r = max(4.0, shoulder_w * 0.5)
    cx, cy = px[IDX["nose"]]
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=line_width)

    return img


def render_overlay(pred, gt, size=CANVAS):
    """预测(红) 叠在 真值(黑) 上 —— 一眼看出差在哪，比并排放有用得多。"""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    render(gt,   size=size, color=(0, 0, 0),   img=img, line_width=3)
    render(pred, size=size, color=(220, 40, 40), img=img, line_width=2)
    return img


# ---------------------------------------------------------------- 度量

def mpjpe(pred, gt, size=CANVAS):
    """平均每关节位置误差，单位：像素。越小越好。"""
    pred, gt = np.asarray(pred), np.asarray(gt)
    return float(np.linalg.norm((pred - gt) * size, axis=1).mean())


def bone_length_error(pred, gt, size=CANVAS):
    """四肢骨长的平均绝对偏差，单位：像素。

    这个指标专门抓「插值把胳膊插短了」这类错误。
    对位置做线性插值时，旋转的肢体会沿弦而不是沿弧移动，骨头必然缩短，
    这里的数值会直接把它暴露出来 —— 而 MPJPE 有时看不太出来。
    """
    pred, gt = np.asarray(pred) * size, np.asarray(gt) * size
    errs = []
    for a, b in LIMB_BONES:
        lp = np.linalg.norm(pred[a] - pred[b])
        lg = np.linalg.norm(gt[a] - gt[b])
        errs.append(abs(lp - lg))
    return float(np.mean(errs))
