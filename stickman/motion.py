"""
程序化生成火柴人动作。

⚠️ 这些数据只用来跑通管线和调试，**不要拿来训练模型**。
模型只会学会你写进正弦函数里的东西，学不到真实运动的统计规律。
v1 开始换成 MediaPipe 从真实视频提取的姿态。

之所以用「固定骨长 + 关节角 + 正向运动学」而不是直接对坐标写正弦，
是因为这样生成的动作骨长严格守恒 —— 于是「对坐标做线性插值会把肢体插短」
这个失败模式就能被干净地测出来。这是 v0 最想让你看到的东西。
"""

import numpy as np
from .skeleton import N_JOINTS, IDX

# ------------------------------------------------ 骨长（归一化，画布边长为 1）

L_SPINE = 0.22      # 胯中点 → 颈（肩中点）
L_NECK_HEAD = 0.10  # 颈 → 鼻
L_SHOULDER = 0.075  # 颈 → 单侧肩
L_HIP = 0.045       # 胯中点 → 单侧胯
L_UPPERARM = 0.11
L_FOREARM = 0.10
L_THIGH = 0.13
L_SHIN = 0.12

# 姿态参数向量的含义（长度 11）。角度单位为弧度，从 +x 轴起算，y 轴向下，
# 所以 -pi/2 表示正上方，+pi/2 表示正下方。
PARAM_NAMES = [
    "root_x", "root_y", "spine",
    "l_upperarm", "l_forearm", "r_upperarm", "r_forearm",
    "l_thigh", "l_shin", "r_thigh", "r_shin",
]
N_PARAMS = len(PARAM_NAMES)


def _dir(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def fk(params):
    """正向运动学：11 维参数 → (17, 2) 关节坐标。"""
    p = np.asarray(params, dtype=float)
    pose = np.zeros((N_JOINTS, 2))

    pelvis = p[0:2]
    spine_a = p[2]
    neck = pelvis + L_SPINE * _dir(spine_a)
    perp = _dir(spine_a + np.pi / 2)

    nose = neck + L_NECK_HEAD * _dir(spine_a)
    pose[IDX["nose"]] = nose
    # 眼睛耳朵不参与绘制，但占着 COCO-17 的位子，随便摆在头附近就行
    pose[IDX["left_eye"]]  = nose + 0.02 * perp
    pose[IDX["right_eye"]] = nose - 0.02 * perp
    pose[IDX["left_ear"]]  = nose + 0.04 * perp
    pose[IDX["right_ear"]] = nose - 0.04 * perp

    ls = neck + L_SHOULDER * perp
    rs = neck - L_SHOULDER * perp
    lh = pelvis + L_HIP * perp
    rh = pelvis - L_HIP * perp
    pose[IDX["left_shoulder"]], pose[IDX["right_shoulder"]] = ls, rs
    pose[IDX["left_hip"]], pose[IDX["right_hip"]] = lh, rh

    le = ls + L_UPPERARM * _dir(p[3]);  pose[IDX["left_elbow"]] = le
    pose[IDX["left_wrist"]] = le + L_FOREARM * _dir(p[4])
    re = rs + L_UPPERARM * _dir(p[5]);  pose[IDX["right_elbow"]] = re
    pose[IDX["right_wrist"]] = re + L_FOREARM * _dir(p[6])

    lk = lh + L_THIGH * _dir(p[7]);   pose[IDX["left_knee"]] = lk
    pose[IDX["left_ankle"]] = lk + L_SHIN * _dir(p[8])
    rk = rh + L_THIGH * _dir(p[9]);   pose[IDX["right_knee"]] = rk
    pose[IDX["right_ankle"]] = rk + L_SHIN * _dir(p[10])

    return pose


# ------------------------------------------------ 动作

UP, DOWN = -np.pi / 2, np.pi / 2


def _rest():
    return np.array([0.5, 0.55, UP,
                     DOWN - 0.25, DOWN - 0.15, DOWN + 0.25, DOWN + 0.15,
                     DOWN - 0.05, DOWN, DOWN + 0.05, DOWN])


def walk(n=24, cycles=1.0):
    """走路循环：四肢摆幅中等，是 lerp 应该能应付的那类动作。"""
    out = []
    for i in range(n):
        t = 2 * np.pi * cycles * i / n
        p = _rest()
        p[1] = 0.55 - 0.012 * abs(np.sin(t))           # 重心起伏
        p[7] = DOWN - 0.55 * np.sin(t)                 # 左大腿
        p[8] = p[7] + 0.35 * (1 - np.cos(t)) / 2
        p[9] = DOWN + 0.55 * np.sin(t)                 # 右大腿
        p[10] = p[9] + 0.35 * (1 + np.cos(t)) / 2
        p[3] = DOWN + 0.45 * np.sin(t)                 # 手臂反相
        p[4] = p[3] - 0.30
        p[5] = DOWN - 0.45 * np.sin(t)
        p[6] = p[5] - 0.30
        out.append(p)
    return np.array(out)


def wave(n=24):
    """挥手：小幅局部运动，lerp 的舒适区。"""
    out = []
    for i in range(n):
        t = 2 * np.pi * i / n
        p = _rest()
        p[5] = UP + 0.35                                # 右上臂举起
        p[6] = UP - 0.45 * np.sin(t)                    # 小臂来回摆
        out.append(p)
    return np.array(out)


def big_arm_swing(n=24):
    """大幅抡臂：手臂从垂下扫到举高，约 170 度。

    这是专门用来打线性插值的。旋转幅度一大，对坐标做 lerp 就会沿弦而不是沿弧走，
    中间帧的胳膊会明显变短，甚至缩进身体里。
    """
    out = []
    for i in range(n):
        s = i / (n - 1)
        a = DOWN + (UP - 0.2 - DOWN) * s
        p = _rest()
        p[5], p[6] = a, a + 0.1
        out.append(p)
    return np.array(out)


def squat(n=24):
    """下蹲起立：躯干平移 + 膝盖大角度弯曲。"""
    out = []
    for i in range(n):
        t = 2 * np.pi * i / n
        d = (1 - np.cos(t)) / 2
        p = _rest()
        p[1] = 0.55 + 0.10 * d
        p[7] = DOWN - 0.9 * d;  p[8] = DOWN + 0.9 * d
        p[9] = DOWN - 0.9 * d;  p[10] = DOWN + 0.9 * d
        p[3] = DOWN - 1.1 * d;  p[4] = p[3]
        p[5] = DOWN + 1.1 * d;  p[6] = p[5]
        out.append(p)
    return np.array(out)


def lean(n=24):
    """整体侧倾：全身刚性旋转，考察模型会不会学到「整体旋转」这件事。"""
    out = []
    for i in range(n):
        t = 2 * np.pi * i / n
        a = 0.45 * np.sin(t)
        p = _rest()
        p[2] = UP + a
        for k in (3, 4, 5, 6):
            p[k] += a
        out.append(p)
    return np.array(out)


MOTIONS = {
    "walk": walk,
    "wave": wave,
    "big_arm_swing": big_arm_swing,
    "squat": squat,
    "lean": lean,
}

# 运动类型标注 —— 评测时按类型分开看，不看总平均。
MOTION_KIND = {
    "walk": "medium_cyclic",
    "wave": "small_local",
    "big_arm_swing": "large_rotation",
    "squat": "large_translation",
    "lean": "global_rotation",
}
