"""
三个基线。模型必须跟它们比，比不过就是没用。

命名跟 LaFAN1 那套基准对齐（zero-velocity / interpolation），
这样将来你想跟论文里的数字对照时不用改口径。
"""

import numpy as np


def zero_velocity(prev_pose, next_pose, t=0.5):
    """中间帧 = 前一帧。看着蠢，但它是尺子：

    它的误差就等于「这段动作到底动了多少」。如果某个动作上你的模型
    只比零速度好一点点，说明它其实什么都没学到，只是在抄前一帧。
    """
    return np.array(prev_pose, dtype=float)


def lerp(prev_pose, next_pose, t=0.5):
    """对关节坐标做线性插值。

    这是真正要打败的对手。注意它有个结构性缺陷：关节沿直线（弦）移动，
    而真实的旋转是沿圆弧走的，所以旋转越大，肢体被插得越短。
    bone_length_error 就是抓这个的。
    """
    p, n = np.asarray(prev_pose, float), np.asarray(next_pose, float)
    return (1 - t) * p + t * n


def angle_lerp(prev_params, next_params, t=0.5):
    """对关节角做插值，再走一遍 FK。

    这不是一个公平的基线 —— 它用到了「我知道骨架结构和关节角」这个
    在真实线稿里拿不到的信息。把它当作**几何上界**：
    一个理想的、懂结构的方法最多能做到这个程度。
    你的模型的目标是从像素/坐标里逼近它。
    """
    from .motion import fk
    p, n = np.asarray(prev_params, float), np.asarray(next_params, float)
    d = n - p
    d[2:] = (d[2:] + np.pi) % (2 * np.pi) - np.pi   # 角度走最短弧
    return fk(p + t * d)


BASELINES = {
    "zero_velocity": zero_velocity,
    "lerp": lerp,
}
