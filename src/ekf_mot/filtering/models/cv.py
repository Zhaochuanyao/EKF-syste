"""
CV 匀速运动模型 - 作为对照模型
状态向量: x = [cx, cy, vx, vy, w, h]^T
"""

import numpy as np


def cv_predict(x: np.ndarray, dt: float) -> np.ndarray:
    """
    CV 线性状态转移: x_new = F * x

    状态向量: [cx, cy, vx, vy, w, h]
    """
    x_new = x.copy()
    x_new[0] = x[0] + x[2] * dt  # cx += vx * dt
    x_new[1] = x[1] + x[3] * dt  # cy += vy * dt
    return x_new


def cv_transition_matrix(dt: float) -> np.ndarray:
    """
    CV 状态转移矩阵 F (6x6)。
    线性模型，F 即为精确雅可比。
    """
    F = np.eye(6, dtype=np.float64)
    F[0, 2] = dt  # d(cx)/d(vx)
    F[1, 3] = dt  # d(cy)/d(vy)
    return F


def cv_process_noise(dt: float, std_acc: float = 1.0) -> np.ndarray:
    """
    CV 过程噪声矩阵 Q (6x6)，基于离散白噪声加速度模型。
    """
    q = std_acc ** 2
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4

    Q = np.zeros((6, 6), dtype=np.float64)
    # cx, vx 方向
    Q[0, 0] = q * dt4 / 4
    Q[0, 2] = q * dt3 / 2
    Q[2, 0] = q * dt3 / 2
    Q[2, 2] = q * dt2
    # cy, vy 方向
    Q[1, 1] = q * dt4 / 4
    Q[1, 3] = q * dt3 / 2
    Q[3, 1] = q * dt3 / 2
    Q[3, 3] = q * dt2
    return Q
