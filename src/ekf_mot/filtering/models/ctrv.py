"""
CTRV 运动模型 - 恒转率恒速度模型
状态向量: x = [cx, cy, v, theta, omega, w, h]^T
"""

import math
import numpy as np

from ...core.constants import (
    IDX_CX, IDX_CY, IDX_V, IDX_THETA, IDX_OMEGA, IDX_W, IDX_H,
    DEFAULT_OMEGA_THRESHOLD,
)


def ctrv_predict(
    x: np.ndarray,
    dt: float,
    omega_threshold: float = DEFAULT_OMEGA_THRESHOLD,
) -> np.ndarray:
    """
    CTRV 非线性状态转移函数 f(x, dt)。

    当 |omega| < omega_threshold 时退化为 CV（匀速直线），避免除零。
    """
    cx, cy, v, theta, omega, w, h = (
        x[IDX_CX], x[IDX_CY], x[IDX_V],
        x[IDX_THETA], x[IDX_OMEGA], x[IDX_W], x[IDX_H],
    )

    x_new = x.copy()

    if abs(omega) < omega_threshold:
        x_new[IDX_CX] = cx + v * math.cos(theta) * dt
        x_new[IDX_CY] = cy + v * math.sin(theta) * dt
        x_new[IDX_THETA] = theta
    else:
        theta_new = theta + omega * dt
        x_new[IDX_CX] = cx + (v / omega) * (math.sin(theta_new) - math.sin(theta))
        x_new[IDX_CY] = cy + (v / omega) * (-math.cos(theta_new) + math.cos(theta))
        x_new[IDX_THETA] = theta_new

    # 速度轻微衰减，防止无观测时速度持续积累
    x_new[IDX_V] = v * 0.995

    return x_new


def ctrv_jacobian(
    x: np.ndarray,
    dt: float,
    omega_threshold: float = DEFAULT_OMEGA_THRESHOLD,
) -> np.ndarray:
    """
    CTRV 状态转移函数关于状态向量的雅可比矩阵 F = df/dx，shape (7, 7)。
    """
    v, theta, omega = x[IDX_V], x[IDX_THETA], x[IDX_OMEGA]

    F = np.eye(7, dtype=np.float64)
    # v 衰减对自身的偏导
    F[IDX_V, IDX_V] = 0.995

    if abs(omega) < omega_threshold:
        F[IDX_CX, IDX_V] = math.cos(theta) * dt
        F[IDX_CX, IDX_THETA] = -v * math.sin(theta) * dt
        F[IDX_CY, IDX_V] = math.sin(theta) * dt
        F[IDX_CY, IDX_THETA] = v * math.cos(theta) * dt
    else:
        theta_new = theta + omega * dt
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        sin_tn = math.sin(theta_new)
        cos_tn = math.cos(theta_new)

        F[IDX_CX, IDX_V] = (sin_tn - sin_t) / omega
        F[IDX_CX, IDX_THETA] = (v / omega) * (cos_tn - cos_t)
        F[IDX_CX, IDX_OMEGA] = (
            (v / omega) * (cos_tn * dt)
            - (v / omega**2) * (sin_tn - sin_t)
        )

        F[IDX_CY, IDX_V] = (-cos_tn + cos_t) / omega
        F[IDX_CY, IDX_THETA] = (v / omega) * (sin_tn - sin_t)
        F[IDX_CY, IDX_OMEGA] = (
            (v / omega) * (sin_tn * dt)
            - (v / omega**2) * (-cos_tn + cos_t)
        )

        F[IDX_THETA, IDX_OMEGA] = dt

    return F
