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

    当 |omega| < omega_threshold 时，退化为匀速直线运动（CV），
    避免除以接近零的 omega 导致数值不稳定。

    Args:
        x: 状态向量 [cx, cy, v, theta, omega, w, h]，shape (7,)
        dt: 时间步长（秒）
        omega_threshold: omega 接近零的判断阈值

    Returns:
        预测后的状态向量，shape (7,)
    """
    cx, cy, v, theta, omega, w, h = (
        x[IDX_CX], x[IDX_CY], x[IDX_V],
        x[IDX_THETA], x[IDX_OMEGA], x[IDX_W], x[IDX_H],
    )

    x_new = x.copy()

    if abs(omega) < omega_threshold:
        # ── 近似直线运动（CV 退化）────────────────────────────
        # 当 omega ≈ 0 时，CTRV 退化为匀速直线运动
        # dx = v * cos(theta) * dt
        # dy = v * sin(theta) * dt
        x_new[IDX_CX] = cx + v * math.cos(theta) * dt
        x_new[IDX_CY] = cy + v * math.sin(theta) * dt
        x_new[IDX_THETA] = theta  # 航向角不变
    else:
        # ── 标准 CTRV 转移 ────────────────────────────────────
        # dx = (v/omega) * (sin(theta + omega*dt) - sin(theta))
        # dy = (v/omega) * (-cos(theta + omega*dt) + cos(theta))
        theta_new = theta + omega * dt
        x_new[IDX_CX] = cx + (v / omega) * (math.sin(theta_new) - math.sin(theta))
        x_new[IDX_CY] = cy + (v / omega) * (-math.cos(theta_new) + math.cos(theta))
        x_new[IDX_THETA] = theta_new

    # v, omega, w, h 保持不变（匀速假设）
    # 速度轻微衰减：防止无观测时速度无限积累（0.98^25fps ≈ 0.6/s，合理）
    x_new[IDX_V] = v * 0.98

    return x_new


def ctrv_jacobian(
    x: np.ndarray,
    dt: float,
    omega_threshold: float = DEFAULT_OMEGA_THRESHOLD,
) -> np.ndarray:
    """
    CTRV 状态转移函数关于状态向量的雅可比矩阵 F = df/dx。

    F 是 7x7 矩阵，F[i,j] = d(x_new[i]) / d(x[j])。

    Args:
        x: 当前状态向量 [cx, cy, v, theta, omega, w, h]
        dt: 时间步长
        omega_threshold: omega 接近零的判断阈值

    Returns:
        雅可比矩阵 F，shape (7, 7)
    """
    v, theta, omega = x[IDX_V], x[IDX_THETA], x[IDX_OMEGA]

    # 初始化为单位矩阵（大多数状态对自身的偏导为1）
    F = np.eye(7, dtype=np.float64)

    if abs(omega) < omega_threshold:
        # ── CV 退化情况下的雅可比 ─────────────────────────────
        # d(cx_new)/d(v)     = cos(theta) * dt
        # d(cx_new)/d(theta) = -v * sin(theta) * dt
        # d(cy_new)/d(v)     = sin(theta) * dt
        # d(cy_new)/d(theta) = v * cos(theta) * dt
        F[IDX_CX, IDX_V] = math.cos(theta) * dt
        F[IDX_CX, IDX_THETA] = -v * math.sin(theta) * dt
        F[IDX_CY, IDX_V] = math.sin(theta) * dt
        F[IDX_CY, IDX_THETA] = v * math.cos(theta) * dt
    else:
        # ── 标准 CTRV 雅可比 ──────────────────────────────────
        theta_new = theta + omega * dt
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        sin_tn = math.sin(theta_new)
        cos_tn = math.cos(theta_new)

        # d(cx_new)/d(v)
        F[IDX_CX, IDX_V] = (sin_tn - sin_t) / omega
        # d(cx_new)/d(theta)
        F[IDX_CX, IDX_THETA] = (v / omega) * (cos_tn - cos_t)
        # d(cx_new)/d(omega)
        F[IDX_CX, IDX_OMEGA] = (
            (v / omega) * (cos_tn * dt)
            - (v / omega**2) * (sin_tn - sin_t)
        )

        # d(cy_new)/d(v)
        F[IDX_CY, IDX_V] = (-cos_tn + cos_t) / omega
        # d(cy_new)/d(theta)
        F[IDX_CY, IDX_THETA] = (v / omega) * (sin_tn - sin_t)
        # d(cy_new)/d(omega)
        F[IDX_CY, IDX_OMEGA] = (
            (v / omega) * (sin_tn * dt)
            - (v / omega**2) * (-cos_tn + cos_t)
        )

        # d(theta_new)/d(theta) = 1 (已在单位矩阵中)
        # d(theta_new)/d(omega) = dt
        F[IDX_THETA, IDX_OMEGA] = dt

    return F
