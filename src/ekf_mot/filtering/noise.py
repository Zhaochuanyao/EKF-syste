"""
噪声矩阵构造模块 - 过程噪声 Q 和观测噪声 R
"""

import numpy as np
from typing import Optional


def build_process_noise_Q(
    dt: float,
    std_acc: float = 1.0,
    std_yaw_rate: float = 0.3,
    std_size: float = 0.1,
    std_pos: float = 0.0,
    std_vel: float = 0.0,
    lost_age: int = 0,
    lost_age_q_scale: float = 1.5,
) -> np.ndarray:
    """
    构造 CTRV 过程噪声矩阵 Q (7x7)。

    使用解耦设计：
      - std_pos: 直接位置噪声（px/step），控制 EKF 对位置观测的响应度
      - std_vel: 直接速度噪声（px/s/step），控制速度估计的灵活度
      - std_acc: Singer model 加速度噪声，产生 cx-v 耦合项（对小 dt 影响极小，保留结构）
    """
    Q = np.zeros((7, 7), dtype=np.float64)

    q = std_acc ** 2
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4

    # Singer model：位置和速度的耦合项（dt 很小时数值极小，主要保留数学结构）
    Q[0, 0] = q * dt4 / 4
    Q[1, 1] = q * dt4 / 4
    Q[2, 2] = q * dt2
    Q[0, 2] = q * dt3 / 2
    Q[2, 0] = q * dt3 / 2
    Q[1, 2] = q * dt3 / 2
    Q[2, 1] = q * dt3 / 2

    # 直接位置噪声：保证稳态卡尔曼增益合理（K ≈ std_pos²/R_cx ≈ 0.1~0.3）
    if std_pos > 0.0:
        Q[0, 0] += std_pos ** 2
        Q[1, 1] += std_pos ** 2

    # 直接速度噪声：允许速度在每帧有合理变化，改善速度估计收敛
    if std_vel > 0.0:
        Q[2, 2] += std_vel ** 2

    # 航向角 theta 和角速度 omega
    q_w = std_yaw_rate ** 2
    Q[3, 3] = q_w * dt2
    Q[4, 4] = q_w
    Q[3, 4] = q_w * dt
    Q[4, 3] = q_w * dt

    # 尺寸 w, h
    q_s = std_size ** 2
    Q[5, 5] = q_s
    Q[6, 6] = q_s

    # Lost 轨迹：随丢失帧数指数放大位置/速度不确定性（最多 8x）
    if lost_age > 0:
        scale = min(float(lost_age_q_scale) ** lost_age, 8.0)
        for i, j in [(0,0),(1,1),(2,2),(0,2),(2,0),(1,2),(2,1)]:
            Q[i, j] *= scale

    return Q


def build_measurement_noise_R(
    std_cx: float = 6.0,
    std_cy: float = 6.0,
    std_w: float = 12.0,
    std_h: float = 12.0,
    score: Optional[float] = None,
    score_adaptive: bool = True,
    bbox_w: Optional[float] = None,
    bbox_h: Optional[float] = None,
    size_adaptive: bool = False,
    size_ref: float = 100.0,
    size_max_scale: float = 3.0,
    img_w: Optional[float] = None,
    img_h: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    border_adaptive: bool = False,
    border_margin: float = 20.0,
    border_scale: float = 2.0,
    aspect_ratio: Optional[float] = None,
    aspect_adaptive: bool = False,
    aspect_normal_range: tuple = (0.3, 3.0),
    aspect_scale: float = 1.5,
) -> np.ndarray:
    """构造观测噪声矩阵 R (4x4)，支持自适应缩放。"""
    scale = 1.0

    if score_adaptive and score is not None:
        scale *= 1.0 + (1.0 - score) * 1.5

    if size_adaptive and bbox_w is not None and bbox_h is not None:
        diag = float(np.sqrt(bbox_w ** 2 + bbox_h ** 2))
        if diag > size_ref:
            scale *= min(1.0 + (diag - size_ref) / size_ref, size_max_scale)

    if (border_adaptive and img_w is not None and img_h is not None
            and cx is not None and cy is not None):
        dist_to_border = min(cx, img_w - cx, cy, img_h - cy)
        if dist_to_border < border_margin:
            scale *= 1.0 + (border_scale - 1.0) * (1.0 - dist_to_border / border_margin)

    if aspect_adaptive and aspect_ratio is not None:
        lo, hi = aspect_normal_range
        if float(aspect_ratio) < lo or float(aspect_ratio) > hi:
            scale *= aspect_scale

    R = np.diag([std_cx**2, std_cy**2, std_w**2, std_h**2]).astype(np.float64)
    R *= scale
    return R


def build_initial_covariance_P(
    std_cx: float = 10.0,
    std_cy: float = 10.0,
    std_v: float = 5.0,
    std_theta: float = 1.0,
    std_omega: float = 0.5,
    std_w: float = 20.0,
    std_h: float = 20.0,
) -> np.ndarray:
    """构造初始状态协方差矩阵 P (7x7)。"""
    stds = [std_cx, std_cy, std_v, std_theta, std_omega, std_w, std_h]
    return np.diag([s**2 for s in stds]).astype(np.float64)
