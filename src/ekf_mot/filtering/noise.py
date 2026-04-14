"""
噪声矩阵构造模块 - 过程噪声 Q 和观测噪声 R

支持多种自适应策略：
  - score_adaptive: 低置信度检测 → 更大 R
  - size_adaptive:  大目标框 → 更大 R（像素误差随尺寸增大）
  - border_adaptive: 靠近图像边界 → 更大 R（截断/遮挡更严重）
  - aspect_adaptive: 极端宽高比 → 更大 R（检测不稳定）
  - lost_age_adaptive (Q): Lost 轨迹随丢失时间增加过程噪声（位置扩散）
"""

import numpy as np
from typing import Optional


def build_process_noise_Q(
    dt: float,
    std_acc: float = 2.0,
    std_yaw_rate: float = 0.5,
    std_size: float = 0.1,
    lost_age: int = 0,
    lost_age_q_scale: float = 1.5,
) -> np.ndarray:
    """
    构造 CTRV 过程噪声矩阵 Q (7x7)。

    过程噪声来源：
    - 加速度扰动（影响 cx, cy, v）
    - 角速度变化（影响 theta, omega）
    - 尺寸变化（影响 w, h）
    - Lost 轨迹随丢失帧数增大不确定性（lost_age_adaptive）

    Args:
        dt: 时间步长
        std_acc: 加速度标准差
        std_yaw_rate: 角速度变化标准差
        std_size: 尺寸变化标准差
        lost_age: 轨迹已丢失的帧数（0 = 正常跟踪，>0 = 丢失状态）
        lost_age_q_scale: 每丢失一帧 Q 放大的基数（指数增长，如 1.5^lost_age）

    Returns:
        Q 矩阵，shape (7, 7)
    """
    Q = np.zeros((7, 7), dtype=np.float64)

    # ── 位置和速度方向的过程噪声 ──────────────────────────────
    q_v = std_acc ** 2
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4

    # cx 方向（索引0）
    Q[0, 0] = q_v * dt4 / 4
    # cy 方向（索引1）
    Q[1, 1] = q_v * dt4 / 4
    # v 方向（索引2）
    Q[2, 2] = q_v * dt2
    # cx-v 交叉项
    Q[0, 2] = q_v * dt3 / 2
    Q[2, 0] = q_v * dt3 / 2
    Q[1, 2] = q_v * dt3 / 2
    Q[2, 1] = q_v * dt3 / 2

    # ── 航向角和角速度方向的过程噪声 ─────────────────────────
    q_omega = std_yaw_rate ** 2
    Q[3, 3] = q_omega * dt2   # theta
    Q[4, 4] = q_omega         # omega
    Q[3, 4] = q_omega * dt
    Q[4, 3] = q_omega * dt

    # ── 尺寸方向的过程噪声 ────────────────────────────────────
    q_size = std_size ** 2
    Q[5, 5] = q_size  # w
    Q[6, 6] = q_size  # h

    # ── Lost 轨迹自适应：随丢失时间扩大位置/速度不确定性 ──────
    if lost_age > 0:
        # 指数放大，最多 8x
        scale = min(float(lost_age_q_scale) ** lost_age, 8.0)
        Q[0, 0] *= scale
        Q[1, 1] *= scale
        Q[2, 2] *= scale
        Q[0, 2] *= scale
        Q[2, 0] *= scale
        Q[1, 2] *= scale
        Q[2, 1] *= scale

    return Q


def build_measurement_noise_R(
    std_cx: float = 5.0,
    std_cy: float = 5.0,
    std_w: float = 10.0,
    std_h: float = 10.0,
    score: Optional[float] = None,
    score_adaptive: bool = True,
    # 新增自适应参数
    bbox_w: Optional[float] = None,
    bbox_h: Optional[float] = None,
    size_adaptive: bool = False,
    size_ref: float = 100.0,        # 参考尺寸（像素），超过此尺寸开始放大 R
    size_max_scale: float = 3.0,    # 尺寸自适应最大放大倍数
    img_w: Optional[float] = None,
    img_h: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    border_adaptive: bool = False,
    border_margin: float = 20.0,    # 距边界多少像素内认为是边界区域
    border_scale: float = 2.0,      # 边界区域 R 放大倍数
    aspect_ratio: Optional[float] = None,
    aspect_adaptive: bool = False,
    aspect_normal_range: tuple = (0.3, 3.0),  # 正常宽高比范围
    aspect_scale: float = 1.5,      # 超出范围时 R 放大倍数
) -> np.ndarray:
    """
    构造观测噪声矩阵 R (4x4)。

    支持多种自适应策略（各策略乘法叠加）：
    - score_adaptive: 低置信度 → 大 R
    - size_adaptive:  大目标 → 大 R（绝对像素误差随目标尺寸增大）
    - border_adaptive: 近边界 → 大 R（截断使检测不准）
    - aspect_adaptive: 极端宽高比 → 大 R（纵横比异常时检测不稳定）

    Returns:
        R 矩阵，shape (4, 4)
    """
    scale = 1.0

    # ── 置信度自适应 ──────────────────────────────────────────
    if score_adaptive and score is not None:
        # score=1.0 → scale=1.0，score=0.3 → scale≈2.4
        scale *= 1.0 + (1.0 - score) * 2.0

    # ── 尺寸自适应 ────────────────────────────────────────────
    if size_adaptive and bbox_w is not None and bbox_h is not None:
        # 以目标对角线长度衡量目标大小
        diag = float(np.sqrt(bbox_w ** 2 + bbox_h ** 2))
        if diag > size_ref:
            size_s = min(1.0 + (diag - size_ref) / size_ref, size_max_scale)
            scale *= size_s

    # ── 边界自适应 ────────────────────────────────────────────
    if (border_adaptive and img_w is not None and img_h is not None
            and cx is not None and cy is not None):
        dist_to_border = min(cx, img_w - cx, cy, img_h - cy)
        if dist_to_border < border_margin:
            border_s = 1.0 + (border_scale - 1.0) * (1.0 - dist_to_border / border_margin)
            scale *= border_s

    # ── 宽高比自适应 ──────────────────────────────────────────
    if aspect_adaptive and aspect_ratio is not None:
        ar = float(aspect_ratio)
        lo, hi = aspect_normal_range
        if ar < lo or ar > hi:
            scale *= aspect_scale

    R = np.diag([std_cx**2, std_cy**2, std_w**2, std_h**2]).astype(np.float64)
    R *= scale
    return R


def build_initial_covariance_P(
    std_cx: float = 10.0,
    std_cy: float = 10.0,
    std_v: float = 5.0,
    std_theta: float = 0.5,
    std_omega: float = 0.2,
    std_w: float = 20.0,
    std_h: float = 20.0,
) -> np.ndarray:
    """
    构造初始状态协方差矩阵 P (7x7)。

    初始时对速度、角度等不可观测量设置较大的不确定性。

    Returns:
        P 矩阵，shape (7, 7)
    """
    stds = [std_cx, std_cy, std_v, std_theta, std_omega, std_w, std_h]
    P = np.diag([s**2 for s in stds]).astype(np.float64)
    return P
