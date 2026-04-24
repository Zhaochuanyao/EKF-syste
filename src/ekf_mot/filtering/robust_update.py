"""
鲁棒更新模块：极端异常观测的两级处理

级别 1 - skip update：
  NIS > drop_threshold 且 score < low_score 时，跳过本次 EKF 更新，
  仅保留预测结果（x = x_pred, P = P_pred）

级别 2 - robust clip：
  未触发 skip 时，对新息向量做逐元素裁剪，抑制极端离群观测的冲击
  e_tilde_i = sign(e_i) * min(|e_i|, clip_delta)
"""

import numpy as np
from typing import Tuple


def should_skip_update(
    nis: float,
    drop_threshold: float,
    score: float,
    low_score: float,
) -> bool:
    """
    判断是否跳过本次 EKF 更新。

    仅当同时满足以下两个条件时跳过：
      - NIS > drop_threshold（极端异常）
      - score < low_score（检测质量差，非高质量误检）

    Args:
        nis:            当前步 NIS 值
        drop_threshold: 极端异常阈值
        score:          检测置信度 [0, 1]
        low_score:      低质量检测分数门限

    Returns:
        True 表示跳过更新，False 表示继续更新
    """
    return nis > drop_threshold and score < low_score


def apply_robust_clip(innov: np.ndarray, clip_delta: float) -> np.ndarray:
    """
    对新息向量逐元素做符号保留裁剪（Clip）。

    e_tilde_i = sign(e_i) * min(|e_i|, clip_delta)

    Args:
        innov:      原始新息向量，shape (4,)
        clip_delta: 裁剪阈值（像素）

    Returns:
        裁剪后的新息向量，与 innov 形状相同
    """
    return np.sign(innov) * np.minimum(np.abs(innov), clip_delta)


def robust_update_step(
    x: np.ndarray,
    P: np.ndarray,
    innov: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    clip_delta: float,
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用裁剪后的新息执行 EKF Joseph 形式更新。

    x_new = x + K * e_tilde
    P_new = (I - K*H) * P * (I - K*H)^T + K*R*K^T  （Joseph 形式）

    Args:
        x:         当前状态均值 (state_dim,)
        P:         当前状态协方差 (state_dim, state_dim)
        innov:     原始新息向量 (meas_dim,)
        K:         卡尔曼增益 (state_dim, meas_dim)
        R:         观测噪声矩阵 (meas_dim, meas_dim)
        H:         观测矩阵 (meas_dim, state_dim)
        clip_delta: 新息裁剪阈值
        state_dim:  状态维度（用于构造单位矩阵）

    Returns:
        (x_new, P_new)
    """
    e_tilde = apply_robust_clip(innov, clip_delta)
    x_new = x + K @ e_tilde

    I_KH = np.eye(state_dim) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_new = (P_new + P_new.T) / 2.0

    return x_new, P_new
