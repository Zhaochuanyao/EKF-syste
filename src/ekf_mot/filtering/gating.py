"""
门控模块 - Mahalanobis 距离计算与卡方门控
"""

import numpy as np
from scipy.stats import chi2
from typing import List, Optional

from ..core.constants import MEAS_DIM, DEFAULT_GATING_THRESHOLD


def mahalanobis_distance(
    z: np.ndarray,
    z_pred: np.ndarray,
    S: np.ndarray,
) -> float:
    """
    计算观测值 z 与预测观测 z_pred 之间的 Mahalanobis 距离。

    d^2 = (z - z_pred)^T * S^{-1} * (z - z_pred)

    Args:
        z: 实际观测向量 (4,)
        z_pred: 预测观测向量 (4,)
        S: 观测协方差矩阵 (4, 4)

    Returns:
        Mahalanobis 距离的平方（标量）
    """
    diff = z - z_pred
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # S 奇异时使用伪逆
        S_inv = np.linalg.pinv(S)
    return float(diff @ S_inv @ diff)


def gating_distance_batch(
    measurements: np.ndarray,
    z_pred: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    批量计算多个观测值与预测观测的 Mahalanobis 距离。

    Args:
        measurements: (N, 4) 观测矩阵
        z_pred: (4,) 预测观测向量
        S: (4, 4) 观测协方差矩阵

    Returns:
        (N,) Mahalanobis 距离平方数组
    """
    diff = measurements - z_pred[np.newaxis, :]  # (N, 4)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    # d^2[i] = diff[i] @ S_inv @ diff[i]
    return np.einsum("ni,ij,nj->n", diff, S_inv, diff)


def chi2_gate(
    distance_sq: float,
    df: int = MEAS_DIM,
    confidence: float = 0.95,
) -> bool:
    """
    卡方门控：判断 Mahalanobis 距离是否在置信区间内。

    Args:
        distance_sq: Mahalanobis 距离的平方
        df: 自由度（等于观测维度）
        confidence: 置信水平

    Returns:
        True 表示通过门控（距离在置信区间内）
    """
    threshold = chi2.ppf(confidence, df=df)
    return distance_sq <= threshold


def get_gating_threshold(df: int = MEAS_DIM, confidence: float = 0.95) -> float:
    """获取卡方门控阈值"""
    return float(chi2.ppf(confidence, df=df))
