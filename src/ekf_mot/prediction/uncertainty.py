"""
不确定性估计 - 从协方差矩阵提取椭圆参数
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple

from ..core.constants import IDX_CX, IDX_CY


def covariance_to_ellipse(
    P: np.ndarray,
    n_std: float = 2.0,
) -> Dict[str, float]:
    """
    从 7x7 协方差矩阵提取位置分量的椭圆参数。

    提取 P 的 cx-cy 子矩阵（2x2），计算特征值和特征向量，
    得到椭圆的长轴、短轴和旋转角。

    Args:
        P: 7x7 协方差矩阵
        n_std: 椭圆覆盖的标准差倍数（2.0 ≈ 95% 置信区间）

    Returns:
        {
            "cx": 中心x,
            "cy": 中心y,
            "a": 长半轴（像素）,
            "b": 短半轴（像素）,
            "angle_deg": 旋转角（度）
        }
    """
    # 提取位置协方差子矩阵
    P_pos = P[np.ix_([IDX_CX, IDX_CY], [IDX_CX, IDX_CY])]  # (2, 2)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(P_pos)

    # 确保特征值非负（数值稳定性）
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # 长轴对应最大特征值
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 椭圆半轴长度 = n_std * sqrt(eigenvalue)
    a = n_std * math.sqrt(eigenvalues[0])
    b = n_std * math.sqrt(eigenvalues[1])

    # 旋转角（长轴方向）
    angle_rad = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = math.degrees(angle_rad)

    return {"a": a, "b": b, "angle_deg": angle_deg}
