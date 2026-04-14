"""
雅可比矩阵计算模块
包含 CTRV 状态转移雅可比和观测雅可比
"""

import numpy as np
from ..core.constants import (
    IDX_CX, IDX_CY, IDX_W, IDX_H,
    STATE_DIM, MEAS_DIM,
)


def observation_jacobian() -> np.ndarray:
    """
    观测函数 h(x) = [cx, cy, w, h] 关于状态向量的雅可比矩阵 H。

    观测函数是线性的：h(x) = H * x
    其中 H 是 4x7 矩阵，直接提取 cx, cy, w, h 分量。

    Returns:
        H 矩阵，shape (4, 7)
    """
    H = np.zeros((MEAS_DIM, STATE_DIM), dtype=np.float64)
    # z[0] = cx = x[0]
    H[0, IDX_CX] = 1.0
    # z[1] = cy = x[1]
    H[1, IDX_CY] = 1.0
    # z[2] = w  = x[5]
    H[2, IDX_W] = 1.0
    # z[3] = h  = x[6]
    H[3, IDX_H] = 1.0
    return H


# 观测雅可比是常数矩阵，预计算一次
H_MATRIX = observation_jacobian()
