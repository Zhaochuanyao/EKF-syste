"""
绘制协方差椭圆
"""

import math
from typing import Tuple
import cv2
import numpy as np

from ..core.constants import COLOR_COVARIANCE
from ..prediction.uncertainty import covariance_to_ellipse


def draw_covariance_ellipse(
    frame: np.ndarray,
    cx: float,
    cy: float,
    P: np.ndarray,
    n_std: float = 2.0,
    color: Tuple[int, int, int] = COLOR_COVARIANCE,
    thickness: int = 1,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    在帧上绘制协方差椭圆（半透明）。

    Args:
        frame: BGR 图像
        cx, cy: 椭圆中心
        P: 7x7 协方差矩阵
        n_std: 覆盖标准差倍数
        color: 颜色 (BGR)
        thickness: 线宽（-1 为填充）
        alpha: 透明度（0=完全透明，1=不透明）

    Returns:
        绘制后的图像
    """
    ellipse_params = covariance_to_ellipse(P, n_std)
    a = int(max(1, ellipse_params["a"]))
    b = int(max(1, ellipse_params["b"]))
    angle = ellipse_params["angle_deg"]

    center = (int(cx), int(cy))

    if alpha < 1.0 and thickness == -1:
        # 半透明填充
        overlay = frame.copy()
        cv2.ellipse(overlay, center, (a, b), angle, 0, 360, color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    else:
        cv2.ellipse(frame, center, (a, b), angle, 0, 360, color, thickness)

    return frame
