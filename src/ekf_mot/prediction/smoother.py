"""
轨迹平滑工具 - 移动平均平滑
"""

from typing import List, Tuple
import numpy as np


def moving_average_smooth(
    points: List[Tuple[float, float]],
    window: int = 5,
) -> List[Tuple[float, float]]:
    """
    对历史轨迹点进行移动平均平滑。

    Args:
        points: [(cx, cy), ...] 历史轨迹点
        window: 平滑窗口大小

    Returns:
        平滑后的轨迹点列表
    """
    if len(points) < 2:
        return points

    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    smoothed = np.zeros_like(pts)

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        smoothed[i] = pts[start:end].mean(axis=0)

    return [(float(p[0]), float(p[1])) for p in smoothed]
