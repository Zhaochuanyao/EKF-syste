"""
绘制未来预测轨迹
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np

from ..core.constants import COLOR_FUTURE


def draw_future_trajectory(
    frame: np.ndarray,
    current_center: Tuple[float, float],
    future_points: Dict[int, Tuple[float, float]],
    color: Tuple[int, int, int] = COLOR_FUTURE,
    radius: int = 4,
    thickness: int = 1,
) -> np.ndarray:
    """
    绘制单条轨迹的未来预测点和连线。

    Args:
        frame: BGR 图像
        current_center: 当前中心点 (cx, cy)
        future_points: {step: (cx, cy)} 预测点字典
        color: 颜色 (BGR)
        radius: 预测点圆半径
        thickness: 连线线宽

    Returns:
        绘制后的图像
    """
    if not future_points:
        return frame

    # 按步数排序
    sorted_steps = sorted(future_points.keys())
    pts = [current_center] + [future_points[s] for s in sorted_steps]

    # 绘制连线（虚线效果：每隔一段绘制）
    for i in range(1, len(pts)):
        pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        pt2 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 绘制预测点
    for step, (cx, cy) in future_points.items():
        pt = (int(cx), int(cy))
        cv2.circle(frame, pt, radius, color, -1)
        # 标注步数
        cv2.putText(
            frame, f"+{step}", (pt[0] + 5, pt[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
        )

    return frame
