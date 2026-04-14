"""
绘制检测框
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from ..core.types import Detection
from ..core.constants import COLOR_CONFIRMED


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    color: Tuple[int, int, int] = (200, 200, 200),
    thickness: int = 1,
    draw_score: bool = True,
    font_scale: float = 0.4,
) -> np.ndarray:
    """
    在帧上绘制检测框（浅色，区别于轨迹框）。

    Args:
        frame: BGR 图像
        detections: 检测结果列表
        color: 框颜色 (BGR)
        thickness: 线宽
        draw_score: 是否显示置信度

    Returns:
        绘制后的图像（原地修改）
    """
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if draw_score:
            label = f"{det.class_name} {det.score:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1,
            )
    return frame


def draw_track_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    track_id: int,
    class_name: str,
    score: float,
    color: Tuple[int, int, int] = COLOR_CONFIRMED,
    thickness: int = 2,
    draw_id: bool = True,
    draw_score: bool = True,
    font_scale: float = 0.5,
) -> np.ndarray:
    """绘制单条轨迹的检测框和标签"""
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if draw_id or draw_score:
        parts = []
        if draw_id:
            parts.append(f"ID:{track_id}")
        if draw_score:
            parts.append(f"{score:.2f}")
        label = " ".join(parts)

        # 背景矩形
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1,
        )
    return frame
