"""
几何工具模块 - bbox 坐标转换与 IoU 计算
"""

import numpy as np
from typing import Tuple


def xyxy_to_cxcywh(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] -> [cx,cy,w,h]"""
    x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=-1)


def cxcywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """[cx,cy,w,h] -> [x1,y1,x2,y2]"""
    cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    计算两组 bbox 之间的 IoU 矩阵。

    Args:
        bboxes_a: (N, 4) xyxy 格式
        bboxes_b: (M, 4) xyxy 格式

    Returns:
        (N, M) IoU 矩阵
    """
    # 扩展维度以支持广播
    a = bboxes_a[:, np.newaxis, :]   # (N, 1, 4)
    b = bboxes_b[np.newaxis, :, :]   # (1, M, 4)

    # 交集
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 各自面积
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union_area = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


def bbox_area(bbox: np.ndarray) -> float:
    """计算单个 bbox 面积，xyxy 格式"""
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def clip_bbox(bbox: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """将 bbox 裁剪到图像边界内，xyxy 格式"""
    clipped = bbox.copy().astype(np.float64)
    clipped[0] = np.clip(clipped[0], 0, img_w)
    clipped[1] = np.clip(clipped[1], 0, img_h)
    clipped[2] = np.clip(clipped[2], 0, img_w)
    clipped[3] = np.clip(clipped[3], 0, img_h)
    return clipped


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi]"""
    import math
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
