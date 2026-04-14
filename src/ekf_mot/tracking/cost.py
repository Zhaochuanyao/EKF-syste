"""
代价矩阵计算模块 - IoU、Mahalanobis、中心距离、融合代价
"""

from typing import List, Optional
import numpy as np

from .track import Track
from ..core.types import Detection
from ..filtering.gating import gating_distance_batch
from ..utils.geometry import iou_batch


def iou_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
) -> np.ndarray:
    """
    计算轨迹与检测框之间的 IoU 代价矩阵。
    代价 = 1 - IoU，范围 [0, 1]。
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float64)

    track_bboxes = np.array([t.get_bbox() for t in tracks])      # (N, 4)
    det_bboxes = np.array([d.bbox for d in detections])           # (M, 4)

    iou_mat = iou_batch(track_bboxes, det_bboxes)                 # (N, M)
    return 1.0 - iou_mat


def mahalanobis_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    gating_threshold: float = 9.4877,
) -> np.ndarray:
    """
    计算轨迹与检测框之间的 Mahalanobis 代价矩阵。
    超过门控阈值的位置设为 inf（不可关联）。
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float64)

    det_measurements = np.array([d.to_measurement() for d in detections])  # (M, 4)
    cost = np.full((len(tracks), len(detections)), np.inf, dtype=np.float64)

    for i, track in enumerate(tracks):
        z_pred = track.get_predicted_measurement()  # (4,)
        S = track.get_innovation_covariance()       # (4, 4)

        dists = gating_distance_batch(det_measurements, z_pred, S)  # (M,)

        valid = dists <= gating_threshold
        cost[i, valid] = dists[valid]

    return cost


def center_distance_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    normalize_factor: float = 200.0,
) -> np.ndarray:
    """
    计算轨迹预测中心与检测中心之间的归一化欧氏距离代价矩阵。

    代价 = dist(track_center, det_center) / normalize_factor。
    normalize_factor 建议设为帧对角线长度的 1/4 左右，
    使代价范围大致在 [0, 1]。

    Args:
        tracks: 轨迹列表 (N)
        detections: 检测列表 (M)
        normalize_factor: 归一化因子（像素），控制代价量级

    Returns:
        代价矩阵 (N, M)
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float64)

    track_centers = np.array([t.get_center() for t in tracks], dtype=np.float64)  # (N, 2)
    det_centers = np.array([[d.cx, d.cy] for d in detections], dtype=np.float64)  # (M, 2)

    diff = track_centers[:, None, :] - det_centers[None, :, :]  # (N, M, 2)
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))                  # (N, M)

    return dists / max(normalize_factor, 1.0)


def fused_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    iou_weight: float = 0.4,
    mahal_weight: float = 0.4,
    center_weight: float = 0.2,
    gating_threshold: float = 9.4877,
    center_norm: float = 200.0,
    check_class: bool = True,
    max_size_ratio: Optional[float] = 3.0,
) -> np.ndarray:
    """
    融合 IoU 代价、Mahalanobis 代价和中心距离代价。

    融合策略:
      cost = iou_weight * (1-IoU) + mahal_weight * norm_mahal + center_weight * norm_center

    额外约束（设为 inf）：
      - Mahalanobis 超出门控阈值（超出 EKF 预期运动范围）
      - 类别不匹配
      - bbox 尺寸突变超过 max_size_ratio

    Args:
        tracks: 轨迹列表 (N)
        detections: 检测列表 (M)
        iou_weight: IoU 代价权重（建议 0.3-0.5）
        mahal_weight: Mahalanobis 代价权重（建议 0.3-0.5）
        center_weight: 中心距离代价权重（建议 0.1-0.3）
        gating_threshold: Mahalanobis 门控阈值
        center_norm: 中心距离归一化因子（像素）
        check_class: 是否拒绝类别不匹配的匹配
        max_size_ratio: 检测框与轨迹框尺寸比最大允许值，None 则不检查

    Returns:
        融合代价矩阵 (N, M)
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float64)

    iou_cost = iou_cost_matrix(tracks, detections)
    mahal_cost = mahalanobis_cost_matrix(tracks, detections, gating_threshold)
    center_cost = center_distance_cost_matrix(tracks, detections, center_norm)

    # 归一化 Mahalanobis 到 [0, 1]
    mahal_normalized = np.where(
        np.isinf(mahal_cost),
        np.inf,
        mahal_cost / gating_threshold,
    )

    # 融合：Mahalanobis 超门控 → inf
    fused = np.where(
        np.isinf(mahal_normalized),
        np.inf,
        iou_weight * iou_cost
        + mahal_weight * mahal_normalized
        + center_weight * np.clip(center_cost, 0, 2.0),  # 限制 center 贡献
    )

    # ── 类别一致性约束 ─────────────────────────────────────────
    if check_class:
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                if track.class_id != det.class_id:
                    fused[i, j] = np.inf

    # ── bbox 尺寸突变约束 ──────────────────────────────────────
    if max_size_ratio is not None:
        track_bboxes = np.array([t.get_bbox() for t in tracks])  # (N, 4)
        det_bboxes = np.array([d.bbox for d in detections])       # (M, 4)

        track_w = track_bboxes[:, 2] - track_bboxes[:, 0]  # (N,)
        track_h = track_bboxes[:, 3] - track_bboxes[:, 1]  # (N,)
        det_w = det_bboxes[:, 2] - det_bboxes[:, 0]        # (M,)
        det_h = det_bboxes[:, 3] - det_bboxes[:, 1]        # (M,)

        # 宽度比和高度比 (N, M)
        w_ratio = np.maximum(
            track_w[:, None] / np.maximum(det_w[None, :], 1.0),
            det_w[None, :] / np.maximum(track_w[:, None], 1.0),
        )
        h_ratio = np.maximum(
            track_h[:, None] / np.maximum(det_h[None, :], 1.0),
            det_h[None, :] / np.maximum(track_h[:, None], 1.0),
        )

        size_jump = (w_ratio > max_size_ratio) | (h_ratio > max_size_ratio)
        fused[size_jump] = np.inf

    return fused
