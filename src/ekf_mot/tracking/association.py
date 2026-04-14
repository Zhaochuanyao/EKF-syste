"""
数据关联模块 - 三阶段匈牙利匹配

关联策略（三阶段）:
  Stage A: Confirmed + Lost 轨迹 × 高置信度检测框
           使用 IoU + Mahalanobis + 中心距离融合代价，严格门控
  Stage B: 未匹配的 Confirmed + Lost × 低置信度检测框
           仅用 IoU，阈值相对宽松（找回弱观测）
  Stage C: Tentative × Stage A 中未匹配的高置信度检测框
           IoU 主导，轻量匹配（Tentative 轨迹不参与 Mahal 门控）

索引约定: 所有 matches / unmatched_tracks / unmatched_dets
均为相对于输入 tracks 和 detections 列表的原始索引。
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from .track import Track
from .cost import iou_cost_matrix, fused_cost_matrix
from ..core.types import Detection


def hungarian_match(
    cost_matrix: np.ndarray,
    threshold: float = 1.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    用匈牙利算法求解最优匹配。

    Args:
        cost_matrix: (N, M) 代价矩阵，inf 表示不可匹配
        threshold: 代价阈值，超过则视为未匹配

    Returns:
        (matches, unmatched_tracks, unmatched_dets)
        - matches: [(track_idx, det_idx), ...]  (局部索引)
        - unmatched_tracks: 未匹配的轨迹局部索引列表
        - unmatched_dets: 未匹配的检测局部索引列表
    """
    if cost_matrix.size == 0:
        n_tracks, n_dets = cost_matrix.shape
        return [], list(range(n_tracks)), list(range(n_dets))

    # 将 inf 替换为大数，避免 scipy 报错
    cost_finite = np.where(np.isinf(cost_matrix), 1e9, cost_matrix)

    row_ind, col_ind = linear_sum_assignment(cost_finite)

    matches = []
    unmatched_tracks = list(range(cost_matrix.shape[0]))
    unmatched_dets = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > threshold:
            continue  # 代价超过阈值，视为未匹配
        matches.append((int(r), int(c)))
        unmatched_tracks.remove(r)
        unmatched_dets.remove(c)

    return matches, unmatched_tracks, unmatched_dets


def associate(
    tracks: List[Track],
    detections: List[Detection],
    # ── 置信度分界线 ──────────────────────────────────────────
    high_conf_threshold: float = 0.5,
    low_conf_threshold: float = 0.1,
    # ── Stage A 参数：(Confirmed+Lost) × 高置信度检测框 ───────
    gating_threshold_confirmed: float = 9.4877,
    iou_weight: float = 0.4,
    mahal_weight: float = 0.4,
    center_weight: float = 0.2,
    cost_threshold_a: float = 0.8,
    center_norm: float = 200.0,
    # ── Stage B 参数：未匹配轨迹 × 低置信度检测框 ─────────────
    iou_threshold_b: float = 0.4,
    # ── Stage C 参数：Tentative × 未匹配高置信度检测框 ─────────
    iou_threshold_c: float = 0.3,
    # ── 开关 ──────────────────────────────────────────────────
    second_stage: bool = True,
    # ── 向后兼容参数（映射到新参数）────────────────────────────
    iou_threshold: float = None,
    gating_threshold: float = None,
    second_stage_conf: float = None,
    second_stage_iou_threshold: float = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    三阶段数据关联。

    返回值索引均相对于输入的 tracks 和 detections 列表（原始索引）。
    """
    # ── 向后兼容处理 ──────────────────────────────────────────
    if second_stage_conf is not None:
        low_conf_threshold = second_stage_conf
    if second_stage_iou_threshold is not None:
        iou_threshold_b = second_stage_iou_threshold
    if iou_threshold is not None:
        iou_threshold_c = iou_threshold
    if gating_threshold is not None:
        gating_threshold_confirmed = gating_threshold

    n_tracks = len(tracks)
    n_dets = len(detections)

    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    # ── 按轨迹状态分类 ────────────────────────────────────────
    confirmed_set = {i for i, t in enumerate(tracks) if t.is_confirmed}
    lost_set = {i for i, t in enumerate(tracks) if t.is_lost}
    tentative_set = {i for i, t in enumerate(tracks) if t.is_tentative}

    # ── 按检测置信度分类 ──────────────────────────────────────
    high_dets_set = {i for i, d in enumerate(detections) if d.score >= high_conf_threshold}
    low_dets_set = {
        i for i, d in enumerate(detections)
        if low_conf_threshold <= d.score < high_conf_threshold
    }

    matched_tracks: set = set()
    matched_dets: set = set()
    all_matches: List[Tuple[int, int]] = []

    # ══════════════════════════════════════════════════════════
    # Stage A: (Confirmed + Lost) × 高置信度检测框
    #          融合代价：IoU + Mahalanobis + 中心距离
    # ══════════════════════════════════════════════════════════
    a_track_list = sorted(confirmed_set | lost_set)
    a_det_list = sorted(high_dets_set)

    unmatched_a_high_dets: set = set(a_det_list)
    unmatched_a_tracks: set = set(a_track_list)

    if a_track_list and a_det_list:
        a_tracks = [tracks[i] for i in a_track_list]
        a_dets = [detections[i] for i in a_det_list]

        cost_a = fused_cost_matrix(
            a_tracks, a_dets,
            iou_weight=iou_weight,
            mahal_weight=mahal_weight,
            center_weight=center_weight,
            gating_threshold=gating_threshold_confirmed,
            center_norm=center_norm,
            check_class=True,
        )

        matches_a_local, unmatched_a_t_local, unmatched_a_d_local = hungarian_match(
            cost_a, threshold=cost_threshold_a
        )

        for r_l, c_l in matches_a_local:
            r = a_track_list[r_l]
            c = a_det_list[c_l]
            all_matches.append((r, c))
            matched_tracks.add(r)
            matched_dets.add(c)

        unmatched_a_tracks = {a_track_list[i] for i in unmatched_a_t_local}
        unmatched_a_high_dets = {a_det_list[i] for i in unmatched_a_d_local}

    # ══════════════════════════════════════════════════════════
    # Stage B: 未匹配的 (Confirmed + Lost) × 低置信度检测框
    #          仅 IoU，阈值放宽（cost < 1 - iou_threshold_b）
    # ══════════════════════════════════════════════════════════
    b_track_list = sorted(unmatched_a_tracks)
    b_det_list = sorted(low_dets_set)

    if b_track_list and b_det_list and second_stage:
        b_tracks = [tracks[i] for i in b_track_list]
        b_dets = [detections[i] for i in b_det_list]

        cost_b = iou_cost_matrix(b_tracks, b_dets)

        matches_b_local, _unmatched_b_t_local, _unmatched_b_d_local = hungarian_match(
            cost_b, threshold=1.0 - iou_threshold_b
        )

        for r_l, c_l in matches_b_local:
            r = b_track_list[r_l]
            c = b_det_list[c_l]
            all_matches.append((r, c))
            matched_tracks.add(r)
            matched_dets.add(c)

    # ══════════════════════════════════════════════════════════
    # Stage C: Tentative × 未匹配的高置信度检测框
    #          仅 IoU，轻量匹配
    # ══════════════════════════════════════════════════════════
    c_track_list = sorted(tentative_set)
    c_det_list = sorted(unmatched_a_high_dets - matched_dets)

    if c_track_list and c_det_list:
        c_tracks = [tracks[i] for i in c_track_list]
        c_dets = [detections[i] for i in c_det_list]

        cost_c = iou_cost_matrix(c_tracks, c_dets)

        matches_c_local, _unmatched_c_t_local, _unmatched_c_d_local = hungarian_match(
            cost_c, threshold=1.0 - iou_threshold_c
        )

        for r_l, c_l in matches_c_local:
            r = c_track_list[r_l]
            c = c_det_list[c_l]
            all_matches.append((r, c))
            matched_tracks.add(r)
            matched_dets.add(c)

    # ── 汇总未匹配结果 ────────────────────────────────────────
    all_unmatched_tracks = sorted(set(range(n_tracks)) - matched_tracks)
    all_unmatched_dets = sorted(set(range(n_dets)) - matched_dets)

    return all_matches, all_unmatched_tracks, all_unmatched_dets
