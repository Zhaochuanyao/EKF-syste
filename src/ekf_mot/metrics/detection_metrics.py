"""
检测评估指标计算模块

支持：
  - 单帧 IoU 匹配，统计 TP / FP / FN
  - 多帧累积后计算 Precision / Recall / F1
  - 简化版 AP50（基于置信度排序的精确率-召回率曲线面积）
  - 可选按类别分别统计

算法说明：
  对每帧预测结果，按置信度从高到低排序，依次与未匹配 GT 框计算 IoU，
  IoU > iou_threshold 则匹配成功（TP），否则为 FP；
  未被匹配的 GT 框计为 FN。
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from ..core.types import Detection


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    计算两个边界框的 IoU（Intersection over Union）。

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        IoU 值 [0, 1]
    """
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def _iou_matrix(preds: List[Detection], gts: List[Detection]) -> np.ndarray:
    """
    计算 N×M 的 IoU 矩阵。

    Args:
        preds: 预测框列表（N 个）
        gts:   真实框列表（M 个）

    Returns:
        iou_mat: shape (N, M)，iou_mat[i, j] 为第 i 个预测与第 j 个 GT 的 IoU
    """
    n, m = len(preds), len(gts)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)
    mat = np.zeros((n, m), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            mat[i, j] = _compute_iou(p.bbox, g.bbox)
    return mat


def match_detections(
    preds: List[Detection],
    gts: List[Detection],
    iou_threshold: float = 0.5,
    match_class: bool = False,
) -> Tuple[int, int, int, List[Tuple[int, int]]]:
    """
    对单帧执行贪婪 IoU 匹配（置信度优先）。

    匹配策略：
      1. 按置信度从高到低排序预测框
      2. 依次为每个预测框找 IoU 最大的未匹配 GT
      3. IoU >= iou_threshold 则匹配（TP），否则 FP
      4. 未被匹配的 GT 计为 FN

    Args:
        preds:          预测框列表
        gts:            真实框列表
        iou_threshold:  IoU 匹配阈值（默认 0.5，即 AP50）
        match_class:    是否要求类别一致才能匹配

    Returns:
        (tp, fp, fn, matched_pairs)
        matched_pairs: [(pred_idx, gt_idx), ...]，原始索引
    """
    if not preds and not gts:
        return 0, 0, 0, []
    if not preds:
        return 0, 0, len(gts), []
    if not gts:
        return 0, len(preds), 0, []

    # 按置信度降序排序预测框（保留原始索引）
    sorted_pred_idx = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)

    iou_mat = _iou_matrix(preds, gts)

    matched_gt: set = set()
    matched_pairs: List[Tuple[int, int]] = []
    tp = 0

    for pred_i in sorted_pred_idx:
        best_iou = iou_threshold - 1e-9  # 必须严格大于阈值
        best_gt_j = -1

        for gt_j in range(len(gts)):
            if gt_j in matched_gt:
                continue
            if match_class and preds[pred_i].class_id != gts[gt_j].class_id:
                continue
            iou_val = iou_mat[pred_i, gt_j]
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_j = gt_j

        if best_gt_j >= 0:
            tp += 1
            matched_gt.add(best_gt_j)
            matched_pairs.append((pred_i, best_gt_j))

    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, matched_pairs


# ─────────────────────────────────────────────────────────────
# AP50 计算
# ─────────────────────────────────────────────────────────────

def compute_ap(
    all_preds: List[Detection],
    all_gts: List[Detection],
    iou_threshold: float = 0.5,
    match_class: bool = False,
) -> float:
    """
    简化版 AP（Average Precision）计算。

    将所有帧的预测结果合并，按置信度降序排列，
    逐步计算精确率-召回率曲线，取 11 点插值面积。

    注意：此实现假设所有帧的 GT 已合并为一个大列表，
    适合序列级 AP 计算。帧内同一 GT 不能被多个预测重复匹配。

    Args:
        all_preds:      所有帧的预测框列表（需含 frame_id）
        all_gts:        所有帧的真实框列表（需含 frame_id）
        iou_threshold:  IoU 阈值（0.5 = AP50）
        match_class:    是否要求类别一致

    Returns:
        AP 值 [0, 1]
    """
    if not all_gts:
        return 0.0
    if not all_preds:
        return 0.0

    # 按帧分组
    from collections import defaultdict
    gt_by_frame: Dict[int, List[Detection]] = defaultdict(list)
    for g in all_gts:
        gt_by_frame[g.frame_id].append(g)

    # 按置信度降序排列所有预测框
    sorted_preds = sorted(all_preds, key=lambda p: p.score, reverse=True)

    # 已匹配的 GT（用 (frame_id, gt_idx) 标识）
    matched_gt_set: set = set()

    tp_list = []
    fp_list = []

    for pred in sorted_preds:
        frame_gts = gt_by_frame.get(pred.frame_id, [])
        best_iou = iou_threshold - 1e-9
        best_gt_key = None

        for gt_j, gt in enumerate(frame_gts):
            key = (pred.frame_id, gt_j)
            if key in matched_gt_set:
                continue
            if match_class and pred.class_id != gt.class_id:
                continue
            iou_val = _compute_iou(pred.bbox, gt.bbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_key = key

        if best_gt_key is not None:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt_set.add(best_gt_key)
        else:
            tp_list.append(0)
            fp_list.append(1)

    # 累积 TP/FP
    tp_cum = np.cumsum(tp_list, dtype=np.float64)
    fp_cum = np.cumsum(fp_list, dtype=np.float64)
    total_gts = len(all_gts)

    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall_curve = tp_cum / (total_gts + 1e-9)

    # 11 点插值 AP
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precision_curve[recall_curve >= thr]
        ap += (prec_at_thr.max() if len(prec_at_thr) > 0 else 0.0)
    ap /= 11.0

    return float(ap)


# ─────────────────────────────────────────────────────────────
# 多帧累积评估器
# ─────────────────────────────────────────────────────────────

class DetectionMetrics:
    """
    多帧累积检测评估器。

    用法：
        metrics = DetectionMetrics(iou_threshold=0.5)
        for preds, gts in frame_pairs:
            metrics.update(preds, gts)
        result = metrics.compute()
        # result = {"precision": ..., "recall": ..., "f1": ..., "ap50": ...}
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        match_class: bool = False,
    ) -> None:
        """
        Args:
            iou_threshold:  IoU 匹配阈值（默认 0.5）
            match_class:    是否要求类别一致才能匹配
        """
        self.iou_threshold = iou_threshold
        self.match_class = match_class
        self._tp = 0
        self._fp = 0
        self._fn = 0
        # 用于 AP 计算的全量列表
        self._all_preds: List[Detection] = []
        self._all_gts: List[Detection] = []
        self._frame_counter = 0

    def update(self, preds: List[Detection], gts: List[Detection]) -> Dict[str, int]:
        """
        更新一帧的统计。

        Args:
            preds: 当前帧的预测框列表
            gts:   当前帧的真实框列表

        Returns:
            当前帧的 {"tp": ..., "fp": ..., "fn": ...}
        """
        # 确保 frame_id 存在（用于 AP 计算时的帧内去重）
        frame_id = self._frame_counter
        for p in preds:
            p.frame_id = frame_id
        for g in gts:
            g.frame_id = frame_id
        self._frame_counter += 1

        tp, fp, fn, _ = match_detections(
            preds, gts,
            iou_threshold=self.iou_threshold,
            match_class=self.match_class,
        )
        self._tp += tp
        self._fp += fp
        self._fn += fn
        self._all_preds.extend(preds)
        self._all_gts.extend(gts)
        return {"tp": tp, "fp": fp, "fn": fn}

    def compute(self) -> Dict[str, float]:
        """
        计算累积指标。

        Returns:
            {
              "precision": float,
              "recall": float,
              "f1": float,
              "ap50": float,        # 简化版 AP50
              "tp": int,
              "fp": int,
              "fn": int,
              "num_frames": int,
            }
        """
        tp, fp, fn = self._tp, self._fp, self._fn
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        ap50 = compute_ap(
            self._all_preds,
            self._all_gts,
            iou_threshold=self.iou_threshold,
            match_class=self.match_class,
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "ap50": round(ap50, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_frames": self._frame_counter,
        }

    def reset(self) -> None:
        """重置所有统计"""
        self._tp = self._fp = self._fn = 0
        self._all_preds.clear()
        self._all_gts.clear()
        self._frame_counter = 0

    def summary(self) -> str:
        """返回格式化的指标摘要字符串"""
        r = self.compute()
        return (
            f"Precision={r['precision']:.4f}  "
            f"Recall={r['recall']:.4f}  "
            f"F1={r['f1']:.4f}  "
            f"AP50={r['ap50']:.4f}  "
            f"(TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, "
            f"Frames={r['num_frames']})"
        )
