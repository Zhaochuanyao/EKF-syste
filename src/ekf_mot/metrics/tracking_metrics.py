"""
跟踪指标 - MOTA、MOTP、ID Switch 完整帧级评估
"""

from typing import Dict, List, Tuple
import numpy as np


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """计算两个 [x1,y1,x2,y2] 边界框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _iou_matrix(
    preds: List[List[float]],
    gts: List[List[float]],
) -> np.ndarray:
    """计算预测框与 GT 框的 IoU 矩阵（N×M）"""
    n, m = len(preds), len(gts)
    mat = np.zeros((n, m), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            mat[i, j] = _compute_iou(p, g)
    return mat


# ══════════════════════════════════════════════════════════════
# TrackingEvaluator — 帧级 IoU 匹配 + ID Switch 检测
# ══════════════════════════════════════════════════════════════

class TrackingEvaluator:
    """
    基于帧级 IoU 匹配的多目标跟踪评估器。

    指标定义
    --------
    MOTA = 1 - (FN + FP + IDSW) / total_GT
    MOTP = 匹配帧对 IoU 的均值（越高越好）

    用法::

        evaluator = TrackingEvaluator(iou_threshold=0.5)
        for frame_id, pred_tracks, gt_annotations in frames:
            pred_bboxes = [(t.track_id, t.get_bbox().tolist()) for t in pred_tracks]
            gt_bboxes   = [(a["id"], a["bbox"]) for a in gt_annotations]
            evaluator.update(pred_bboxes, gt_bboxes)
        report = evaluator.compute()
        evaluator.print_summary()
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._gt_to_track: Dict[int, int] = {}    # gt_id → 上一帧匹配的 track_id
        self._track_lengths: Dict[int, int] = {}  # track_id → 出现帧数
        self._total_gt = 0
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._id_switches = 0
        self._iou_sum = 0.0
        self._matched_count = 0
        self._num_frames = 0

    def update(
        self,
        pred_bboxes: List[Tuple[int, List[float]]],
        gt_bboxes: List[Tuple[int, List[float]]],
    ) -> Dict[str, int]:
        """
        更新一帧的跟踪统计。

        Args:
            pred_bboxes: [(track_id, [x1,y1,x2,y2]), ...]  预测轨迹框
            gt_bboxes:   [(gt_id,    [x1,y1,x2,y2]), ...]  GT 标注框

        Returns:
            当帧统计字典 {tp, fp, fn, id_switches}
        """
        self._num_frames += 1
        self._total_gt += len(gt_bboxes)

        # 更新 track_lengths
        for tid, _ in pred_bboxes:
            self._track_lengths[tid] = self._track_lengths.get(tid, 0) + 1

        # 空帧快速路径
        if not pred_bboxes and not gt_bboxes:
            return {"tp": 0, "fp": 0, "fn": 0, "id_switches": 0}
        if not pred_bboxes:
            fn = len(gt_bboxes)
            self._fn += fn
            return {"tp": 0, "fp": 0, "fn": fn, "id_switches": 0}
        if not gt_bboxes:
            fp = len(pred_bboxes)
            self._fp += fp
            return {"tp": 0, "fp": fp, "fn": 0, "id_switches": 0}

        # IoU 矩阵 + 贪心匹配（按 IoU 降序）
        pred_ids = [p[0] for p in pred_bboxes]
        pred_boxes = [p[1] for p in pred_bboxes]
        gt_ids = [g[0] for g in gt_bboxes]
        gt_boxes = [g[1] for g in gt_bboxes]

        iou_mat = _iou_matrix(pred_boxes, gt_boxes)

        matched_pred: set = set()
        matched_gt: set = set()
        matches: List[Tuple[int, int, float]] = []

        pairs = sorted(
            [
                (i, j, float(iou_mat[i, j]))
                for i in range(len(pred_boxes))
                for j in range(len(gt_boxes))
            ],
            key=lambda x: -x[2],
        )
        for pi, gi, iou in pairs:
            if iou < self.iou_threshold:
                break
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)
            matches.append((pi, gi, iou))

        # TP / FP / FN / IDSW
        tp = len(matches)
        fp = len(pred_bboxes) - tp
        fn = len(gt_bboxes) - tp
        idsw = 0

        for pi, gi, iou in matches:
            self._iou_sum += iou
            self._matched_count += 1
            tid = pred_ids[pi]
            gid = gt_ids[gi]
            if gid in self._gt_to_track and self._gt_to_track[gid] != tid:
                idsw += 1
            self._gt_to_track[gid] = tid

        self._tp += tp
        self._fp += fp
        self._fn += fn
        self._id_switches += idsw

        return {"tp": tp, "fp": fp, "fn": fn, "id_switches": idsw}

    def compute(self) -> Dict:
        """返回汇总跟踪指标字典"""
        gt = max(self._total_gt, 1)
        mota = 1.0 - (self._fn + self._fp + self._id_switches) / gt
        motp = self._iou_sum / max(self._matched_count, 1)

        avg_len = (
            sum(self._track_lengths.values()) / len(self._track_lengths)
            if self._track_lengths
            else 0.0
        )

        return {
            "MOTA": round(mota, 4),
            "MOTP": round(motp, 4),
            "TP": self._tp,
            "FP": self._fp,
            "FN": self._fn,
            "ID_Switch": self._id_switches,
            "total_GT": self._total_gt,
            "num_frames": self._num_frames,
            "num_tracks": len(self._track_lengths),
            "avg_track_length": round(avg_len, 2),
        }

    def print_summary(self) -> None:
        r = self.compute()
        print(
            f"\n[TrackingEvaluator]\n"
            f"  MOTA={r['MOTA']:.4f}  MOTP(IoU)={r['MOTP']:.4f}\n"
            f"  TP={r['TP']}  FP={r['FP']}  FN={r['FN']}  ID_Switch={r['ID_Switch']}\n"
            f"  Frames={r['num_frames']}  Tracks={r['num_tracks']}  "
            f"AvgLen={r['avg_track_length']}"
        )

    def reset(self) -> None:
        self._gt_to_track.clear()
        self._track_lengths.clear()
        self._total_gt = 0
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._id_switches = 0
        self._iou_sum = 0.0
        self._matched_count = 0
        self._num_frames = 0


# ══════════════════════════════════════════════════════════════
# TrackingMetrics — 轻量级累积器（已知 TP/FP/FN 时使用）
# ══════════════════════════════════════════════════════════════

class TrackingMetrics:
    """
    轻量级跟踪指标累积器。
    若需要帧级 IoU 匹配与 ID Switch 检测，请使用 TrackingEvaluator。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._id_switches = 0
        self._total_gt = 0
        self._total_dist = 0.0
        self._matched_count = 0

    def update(
        self,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
        id_switches: int = 0,
        dist_sum: float = 0.0,
        matched: int = 0,
    ) -> None:
        self._tp += tp
        self._fp += fp
        self._fn += fn
        self._id_switches += id_switches
        self._total_dist += dist_sum
        self._matched_count += matched
        self._total_gt += tp + fn

    def compute(self) -> Dict[str, float]:
        gt = max(self._total_gt, 1)
        mota = 1.0 - (self._fn + self._fp + self._id_switches) / gt
        motp = self._total_dist / max(self._matched_count, 1)
        precision = self._tp / max(self._tp + self._fp, 1)
        recall = self._tp / max(self._tp + self._fn, 1)
        return {
            "MOTA": mota,
            "MOTP": motp,
            "precision": precision,
            "recall": recall,
            "TP": self._tp,
            "FP": self._fp,
            "FN": self._fn,
            "ID_Switch": self._id_switches,
        }
