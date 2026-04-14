"""
跟踪指标 - MOTA、MOTP、ID Switch 等基础统计
"""

from typing import Dict


class TrackingMetrics:
    """
    基础跟踪指标统计器。
    完整实现需要 ground truth 标注，此处提供接口框架。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._tp = 0        # 正确匹配
        self._fp = 0        # 误检（幽灵轨迹）
        self._fn = 0        # 漏检
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
        """
        计算 MOTA 和 MOTP。

        MOTA = 1 - (FN + FP + IDSW) / GT
        MOTP = sum(dist) / matched
        """
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
