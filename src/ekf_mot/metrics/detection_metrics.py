"""检测指标"""
from typing import Dict, List
from ..core.types import Detection


class DetectionMetrics:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._tp = self._fp = self._fn = 0

    def update(self, preds: List[Detection], gts: List[Detection]) -> None:
        # TODO: 完整实现 TP/FP/FN 统计（需要 IoU 匹配）
        pass

    def compute(self) -> Dict[str, float]:
        p = self._tp / (self._tp + self._fp + 1e-6)
        r = self._tp / (self._tp + self._fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        return {"precision": p, "recall": r, "f1": f1}

    def reset(self) -> None:
        self._tp = self._fp = self._fn = 0
