"""检测评估器（占位实现）"""
from typing import List, Dict
from ..core.types import Detection


class DetectionEvaluator:
    """基础检测指标计算接口"""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._tp = self._fp = self._fn = 0

    def update(self, preds: List[Detection], gts: List[Detection]) -> None:
        """更新一帧的统计"""
        # TODO: 实现完整的 TP/FP/FN 统计
        pass

    def compute(self) -> Dict[str, float]:
        """计算 precision/recall/F1"""
        precision = self._tp / (self._tp + self._fp + 1e-6)
        recall = self._tp / (self._tp + self._fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return {"precision": precision, "recall": recall, "f1": f1}

    def reset(self) -> None:
        self._tp = self._fp = self._fn = 0
