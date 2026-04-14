"""
预测指标 - ADE、FDE、RMSE
"""

from typing import Dict, List, Tuple
import numpy as np


class PredictionMetrics:
    """
    轨迹预测误差指标。

    ADE (Average Displacement Error): 所有预测步的平均位移误差
    FDE (Final Displacement Error): 最终预测步的位移误差
    RMSE: 均方根误差
    """

    def __init__(self) -> None:
        self._errors: List[List[float]] = []  # [[step1_err, step5_err, step10_err], ...]

    def update(
        self,
        pred_points: Dict[int, Tuple[float, float]],
        gt_points: Dict[int, Tuple[float, float]],
    ) -> None:
        """
        更新一条轨迹的预测误差。

        Args:
            pred_points: {step: (cx, cy)} 预测点
            gt_points: {step: (cx, cy)} 真实点
        """
        errors = []
        for step in sorted(pred_points.keys()):
            if step in gt_points:
                px, py = pred_points[step]
                gx, gy = gt_points[step]
                err = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                errors.append(float(err))
        if errors:
            self._errors.append(errors)

    def compute(self) -> Dict[str, float]:
        if not self._errors:
            return {"ADE": 0.0, "FDE": 0.0, "RMSE": 0.0}

        all_errors = np.array(self._errors)  # (N, T)

        ade = float(all_errors.mean())
        fde = float(all_errors[:, -1].mean())
        rmse = float(np.sqrt((all_errors ** 2).mean()))

        return {"ADE": ade, "FDE": fde, "RMSE": rmse}

    def reset(self) -> None:
        self._errors.clear()
