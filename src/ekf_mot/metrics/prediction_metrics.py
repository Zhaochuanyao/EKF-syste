"""
预测指标 - ADE、FDE、RMSE（含逐步分解）
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


# ══════════════════════════════════════════════════════════════
# 独立工具函数
# ══════════════════════════════════════════════════════════════

def compute_ade_fde(
    pred_traj: List[Tuple[float, float]],
    gt_traj: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    计算单条轨迹的 ADE 和 FDE。

    Args:
        pred_traj: 预测轨迹 [(cx, cy), ...]，长度 T
        gt_traj:   GT 轨迹   [(cx, cy), ...]，长度 T（与 pred_traj 等长）

    Returns:
        (ADE, FDE) — ADE: 所有步平均位移误差；FDE: 最后一步误差
    """
    if not pred_traj or not gt_traj:
        return 0.0, 0.0
    n = min(len(pred_traj), len(gt_traj))
    errors = [
        float(np.sqrt((pred_traj[i][0] - gt_traj[i][0]) ** 2
                      + (pred_traj[i][1] - gt_traj[i][1]) ** 2))
        for i in range(n)
    ]
    ade = float(np.mean(errors))
    fde = errors[-1]
    return ade, fde


# ══════════════════════════════════════════════════════════════
# PredictionMetrics — 批量累积器
# ══════════════════════════════════════════════════════════════

class PredictionMetrics:
    """
    轨迹预测误差指标累积器。

    ADE (Average Displacement Error)  — 所有预测步的平均位移误差
    FDE (Final Displacement Error)    — 最终预测步的位移误差
    RMSE                              — 均方根误差
    per_step_ADE                      — 各预测步单独的平均误差

    用法::

        metrics = PredictionMetrics()
        for track in confirmed_tracks:
            pred = {1: (px1, py1), 5: (px5, py5), 10: (px10, py10)}
            gt   = {1: (gx1, gy1), 5: (gx5, gy5), 10: (gx10, gy10)}
            metrics.update(pred, gt)
        report = metrics.compute()
    """

    def __init__(self) -> None:
        # _errors[i] = [err_step1, err_step5, err_step10, ...] for i-th sample
        self._errors: List[List[float]] = []
        # _step_errors[step] = [err, err, ...] across all samples
        self._step_errors: Dict[int, List[float]] = {}

    def update(
        self,
        pred_points: Dict[int, Tuple[float, float]],
        gt_points: Dict[int, Tuple[float, float]],
    ) -> Optional[List[float]]:
        """
        更新一条轨迹在各预测步的误差。

        Args:
            pred_points: {step: (cx, cy)} 预测点
            gt_points:   {step: (cx, cy)} 真实点

        Returns:
            当前样本各步误差列表（可为 None）
        """
        errors = []
        for step in sorted(pred_points.keys()):
            if step in gt_points:
                px, py = pred_points[step]
                gx, gy = gt_points[step]
                err = float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))
                errors.append(err)
                self._step_errors.setdefault(step, []).append(err)
        if errors:
            self._errors.append(errors)
            return errors
        return None

    def compute(self) -> Dict:
        """
        返回完整预测指标字典。

        Keys: ADE, FDE, RMSE, num_samples, per_step_ADE
        """
        if not self._errors:
            return {
                "ADE": 0.0,
                "FDE": 0.0,
                "RMSE": 0.0,
                "num_samples": 0,
                "per_step_ADE": {},
            }

        # 补齐不等长序列（用最后一个值填充）
        max_len = max(len(e) for e in self._errors)
        padded = np.array(
            [e + [e[-1]] * (max_len - len(e)) for e in self._errors],
            dtype=np.float64,
        )  # (N, T)

        ade = float(padded.mean())
        fde = float(padded[:, -1].mean())
        rmse = float(np.sqrt((padded ** 2).mean()))

        per_step = {
            str(step): round(float(np.mean(errs)), 4)
            for step, errs in sorted(self._step_errors.items())
        }

        return {
            "ADE": round(ade, 4),
            "FDE": round(fde, 4),
            "RMSE": round(rmse, 4),
            "num_samples": len(self._errors),
            "per_step_ADE": per_step,
        }

    def reset(self) -> None:
        self._errors.clear()
        self._step_errors.clear()
