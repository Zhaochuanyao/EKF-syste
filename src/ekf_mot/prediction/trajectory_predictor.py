"""
轨迹预测模块 - 基于 EKF 递推预测未来轨迹

改进点（v2）:
  - 预测质量门限：只对高质量轨迹输出未来预测
  - 预测置信度输出：基于位置不确定性计算可信度分数
  - fixed-lag smoothing（可选，离线模式）
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from ..tracking.track import Track
from ..tracking.lifecycle import get_prediction_eligible_tracks
from ..core.types import PredictionResult
from ..core.constants import IDX_CX, IDX_CY, IDX_W, IDX_H
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.prediction")


class TrajectoryPredictor:
    """
    对满足质量门限的 Confirmed 轨迹执行未来多步递推预测。

    预测质量门限（可配置）:
      - state == Confirmed
      - hits >= min_hits_for_prediction
      - time_since_update == 0（当前帧有检测命中）
      - 位置协方差迹 < max_position_cov_trace

    对 Lost 轨迹：最多预测 max_lost_predict_steps 步（避免长时外推失真）。
    """

    def __init__(
        self,
        future_steps: List[int] = None,
        dt: float = 0.04,
        min_hits_for_prediction: int = 3,
        max_position_cov_trace: float = 1e6,
        max_lost_predict_steps: int = 1,    # Lost 轨迹最多预测多少步
        fixed_lag_smoothing: bool = False,  # 是否启用固定滞后平滑（仅离线模式）
        smoothing_lag: int = 3,             # 固定滞后平滑的滞后帧数
    ) -> None:
        """
        Args:
            future_steps: 需要预测的未来帧数列表，如 [1, 5, 10]
            dt: 时间步长
            min_hits_for_prediction: 触发预测的最少命中帧数
            max_position_cov_trace: 位置协方差迹上限（超过则视为不可信）
            max_lost_predict_steps: Lost 轨迹的预测步数上限
            fixed_lag_smoothing: 是否启用固定滞后平滑
            smoothing_lag: 固定滞后帧数（smoothing_lag > max(future_steps) 时才有意义）
        """
        self.future_steps = future_steps or [1, 5, 10]
        self.dt = dt
        self.min_hits_for_prediction = min_hits_for_prediction
        self.max_position_cov_trace = max_position_cov_trace
        self.max_lost_predict_steps = max_lost_predict_steps
        self.fixed_lag_smoothing = fixed_lag_smoothing
        self.smoothing_lag = smoothing_lag

    # ──────────────────────────────────────────────────────────
    # 核心预测接口
    # ──────────────────────────────────────────────────────────

    def is_eligible(self, track: Track) -> bool:
        """
        判断轨迹是否满足预测质量门限。

        Returns:
            True = 可以做未来预测，False = 质量不足跳过
        """
        if not track.is_confirmed:
            return False
        if track.hits < self.min_hits_for_prediction:
            return False
        if track.time_since_update != 0:
            return False
        pos_trace = track.ekf.get_position_uncertainty_trace()
        if pos_trace > self.max_position_cov_trace:
            return False
        return True

    def predict_track(
        self,
        track: Track,
        dt: float = None,
    ) -> Dict[int, Tuple[float, float]]:
        """
        对单条轨迹预测未来位置（仅中心点）。

        注意：不检查 eligibility，调用方负责过滤。

        Args:
            track: 目标轨迹
            dt: 时间步长（None 则使用默认值）

        Returns:
            {step: (cx, cy)} 字典
        """
        _dt = dt or self.dt
        max_steps = max(self.future_steps)

        # Lost 轨迹限制预测步数
        if track.is_lost:
            max_steps = min(max_steps, self.max_lost_predict_steps)

        future_states = track.ekf.predict_n_steps(max_steps, _dt)

        result = {}
        for step in self.future_steps:
            if step <= len(future_states):
                state = future_states[step - 1]
                cx = float(state.x[IDX_CX])
                cy = float(state.x[IDX_CY])
                # bottom_center 模式下转换回几何中心
                if track.anchor_mode == "bottom_center":
                    h = float(state.x[IDX_H])
                    cy = cy - h / 2.0
                result[step] = (cx, cy)

        return result

    def predict_track_bboxes(
        self,
        track: Track,
        dt: float = None,
    ) -> Dict[int, np.ndarray]:
        """
        对单条轨迹预测未来 bbox [x1, y1, x2, y2]。

        Returns:
            {step: [x1, y1, x2, y2]} 字典
        """
        _dt = dt or self.dt
        max_steps = max(self.future_steps)

        if track.is_lost:
            max_steps = min(max_steps, self.max_lost_predict_steps)

        future_states = track.ekf.predict_n_steps(max_steps, _dt)

        result = {}
        for step in self.future_steps:
            if step <= len(future_states):
                state = future_states[step - 1]
                result[step] = state.to_bbox(anchor_mode=track.anchor_mode)

        return result

    def compute_prediction_confidence(self, track: Track) -> float:
        """
        计算预测置信度分数 [0, 1]。

        基于以下因素：
        - 位置协方差不确定性（越小越好）
        - 轨迹命中次数（越多越好）
        - 速度/航向估计有效性
        - 距上次更新的帧数（越近越好）

        Returns:
            置信度 [0, 1]，越接近 1 表示预测越可靠
        """
        # 位置不确定性因子
        pos_trace = track.ekf.get_position_uncertainty_trace()
        uncertainty_factor = 1.0 / (1.0 + pos_trace / 100.0)  # 归一化

        # 命中次数因子（命中 >= 10 次算满分）
        hits_factor = min(track.hits / 10.0, 1.0)

        # 运动状态因子
        motion_factor = 0.6
        if track.velocity_valid:
            motion_factor += 0.2
        if track.heading_valid:
            motion_factor += 0.2

        # 时效因子（最近命中越好）
        recency_factor = 1.0 / (1.0 + track.time_since_update * 0.5)

        confidence = (
            0.35 * uncertainty_factor
            + 0.25 * hits_factor
            + 0.25 * motion_factor
            + 0.15 * recency_factor
        )
        return float(min(max(confidence, 0.0), 1.0))

    def predict_all_confirmed(
        self,
        tracks: List[Track],
        dt: float = None,
    ) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """
        对所有满足质量门限的 Confirmed 轨迹批量预测。

        Returns:
            {track_id: {step: (cx, cy)}} 字典
        """
        results = {}
        for track in tracks:
            if track.is_confirmed and self.is_eligible(track):
                results[track.track_id] = self.predict_track(track, dt)
        return results

    def predict_with_confidence(
        self,
        track: Track,
        dt: float = None,
    ) -> Tuple[Dict[int, Tuple[float, float]], float, bool]:
        """
        对单条轨迹预测，同时返回置信度和是否满足质量门限。

        Returns:
            (future_points, confidence, is_eligible)
            - future_points: {step: (cx, cy)}，不满足门限时为 {}
            - confidence: 预测置信度 [0, 1]
            - is_eligible: 是否满足质量门限
        """
        eligible = self.is_eligible(track)
        if not eligible:
            return {}, 0.0, False

        future_points = self.predict_track(track, dt)
        confidence = self.compute_prediction_confidence(track)
        return future_points, confidence, True
