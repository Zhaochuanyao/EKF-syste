"""
轨迹类 - 封装单条目标轨迹的完整状态
"""

import collections
import math
from typing import List, Optional, Tuple
import numpy as np

from .track_state import TrackState
from ..filtering.ekf import ExtendedKalmanFilter
from ..core.types import Detection, Measurement, TrackStateVector
from ..core.constants import IDX_CX, IDX_CY, IDX_W, IDX_H, IDX_OMEGA, IDX_V, IDX_THETA


class Track:
    """
    单条目标轨迹。

    设计原则：让 EKF 自主估计速度和航向，只在初始化时做一次 bootstrap 注入。
    稳定跟踪后完全依赖 EKF 预测-更新循环，不再外部写回状态。
    """

    _id_counter: int = 0

    def __init__(
        self,
        detection: Detection,
        ekf: ExtendedKalmanFilter,
        n_init: int = 3,
        max_age: int = 20,
        frame_id: int = 0,
        anchor_mode: str = "center",
    ) -> None:
        Track._id_counter += 1
        self.track_id: int = Track._id_counter

        self.ekf = ekf
        self.n_init = n_init
        self.max_age = max_age
        self.anchor_mode = anchor_mode

        self.state: TrackState = TrackState.Tentative
        self.hits: int = 1
        self.age: int = 1
        self.time_since_update: int = 0

        self.class_id: int = detection.class_id
        self.class_name: str = detection.class_name
        self.score: float = detection.score
        self.frame_id_created: int = frame_id
        self.frame_id_last: int = frame_id

        self.history: List[Tuple[float, float]] = [
            (float(ekf.x[IDX_CX]), float(ekf.x[IDX_CY]))
        ]

        # 仅用于初始速度估计（前几帧）
        self._prev_cx: float = detection.cx
        self._prev_cy: float = detection.cy
        self._init_done: bool = False  # 初始速度是否已注入

        # 轨迹质量指标
        self.velocity_valid: bool = False
        self.heading_valid: bool = False
        self.stability_score: float = 0.0
        self.recovered_recently: bool = False

    # ──────────────────────────────────────────────────────────
    # 状态访问
    # ──────────────────────────────────────────────────────────

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.Lost

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.Removed

    def get_state(self) -> TrackStateVector:
        return self.ekf.get_state()

    def get_bbox(self) -> np.ndarray:
        return self.ekf.get_state().to_bbox(anchor_mode=self.anchor_mode)

    def get_center(self) -> Tuple[float, float]:
        cx = float(self.ekf.x[IDX_CX])
        cy_anchor = float(self.ekf.x[IDX_CY])
        if self.anchor_mode == "bottom_center":
            h = float(self.ekf.x[IDX_H])
            return (cx, cy_anchor - h / 2)
        return (cx, cy_anchor)

    def get_predicted_measurement(self) -> np.ndarray:
        return self.ekf.get_predicted_measurement()

    def get_innovation_covariance(self) -> np.ndarray:
        return self.ekf.get_innovation_covariance()

    # ──────────────────────────────────────────────────────────
    # 生命周期操作
    # ──────────────────────────────────────────────────────────

    def predict(self, dt: Optional[float] = None) -> None:
        """执行 EKF 预测步骤"""
        _lost_age = self.time_since_update if self.is_lost else 0
        self.ekf.predict(dt, lost_age=_lost_age)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, frame_id: int, dt: Optional[float] = None) -> None:
        """用新检测结果更新轨迹。EKF 自主完成状态估计，仅在第2帧注入初始速度。"""
        was_lost = self.is_lost
        obs_cx, obs_cy = detection.cx, detection.cy

        # EKF 观测更新
        z = detection.to_measurement(anchor_mode=self.anchor_mode)
        meas = Measurement(
            z=z,
            score=detection.score,
            frame_id=frame_id,
            bbox_w=detection.w,
            bbox_h=detection.h,
            aspect_ratio=detection.w / max(detection.h, 1.0),
        )
        self.ekf.update(meas)

        self.hits += 1
        self.time_since_update = 0
        self.score = detection.score
        self.frame_id_last = frame_id

        cx = float(self.ekf.x[IDX_CX])
        cy = float(self.ekf.x[IDX_CY])
        self.history.append((cx, cy))

        # 第2帧：注入初始速度和航向，帮助 EKF 快速收敛
        # 此后完全依赖 EKF 自主估计，不再外部写回
        if not self._init_done and dt is not None and dt > 0 and self.hits == 2:
            dx = obs_cx - self._prev_cx
            dy = obs_cy - self._prev_cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 2.0:
                v_init = min(dist / dt, 500.0)
                theta_init = math.atan2(dy, dx)
                self.ekf.set_kinematics(
                    v=v_init,
                    theta=theta_init,
                    inflate_cov=True,
                    v_var_scale=2.0,
                    theta_var_scale=2.0,
                )
                self.velocity_valid = True
                self.heading_valid = True
            self._init_done = True

        # Lost 恢复：清空历史防止轨迹跨车错乱
        if was_lost:
            self.history.clear()
            self.history.append((cx, cy))
            self.recovered_recently = True
        else:
            self.recovered_recently = False

        self._prev_cx = obs_cx
        self._prev_cy = obs_cy

        self._update_stability()

        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        elif self.state == TrackState.Lost:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        """标记本帧未命中。"""
        if self.state == TrackState.Tentative:
            if self.time_since_update > 1:
                self.state = TrackState.Removed
            else:
                self.state = TrackState.Lost
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Removed
        else:
            self.state = TrackState.Lost

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0

    # ──────────────────────────────────────────────────────────
    # 内部辅助
    # ──────────────────────────────────────────────────────────

    def _update_stability(self) -> None:
        penalty = max(self.time_since_update, 0)
        denom = self.hits + penalty * 2
        self.stability_score = float(self.hits) / denom if denom > 0 else 0.0

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
