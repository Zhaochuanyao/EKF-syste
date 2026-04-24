"""
轨迹类 - 封装单条目标轨迹的完整状态
"""

import collections
import math
from typing import List, Optional, Tuple
import numpy as np

from .track_state import TrackState
from ..filtering.ekf import ExtendedKalmanFilter
from ..filtering.adaptive_noise import AdaptiveNoiseController, TrackAdaptiveState
from ..filtering.robust_update import should_skip_update
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
        adaptive_controller: Optional[AdaptiveNoiseController] = None,
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

        # 自适应噪声调度状态（controller 为 None 或 disabled 时均不启用）
        self._adaptive_ctrl: Optional[AdaptiveNoiseController] = (
            adaptive_controller if (adaptive_controller is not None and adaptive_controller.cfg.enabled) else None
        )
        self.adaptive_state: Optional[TrackAdaptiveState] = (
            TrackAdaptiveState() if self._adaptive_ctrl is not None else None
        )
        # 用于 Q 调度时计算 delta_theta / delta_omega 的上一帧参考值
        self._prev_theta: float = float(ekf.x[IDX_THETA])
        self._prev_omega: float = float(ekf.x[IDX_OMEGA])

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
        """执行 EKF 预测步骤，自适应开启时注入 Q_adapt。"""
        _lost_age = self.time_since_update if self.is_lost else 0
        _dt = dt if dt is not None else self.ekf.dt

        Q_override = None
        if self._adaptive_ctrl is not None and self._adaptive_ctrl.cfg.q_adapt_on:
            ctrl = self._adaptive_ctrl
            Q_base = self.ekf.build_Q(_dt, _lost_age)
            # 使用上一帧存储的 delta（predict 前计算，体现上一步的机动幅度）
            Q_override, self.adaptive_state = ctrl.adapt_Q(
                Q_base=Q_base,
                nis=self.adaptive_state.prev_nis,
                state=self.adaptive_state,
                dt=_dt,
                delta_theta=self.adaptive_state.last_delta_theta,
                delta_omega=self.adaptive_state.last_delta_omega,
            )
            # 记录本帧 predict 前的 theta/omega，用于下一步 delta 计算
            self._prev_theta = float(self.ekf.x[IDX_THETA])
            self._prev_omega = float(self.ekf.x[IDX_OMEGA])

        self.ekf.predict(dt, lost_age=_lost_age, Q_override=Q_override)

        # 更新 delta 供下一帧 Q 调度使用
        if self._adaptive_ctrl is not None and self._adaptive_ctrl.cfg.q_adapt_on:
            new_theta = float(self.ekf.x[IDX_THETA])
            new_omega = float(self.ekf.x[IDX_OMEGA])
            self.adaptive_state.last_delta_theta = _normalize_angle_diff(new_theta - self._prev_theta)
            self.adaptive_state.last_delta_omega = new_omega - self._prev_omega

        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, frame_id: int, dt: Optional[float] = None) -> None:
        """用新检测结果更新轨迹。EKF 自主完成状态估计，仅在第2帧注入初始速度。"""
        was_lost = self.is_lost
        obs_cx, obs_cy = detection.cx, detection.cy

        z = detection.to_measurement(anchor_mode=self.anchor_mode)
        meas = Measurement(
            z=z,
            score=detection.score,
            frame_id=frame_id,
            bbox_w=detection.w,
            bbox_h=detection.h,
            aspect_ratio=detection.w / max(detection.h, 1.0),
        )

        # ── 自适应噪声更新 ─────────────────────────────────────
        R_override = None
        innov_clip = None
        skip = False

        if self._adaptive_ctrl is not None:
            ctrl = self._adaptive_ctrl
            # 用当前预测状态计算 R_base、新息、NIS
            R_base = self.ekf.build_R(meas)
            innov = meas.z - self.ekf.H @ self.ekf.x
            S = self.ekf.H @ self.ekf.P @ self.ekf.H.T + R_base
            nis = ctrl.compute_nis(innov, S)

            if ctrl.cfg.r_adapt_on:
                R_override, self.adaptive_state = ctrl.adapt_R(R_base, innov, nis, self.adaptive_state)

            if ctrl.cfg.robust_on:
                skip = should_skip_update(
                    nis, ctrl.cfg.drop_threshold, detection.score, ctrl.cfg.low_score
                )
                if not skip:
                    innov_clip = ctrl.cfg.robust_clip_delta

            self.adaptive_state = ctrl.record_update(self.adaptive_state, nis, skipped=skip)

        # EKF 观测更新（backward compatible：无自适应时三个参数均为默认值）
        self.ekf.update(meas, R_override=R_override, innov_clip=innov_clip, skip=skip)

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

    def get_adaptive_diagnostics(self) -> Optional[dict]:
        """返回自适应噪声诊断信息（disabled 时返回 None）。"""
        if self.adaptive_state is None:
            return None
        return self.adaptive_state.get_diagnostics()


def _normalize_angle_diff(diff: float) -> float:
    """将角度差归一化到 [-pi, pi]（Track.predict 内部用）。"""
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff
