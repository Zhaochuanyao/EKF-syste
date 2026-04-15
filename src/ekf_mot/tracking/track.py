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

    Bootstrap 策略：
    - 仅在"出生早期"（hits <= _warmup_hits_limit）和"Lost 恢复早期"做受限 bootstrap
    - 稳定 Confirmed 轨迹（_bootstrap_frozen=True）不再每帧写回 v/theta/omega
    - bootstrap 写回时同步放大对应协方差（inflate_cov=True），避免 P 过于自信
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

        # ── 运动学 bootstrap 缓存 ────────────────────────────────
        self._obs_centers: collections.deque = collections.deque(maxlen=5)
        self._obs_centers.append((detection.cx, detection.cy))
        self._obs_headings: collections.deque = collections.deque(maxlen=5)

        self._ema_v: Optional[float] = None
        self._ema_theta: Optional[float] = None

        # ── Bootstrap 冻结控制 ───────────────────────────────────
        # 只在出生早期（hits <= _warmup_hits_limit）和 Lost 恢复早期做 bootstrap
        self._bootstrap_frozen: bool = False
        self._warmup_hits_limit: int = max(self.n_init + 2, 4)

        # ── 轨迹质量指标 ─────────────────────────────────────────
        self.velocity_valid: bool = False
        self.heading_valid: bool = False
        self.stability_score: float = 0.0

        # ── 恢复保护状态 ─────────────────────────────────────────
        self.recovered_recently: bool = False
        self._recover_frames_left: int = 0

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
        """执行 EKF 预测步骤，Lost 轨迹传入丢失帧数以放大过程噪声"""
        _lost_age = self.time_since_update if self.is_lost else 0
        self.ekf.predict(dt, lost_age=_lost_age)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, frame_id: int, dt: Optional[float] = None) -> None:
        """
        用新检测结果更新轨迹。

        Bootstrap 触发条件（allow_bootstrap）：
          - 出生早期（hits <= _warmup_hits_limit），或
          - Lost 恢复帧（was_lost），或
          - 恢复保护期内（_recover_frames_left > 0）
        稳定 Confirmed 轨迹不再写回运动学状态。
        """
        was_lost = self.is_lost
        obs_cx, obs_cy = detection.cx, detection.cy

        # ── EKF 观测更新 ─────────────────────────────────────────
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

        # ── Lost 恢复处理 ────────────────────────────────────────
        if was_lost:
            self._obs_centers.clear()
            self._obs_centers.append((obs_cx, obs_cy))
            self._obs_headings.clear()
            self._ema_v = None
            self._ema_theta = None

            self.recovered_recently = True
            self._recover_frames_left = 2
            self._bootstrap_frozen = False  # 恢复后重新允许 bootstrap

            # 大幅衰减 Lost 期间积累的运动学状态
            current_omega = float(self.ekf.x[IDX_OMEGA])
            current_v = float(self.ekf.x[IDX_V])
            self.ekf.set_kinematics(
                v=current_v * 0.5,
                omega=current_omega * 0.15,
                inflate_cov=True,
                v_var_scale=4.0,
                theta_var_scale=3.0,
                omega_var_scale=5.0,
            )
        else:
            # ── Bootstrap 触发判断 ───────────────────────────────
            allow_bootstrap = (
                self.hits <= self._warmup_hits_limit
                or (self._recover_frames_left > 0)
            )

            if not allow_bootstrap:
                self._bootstrap_frozen = True

            if (
                allow_bootstrap
                and dt is not None
                and dt > 0
                and len(self._obs_centers) >= 1
                and self._should_accept_bootstrap(obs_cx, obs_cy, detection)
            ):
                v_est, theta_est, omega_est, valid = self._bootstrap_kinematics(
                    obs_cx, obs_cy, dt
                )
                if valid:
                    self._apply_bootstrap_to_ekf(v_est, theta_est, omega_est)

            self._obs_centers.append((obs_cx, obs_cy))

        # ── 恢复保护帧递减 ───────────────────────────────────────
        if self._recover_frames_left > 0:
            self._recover_frames_left -= 1
            if self._recover_frames_left == 0:
                self.recovered_recently = False

        # ── 稳定性分数 & 状态转移 ────────────────────────────────
        self._update_stability()

        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        elif self.state == TrackState.Lost:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        """标记本帧未命中。Tentative 轨迹允许 1 次 miss，第 2 次才删除。"""
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
    # Bootstrap 辅助方法
    # ──────────────────────────────────────────────────────────

    def _should_accept_bootstrap(
        self, curr_cx: float, curr_cy: float, detection: Detection
    ) -> bool:
        """
        判断本帧观测是否可信，用于决定是否执行 bootstrap 写回。

        大位移 + 低置信度 → 可能是误检跳变，拒绝写回。
        """
        if not self._obs_centers:
            return True
        prev_cx, prev_cy = self._obs_centers[-1]
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        dist = math.sqrt(dx * dx + dy * dy)
        max_side = max(detection.w, detection.h)
        if dist > max(12.0, 0.25 * max_side) and detection.score < 0.7:
            return False
        return True

    def _bootstrap_kinematics(
        self, curr_cx: float, curr_cy: float, dt: float
    ) -> Tuple[float, float, float, bool]:
        """
        估计运动学候选值，不直接写回 EKF。

        Returns:
            (v_est, theta_est, omega_est, valid)
            valid=False 表示位移过小，不应写回。
        """
        if not self._obs_centers:
            return 0.0, 0.0, 0.0, False

        prev_cx, prev_cy = self._obs_centers[-1]
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        dist = math.sqrt(dx * dx + dy * dy)

        # 动态最小位移阈值（基于检测框尺寸，比固定常数更鲁棒）
        # 此处无法直接访问 detection，用 EKF 当前 w/h 估算
        ekf_w = float(self.ekf.x[IDX_W])
        ekf_h = float(self.ekf.x[IDX_H])
        min_move_px = max(4.0, 0.08 * max(ekf_w, ekf_h))

        if dist < min_move_px:
            # 静止：速度 EMA 缓慢衰减
            if self._ema_v is not None and self._ema_v > 0:
                self._ema_v *= 0.75
                if self._ema_v < 1.0:
                    self._ema_v = 0.0
            return 0.0, 0.0, 0.0, False

        # ── 线性回归估计 vx/vy ──────────────────────────────────
        pts = list(self._obs_centers) + [(curr_cx, curr_cy)]
        n = len(pts)

        if n >= 3:
            times = [i * dt for i in range(n)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            t_mean = sum(times) / n
            x_mean = sum(xs) / n
            y_mean = sum(ys) / n
            t_var = sum((t - t_mean) ** 2 for t in times)
            if t_var > 1e-12:
                vx_est = sum((times[i] - t_mean) * (xs[i] - x_mean) for i in range(n)) / t_var
                vy_est = sum((times[i] - t_mean) * (ys[i] - y_mean) for i in range(n)) / t_var
            else:
                vx_est = dx / dt
                vy_est = dy / dt
        else:
            vx_est = dx / dt
            vy_est = dy / dt

        v_est = math.sqrt(vx_est ** 2 + vy_est ** 2)
        theta_est = math.atan2(vy_est, vx_est)

        # ── EMA 融合 ─────────────────────────────────────────────
        in_recovery = self._recover_frames_left > 0
        alpha_v = 0.2 if in_recovery else 0.4
        alpha_t = 0.1 if in_recovery else 0.25

        if self._ema_v is None:
            self._ema_v = v_est
            self._ema_theta = theta_est
        else:
            self._ema_v = alpha_v * v_est + (1 - alpha_v) * self._ema_v
            dtheta = self._normalize_angle(theta_est - self._ema_theta)
            self._ema_theta = self._normalize_angle(self._ema_theta + alpha_t * dtheta)

        self.velocity_valid = True
        self.heading_valid = True

        # ── omega 估计（至少 4 个航向历史点）───────────────────
        self._obs_headings.append(self._ema_theta)
        omega_est = 0.0
        if len(self._obs_headings) >= 4:
            prev_h = self._obs_headings[-2]
            dtheta_o = self._normalize_angle(self._ema_theta - prev_h)

            # 微小转向不更新，向 0 衰减
            if abs(dtheta_o) < 0.03:
                curr_omega = float(self.ekf.x[IDX_OMEGA])
                omega_est = curr_omega * 0.8
            else:
                omega_raw = dtheta_o / dt
                # 严格限幅：恢复期 0.15，正常期 0.25 rad/s
                omega_limit = 0.15 if in_recovery else 0.25
                omega_raw = max(-omega_limit, min(omega_limit, omega_raw))
                curr_omega = float(self.ekf.x[IDX_OMEGA])
                omega_est = 0.8 * curr_omega + 0.2 * omega_raw

        return self._ema_v, self._ema_theta, omega_est, True

    def _apply_bootstrap_to_ekf(
        self,
        v: float,
        theta: float,
        omega: float,
        recovery_mode: bool = False,
    ) -> None:
        """
        将 bootstrap 估计值写回 EKF，同时放大对应协方差。

        只在 EKF 速度仍较低时写入 v/theta（避免覆盖已收敛状态）。
        omega 始终写入（带协方差放大）。
        """
        ekf_v = abs(float(self.ekf.x[IDX_V]))
        if ekf_v < 8.0:
            self.ekf.set_kinematics(
                v=v,
                theta=theta,
                inflate_cov=True,
                v_var_scale=4.0,
                theta_var_scale=3.0,
            )
        if omega != 0.0:
            self.ekf.set_kinematics(
                omega=omega,
                inflate_cov=True,
                omega_var_scale=5.0,
            )

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
