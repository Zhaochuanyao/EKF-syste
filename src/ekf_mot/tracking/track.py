"""
轨迹类 - 封装单条目标轨迹的完整状态
"""

import math
from typing import List, Optional, Tuple
import numpy as np

from .track_state import TrackState
from ..filtering.ekf import ExtendedKalmanFilter
from ..core.types import Detection, Measurement, TrackStateVector
from ..core.constants import IDX_CX, IDX_CY, IDX_W, IDX_H


class Track:
    """
    单条目标轨迹。

    每条轨迹维护:
    - 唯一 ID
    - EKF 滤波器实例
    - 生命周期状态
    - 历史中心点轨迹
    - 命中/丢失统计
    - 质量指标（stability_score, velocity_valid, heading_valid）
    - 运动学 bootstrap（第 2、3 次命中时估计 v/theta/omega）
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
        """
        Args:
            detection: 触发创建的检测结果
            ekf: 已初始化的 EKF 实例
            n_init: 确认所需连续命中帧数
            max_age: 最大丢失帧数
            frame_id: 创建时的帧ID
            anchor_mode: 观测锚点模式（"center" 或 "bottom_center"）
        """
        Track._id_counter += 1
        self.track_id: int = Track._id_counter

        self.ekf = ekf
        self.n_init = n_init
        self.max_age = max_age
        self.anchor_mode = anchor_mode

        self.state: TrackState = TrackState.Tentative
        self.hits: int = 1              # 总命中次数
        self.age: int = 1               # 总存活帧数
        self.time_since_update: int = 0 # 距上次更新的帧数

        self.class_id: int = detection.class_id
        self.class_name: str = detection.class_name
        self.score: float = detection.score
        self.frame_id_created: int = frame_id
        self.frame_id_last: int = frame_id

        # 历史中心点（用于可视化轨迹线）
        self.history: List[Tuple[float, float]] = [
            (float(ekf.x[IDX_CX]), float(ekf.x[IDX_CY]))
        ]

        # ── 运动学 bootstrap 缓存 ──────────────────────────────
        # 存储最近 3 次实际观测到的中心点（未经 EKF 滤波）
        self._obs_centers: List[Tuple[float, float]] = [
            (detection.cx, detection.cy)
        ]
        self._obs_headings: List[float] = []  # 从连续观测估算的航向角

        # ── 轨迹质量指标 ───────────────────────────────────────
        self.velocity_valid: bool = False   # 速度估计是否有效（第2+次命中后）
        self.heading_valid: bool = False    # 航向估计是否有效（第2+次命中后）
        self.stability_score: float = 0.0  # 轨迹稳定性分数 [0, 1]

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
        """返回当前 EKF 估计的 bbox [x1,y1,x2,y2]"""
        return self.ekf.get_state().to_bbox(anchor_mode=self.anchor_mode)

    def get_center(self) -> Tuple[float, float]:
        """返回当前 EKF 估计的中心点（对 bottom_center 模式返回几何中心）"""
        cx = float(self.ekf.x[IDX_CX])
        cy_anchor = float(self.ekf.x[IDX_CY])
        if self.anchor_mode == "bottom_center":
            h = float(self.ekf.x[IDX_H])
            return (cx, cy_anchor - h / 2)
        return (cx, cy_anchor)

    def get_predicted_measurement(self) -> np.ndarray:
        """获取预测观测值 [cx, cy, w, h]"""
        return self.ekf.get_predicted_measurement()

    def get_innovation_covariance(self) -> np.ndarray:
        """获取新息协方差 S，用于 Mahalanobis 距离计算"""
        return self.ekf.get_innovation_covariance()

    # ──────────────────────────────────────────────────────────
    # 生命周期操作
    # ──────────────────────────────────────────────────────────

    def predict(self, dt: Optional[float] = None) -> None:
        """执行 EKF 预测步骤"""
        self.ekf.predict(dt)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, frame_id: int, dt: Optional[float] = None) -> None:
        """
        用新检测结果更新轨迹，并尝试 bootstrap 运动学状态。

        Args:
            detection: 匹配到的检测结果
            frame_id: 当前帧ID
            dt: 当前帧时间步长（用于 bootstrap 速度估计，None 则跳过 bootstrap）
        """
        # 记录本次实际观测中心（EKF 更新前）
        obs_cx, obs_cy = detection.cx, detection.cy

        z = detection.to_measurement(anchor_mode=self.anchor_mode)
        meas = Measurement(z=z, score=detection.score, frame_id=frame_id)
        self.ekf.update(meas)

        self.hits += 1
        self.time_since_update = 0
        self.score = detection.score
        self.frame_id_last = frame_id

        # 记录 EKF 滤波后的历史轨迹
        cx = float(self.ekf.x[IDX_CX])
        cy = float(self.ekf.x[IDX_CY])
        self.history.append((cx, cy))

        # ── 运动学 Bootstrap ─────────────────────────────────
        # 在 EKF 更新之后，用实际观测中心估计运动方向/速度
        if dt is not None and dt > 0 and len(self._obs_centers) >= 1:
            self._bootstrap_kinematics(obs_cx, obs_cy, dt)

        # 保存本次观测（限制缓存大小为 4）
        self._obs_centers.append((obs_cx, obs_cy))
        if len(self._obs_centers) > 4:
            self._obs_centers.pop(0)

        # ── 更新稳定性分数 ────────────────────────────────────
        self._update_stability()

        # ── 状态转移 ──────────────────────────────────────────
        # Tentative → Confirmed（连续 n_init 次命中后确认）
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        # Lost → Confirmed（重新找回后恢复，保留已有运动状态）
        elif self.state == TrackState.Lost:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        """标记本帧未命中"""
        if self.state == TrackState.Tentative:
            # Tentative 轨迹未命中直接删除
            self.state = TrackState.Removed
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Removed
        else:
            self.state = TrackState.Lost

    @classmethod
    def reset_id_counter(cls) -> None:
        """重置 ID 计数器（用于测试）"""
        cls._id_counter = 0

    # ──────────────────────────────────────────────────────────
    # 内部方法：Bootstrap 运动学
    # ──────────────────────────────────────────────────────────

    def _bootstrap_kinematics(self, curr_cx: float, curr_cy: float, dt: float) -> None:
        """
        根据前后两次实际观测中心估计速度和航向，
        并写入 EKF 状态向量（bootstrap 初始化）。

        调用时机：
          - 第 2 次命中：估计 v 和 theta
          - 第 3+ 次命中：附加估计 omega（航向变化率）

        只有当目标移动距离超过最小阈值时才更新，
        避免静止目标的噪声干扰运动估计。
        """
        if len(self._obs_centers) < 1:
            return

        prev_cx, prev_cy = self._obs_centers[-1]
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        dist = math.sqrt(dx * dx + dy * dy)

        # 最小移动阈值：避免静止噪声污染运动状态
        min_move_px = max(3.0, dt * 5.0)  # 至少 3px 或 5px/s
        if dist < min_move_px:
            return

        # ── 估计速度和航向 ────────────────────────────────────
        v_est = dist / dt
        theta_est = math.atan2(dy, dx)

        self.velocity_valid = True
        self.heading_valid = True
        self._obs_headings.append(theta_est)
        if len(self._obs_headings) > 4:
            self._obs_headings.pop(0)

        # 第 2 次命中：初始化 v 和 theta（若 EKF 速度仍接近 0）
        if abs(float(self.ekf.x[2])) < 1.0:
            self.ekf.set_kinematics(v=v_est, theta=theta_est)

        # ── 估计角速度 omega ──────────────────────────────────
        if len(self._obs_headings) >= 2:
            prev_heading = self._obs_headings[-2]
            dtheta = self._normalize_angle(theta_est - prev_heading)
            omega_est = dtheta / dt
            # 仅当角速度估计合理时更新（避免噪声导致的虚假角速度）
            if abs(omega_est) < 5.0:  # 最大 5 rad/s ≈ 286 °/s
                self.ekf.set_kinematics(omega=omega_est)

    def _update_stability(self) -> None:
        """
        更新轨迹稳定性分数。

        稳定性 = hits / (hits + time_since_update × penalty)，
        值域 [0, 1]，1 = 完全稳定（每帧都有检测命中）。
        """
        penalty = max(self.time_since_update, 0)
        denom = self.hits + penalty * 2
        self.stability_score = float(self.hits) / denom if denom > 0 else 0.0

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
