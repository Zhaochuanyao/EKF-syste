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

    每条轨迹维护:
    - 唯一 ID
    - EKF 滤波器实例
    - 生命周期状态
    - 历史中心点轨迹
    - 命中/丢失统计
    - 质量指标（stability_score, velocity_valid, heading_valid）
    - 运动学 bootstrap（滑动窗口线性拟合 + EMA 融合）
    - 恢复保护状态（Lost 恢复后限制几帧内的更新幅度）
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

        # ── 运动学 bootstrap 缓存（改用 deque，自动限制大小）────
        # 最近 5 次实际观测到的中心点（未经 EKF 滤波），用于线性拟合
        self._obs_centers: collections.deque = collections.deque(maxlen=5)
        self._obs_centers.append((detection.cx, detection.cy))
        # EMA 平滑后的航向历史，用于 omega 估计
        self._obs_headings: collections.deque = collections.deque(maxlen=5)

        # EMA 融合缓存（避免两帧差分直接覆盖 EKF 状态）
        self._ema_v: Optional[float] = None     # 速度 EMA 缓存
        self._ema_theta: Optional[float] = None  # 航向角 EMA 缓存

        # ── 轨迹质量指标 ───────────────────────────────────────
        self.velocity_valid: bool = False   # 速度估计是否有效（第2+次命中后）
        self.heading_valid: bool = False    # 航向估计是否有效（第2+次命中后）
        self.stability_score: float = 0.0  # 轨迹稳定性分数 [0, 1]

        # ── 恢复保护状态（Lost → Confirmed 后限制几帧更新幅度）──
        self.recovered_recently: bool = False  # 是否处于恢复保护期
        self._recover_frames_left: int = 0     # 剩余保护帧数

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
        """执行 EKF 预测步骤，Lost 轨迹传入丢失帧数以放大过程噪声"""
        # Lost 轨迹传递 time_since_update，Confirmed/Tentative 传 0
        # 注意：time_since_update 在此处读取（当前值），方法末尾才 +1
        _lost_age = self.time_since_update if self.is_lost else 0
        self.ekf.predict(dt, lost_age=_lost_age)
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
        # 记录恢复前的状态（Lost→Confirmed 需特殊处理）
        was_lost = self.is_lost

        # 记录本次实际观测中心（EKF 更新前）
        obs_cx, obs_cy = detection.cx, detection.cy

        # 构造含尺寸信息的 Measurement（供 R 自适应使用）
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

        # 记录 EKF 滤波后的历史轨迹
        cx = float(self.ekf.x[IDX_CX])
        cy = float(self.ekf.x[IDX_CY])
        self.history.append((cx, cy))

        # ── Lost 轨迹恢复处理 ────────────────────────────────
        if was_lost:
            # 重置 bootstrap 缓存：避免长时间隙导致错误的速度/航向估计
            self._obs_centers.clear()
            self._obs_centers.append((obs_cx, obs_cy))
            self._obs_headings.clear()
            self._ema_v = None
            self._ema_theta = None

            # 设置恢复保护状态：恢复后3帧内限制更新幅度
            self.recovered_recently = True
            self._recover_frames_left = 3

            # 抑制 CTRV 在 Lost 期间积累的角速度和速度（可能已严重偏离）
            current_omega = float(self.ekf.x[IDX_OMEGA])
            if abs(current_omega) > 0.2:
                self.ekf.set_kinematics(omega=current_omega * 0.3)
            current_v = float(self.ekf.x[IDX_V])
            self.ekf.set_kinematics(v=current_v * 0.7)
        else:
            # ── 正常运动学 Bootstrap ─────────────────────────
            # 在 EKF 更新之后，用实际观测中心估计运动方向/速度
            if dt is not None and dt > 0 and len(self._obs_centers) >= 1:
                self._bootstrap_kinematics(obs_cx, obs_cy, dt)

            # 保存本次观测（deque 自动限制大小为 5）
            self._obs_centers.append((obs_cx, obs_cy))

        # ── 恢复保护帧递减 ───────────────────────────────────
        if self._recover_frames_left > 0:
            self._recover_frames_left -= 1
            if self._recover_frames_left == 0:
                self.recovered_recently = False

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
            # Tentative 轨迹允许 1 次 miss（避免单帧漏检导致轨迹碎片化）
            # time_since_update > 1 时才删除（即连续2次未命中）
            if self.time_since_update > 1:
                self.state = TrackState.Removed
            else:
                self.state = TrackState.Lost  # 短暂进入 Lost，下帧命中可恢复
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Removed
        else:
            self.state = TrackState.Lost

    @classmethod
    def reset_id_counter(cls) -> None:
        """重置 ID 计数器（用于测试）"""
        cls._id_counter = 0

    # ──────────────────────────────────────────────────────────
    # 内部方法：Bootstrap 运动学（滑动窗口线性拟合 + EMA）
    # ──────────────────────────────────────────────────────────

    def _bootstrap_kinematics(self, curr_cx: float, curr_cy: float, dt: float) -> None:
        """
        用滑动窗口线性拟合估计 v/theta，EMA 融合避免突变，omega 带衰减保护。

        改进原因：
        - 原来的两帧差分直接覆盖 EKF 状态，对检测框±5px 噪声极敏感
        - 现用最近5帧中心点做最小二乘线性回归，鲁棒性提升约 sqrt(n/2) 倍
        - EMA 融合确保 v/theta/omega 不在单帧内大幅跳变
        - 恢复保护期内使用更保守的 EMA 系数，避免过冲

        调用时机：正常命中帧（非 Lost 恢复帧）
        """
        if len(self._obs_centers) < 1:
            return

        # ── 1. 小位移保护 ──────────────────────────────────────
        # 静止/几乎静止时不更新 heading/omega，速度 EMA 缓慢衰减
        prev_cx, prev_cy = self._obs_centers[-1]
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        dist = math.sqrt(dx * dx + dy * dy)
        min_move_px = max(3.0, dt * 5.0)  # 至少 3px 或 5px/s

        if dist < min_move_px:
            # 静止时让速度 EMA 缓慢衰减（避免静止噪声积累虚假速度）
            if self._ema_v is not None and self._ema_v > 0:
                self._ema_v *= 0.75
                if self._ema_v < 1.0:
                    self._ema_v = 0.0
                    self.ekf.set_kinematics(v=0.0)
            return

        # ── 2. 构建时间序列（含当前点）─────────────────────────
        pts = list(self._obs_centers) + [(curr_cx, curr_cy)]
        n = len(pts)

        if n >= 3:
            # 线性回归估计 vx, vy（比两帧差分鲁棒 ~sqrt(n/2) 倍）
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
            # 少于3点时退化为两帧差分
            vx_est = dx / dt
            vy_est = dy / dt

        v_est = math.sqrt(vx_est ** 2 + vy_est ** 2)
        theta_est = math.atan2(vy_est, vx_est)

        # ── 3. EMA 融合（恢复期使用更保守系数，避免过冲）──────
        in_recovery = self._recover_frames_left > 0
        alpha_v = 0.25 if in_recovery else 0.4
        alpha_t = 0.15 if in_recovery else 0.25

        if self._ema_v is None:
            # 首次初始化
            self._ema_v = v_est
            self._ema_theta = theta_est
        else:
            self._ema_v = alpha_v * v_est + (1 - alpha_v) * self._ema_v
            # 航向角需要走最短路径（处理 ±π 跳变）
            dtheta = self._normalize_angle(theta_est - self._ema_theta)
            self._ema_theta = self._normalize_angle(self._ema_theta + alpha_t * dtheta)

        self.velocity_valid = True
        self.heading_valid = True

        # ── 4. 写入 EKF 状态（仅当速度仍较低时，避免覆盖已收敛状态）
        ekf_v = abs(float(self.ekf.x[IDX_V]))
        if ekf_v < 8.0:
            self.ekf.set_kinematics(v=self._ema_v, theta=self._ema_theta)

        # ── 5. omega 估计（至少2个航向历史点）──────────────────
        self._obs_headings.append(self._ema_theta)
        if len(self._obs_headings) >= 2:
            prev_h = self._obs_headings[-2]
            dtheta_o = self._normalize_angle(self._ema_theta - prev_h)
            omega_est = dtheta_o / dt

            # 恢复期限制更严，避免 omega 过冲
            omega_limit = 0.35 if in_recovery else 0.8
            if abs(omega_est) < omega_limit:
                # EMA 融合 omega（而不是直接覆盖）
                curr_omega = float(self.ekf.x[IDX_OMEGA])
                new_omega = 0.8 * curr_omega + 0.2 * omega_est
                self.ekf.set_kinematics(omega=new_omega)
            else:
                # 超出阈值：omega 向零衰减，而不是忽略
                curr_omega = float(self.ekf.x[IDX_OMEGA])
                self.ekf.set_kinematics(omega=curr_omega * 0.5)

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
