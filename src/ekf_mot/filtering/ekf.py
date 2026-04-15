"""
扩展卡尔曼滤波核心实现
状态向量: x = [cx, cy, v, theta, omega, w, h]^T  (7维)
观测向量: z = [cx, cy, w, h]^T                    (4维)
"""

import math
import numpy as np
from typing import Optional

from ..core.types import TrackStateVector, Measurement
from ..core.constants import (
    IDX_CX, IDX_CY, IDX_V, IDX_THETA, IDX_OMEGA, IDX_W, IDX_H,
    STATE_DIM, MEAS_DIM, DEFAULT_OMEGA_THRESHOLD,
)
from .models.ctrv import ctrv_predict, ctrv_jacobian
from .jacobians import H_MATRIX
from .noise import build_process_noise_Q, build_measurement_noise_R, build_initial_covariance_P


class ExtendedKalmanFilter:
    """
    基于 CTRV 运动模型的扩展卡尔曼滤波器。

    状态向量定义:
        x = [cx, cy, v, theta, omega, w, h]^T
        - cx, cy  : 目标框中心坐标（像素）
        - v       : 速度模长（像素/秒）
        - theta   : 航向角（弧度，x轴正方向为0）
        - omega   : 角速度（弧度/秒）
        - w, h    : 目标框宽高（像素）

    观测向量定义:
        z = [cx, cy, w, h]^T

    EKF 两步流程:
        1. 预测步骤: x_pred = f(x), P_pred = F*P*F^T + Q
        2. 更新步骤: x = x_pred + K*(z - h(x_pred)), P = (I - K*H)*P_pred
    """

    def __init__(
        self,
        dt: float = 0.04,
        std_acc: float = 2.0,
        std_yaw_rate: float = 0.5,
        std_size: float = 0.1,
        std_cx: float = 5.0,
        std_cy: float = 5.0,
        std_w: float = 10.0,
        std_h: float = 10.0,
        score_adaptive: bool = True,
        size_adaptive: bool = False,
        aspect_adaptive: bool = False,
        lost_age_q_scale: float = 1.3,
        omega_threshold: float = DEFAULT_OMEGA_THRESHOLD,
    ) -> None:
        """
        Args:
            dt: 默认时间步长（秒）
            std_acc: 加速度过程噪声标准差
            std_yaw_rate: 角速度变化过程噪声标准差
            std_size: 尺寸变化过程噪声标准差
            std_cx/cy/w/h: 观测噪声标准差（像素）
            score_adaptive: 是否根据检测置信度自适应调整 R
            size_adaptive: 是否根据目标尺寸自适应调整 R（大目标噪声更大）
            aspect_adaptive: 是否根据宽高比自适应调整 R（极端比例检测不稳定）
            lost_age_q_scale: Lost 轨迹每丢失一帧 Q 放大基数（指数增长）
            omega_threshold: omega 接近零的判断阈值
        """
        self.dt = dt
        self.omega_threshold = omega_threshold
        self.score_adaptive = score_adaptive
        self.size_adaptive = size_adaptive
        self.aspect_adaptive = aspect_adaptive
        self._lost_age_q_scale = lost_age_q_scale

        # 噪声参数（用于动态构造 Q 和 R）
        self._std_acc = std_acc
        self._std_yaw_rate = std_yaw_rate
        self._std_size = std_size
        self._std_cx = std_cx
        self._std_cy = std_cy
        self._std_w = std_w
        self._std_h = std_h

        # 状态均值和协方差（初始化后才有效）
        self.x: np.ndarray = np.zeros(STATE_DIM, dtype=np.float64)
        self.P: np.ndarray = np.eye(STATE_DIM, dtype=np.float64)

        # 观测矩阵（线性，常数）
        self.H: np.ndarray = H_MATRIX.copy()

        self._initialized = False

    # ──────────────────────────────────────────────────────────
    # 初始化
    # ──────────────────────────────────────────────────────────

    def initialize(
        self,
        z: np.ndarray,
        score: float = 1.0,
        std_cx: float = 10.0,
        std_cy: float = 10.0,
        std_v: float = 5.0,
        std_theta: float = 0.5,
        std_omega: float = 0.2,
        std_w: float = 20.0,
        std_h: float = 20.0,
    ) -> None:
        """
        用第一个观测值初始化状态。

        速度、航向角、角速度初始化为 0，
        对应分量的协方差设置较大值反映初始不确定性。

        Args:
            z: 初始观测向量 [cx, cy, w, h]
            score: 检测置信度（影响初始 P 的位置分量）
        """
        # 初始化状态均值
        self.x = np.zeros(STATE_DIM, dtype=np.float64)
        self.x[IDX_CX] = z[0]   # cx
        self.x[IDX_CY] = z[1]   # cy
        self.x[IDX_V] = 0.0     # 初始速度未知，设为 0
        self.x[IDX_THETA] = 0.0 # 初始航向未知，设为 0
        self.x[IDX_OMEGA] = 0.0 # 初始角速度未知，设为 0
        self.x[IDX_W] = z[2]    # w
        self.x[IDX_H] = z[3]    # h

        # 初始化协方差矩阵
        self.P = build_initial_covariance_P(
            std_cx=std_cx,
            std_cy=std_cy,
            std_v=std_v,
            std_theta=std_theta,
            std_omega=std_omega,
            std_w=std_w,
            std_h=std_h,
        )

        self._initialized = True

    # ──────────────────────────────────────────────────────────
    # 预测步骤
    # ──────────────────────────────────────────────────────────

    def predict(self, dt: Optional[float] = None, lost_age: int = 0) -> TrackStateVector:
        """
        EKF 预测步骤。

        1. 用 CTRV 非线性函数传播状态均值: x_pred = f(x, dt)
        2. 线性化（计算雅可比 F）
        3. 传播协方差: P_pred = F * P * F^T + Q

        Args:
            dt: 时间步长（None 则使用默认 self.dt）
            lost_age: 轨迹已丢失的帧数（0=正常，>0=Lost 状态，放大 Q 的位置/速度分量）

        Returns:
            预测后的状态
        """
        if not self._initialized:
            raise RuntimeError("EKF 尚未初始化，请先调用 initialize()")

        _dt = dt if dt is not None else self.dt

        # ── Step 1: 非线性状态传播 ────────────────────────────
        x_pred = ctrv_predict(self.x, _dt, self.omega_threshold)

        # 将航向角归一化到 [-pi, pi]
        x_pred[IDX_THETA] = self._normalize_angle(x_pred[IDX_THETA])

        # ── Step 2: 计算状态转移雅可比 F ─────────────────────
        F = ctrv_jacobian(self.x, _dt, self.omega_threshold)

        # ── Step 3: 构造过程噪声矩阵 Q ───────────────────────
        # lost_age > 0 时 Q 指数放大（最多 8x），使 Lost 轨迹的协方差迅速扩大，
        # 恢复时 EKF 更信任新观测，减少位置跳变
        Q = build_process_noise_Q(
            dt=_dt,
            std_acc=self._std_acc,
            std_yaw_rate=self._std_yaw_rate,
            std_size=self._std_size,
            lost_age=lost_age,
            lost_age_q_scale=self._lost_age_q_scale,
        )

        # ── Step 4: 传播协方差 ────────────────────────────────
        # P_pred = F * P * F^T + Q
        P_pred = F @ self.P @ F.T + Q

        # 保证协方差矩阵对称性（消除浮点误差）
        P_pred = (P_pred + P_pred.T) / 2.0

        # 保证协方差矩阵正定（裁剪过小的对角元素）
        for i in range(STATE_DIM):
            if P_pred[i, i] < 1e-6:
                P_pred[i, i] = 1e-6

        # 更新内部状态
        self.x = x_pred
        self.P = P_pred

        return TrackStateVector(x=self.x.copy(), P=self.P.copy())

    # ──────────────────────────────────────────────────────────
    # 更新步骤
    # ──────────────────────────────────────────────────────────

    def update(self, measurement: Measurement) -> TrackStateVector:
        """
        EKF 更新步骤（观测更新）。

        1. 计算预测观测: z_pred = H * x_pred
        2. 计算新息: y = z - z_pred
        3. 计算新息协方差: S = H * P * H^T + R
        4. 计算卡尔曼增益: K = P * H^T * S^{-1}
        5. 更新状态均值: x = x_pred + K * y
        6. 更新协方差: P = (I - K*H) * P_pred

        Args:
            measurement: 观测值，包含 z 向量和可选的 R 矩阵

        Returns:
            更新后的状态
        """
        z = measurement.z

        # ── Step 1: 预测观测值 ────────────────────────────────
        # 观测函数 h(x) = H * x（线性，直接提取 cx,cy,w,h）
        z_pred = self.H @ self.x  # (4,)

        # ── Step 2: 计算新息（残差）──────────────────────────
        y = z - z_pred  # (4,)

        # ── Step 3: 构造观测噪声矩阵 R ───────────────────────
        if measurement.R is not None:
            R = measurement.R
        else:
            # 自适应 R：score/size/aspect 各策略乘法叠加
            R = build_measurement_noise_R(
                std_cx=self._std_cx,
                std_cy=self._std_cy,
                std_w=self._std_w,
                std_h=self._std_h,
                score=measurement.score if self.score_adaptive else None,
                score_adaptive=self.score_adaptive,
                # 尺寸自适应：大目标像素误差更大，放大 R 使 EKF 更信任模型
                bbox_w=measurement.bbox_w,
                bbox_h=measurement.bbox_h,
                size_adaptive=self.size_adaptive,
                # 宽高比自适应：极端比例（如长车侧面）检测不稳定，放大 R
                aspect_ratio=measurement.aspect_ratio,
                aspect_adaptive=self.aspect_adaptive,
            )

        # ── Step 4: 计算新息协方差 S ─────────────────────────
        # S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + R  # (4, 4)

        # ── Step 5: 计算卡尔曼增益 K ─────────────────────────
        # K = P * H^T * S^{-1}
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ self.H.T @ S_inv  # (7, 4)

        # ── Step 6: 更新状态均值 ──────────────────────────────
        # x = x_pred + K * y
        self.x = self.x + K @ y

        # 归一化航向角
        self.x[IDX_THETA] = self._normalize_angle(self.x[IDX_THETA])

        # ── Step 7: 更新协方差（Joseph 形式，数值稳定）────────
        # P = (I - K*H) * P * (I - K*H)^T + K*R*K^T
        I_KH = np.eye(STATE_DIM) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # 保证对称性
        self.P = (self.P + self.P.T) / 2.0

        return TrackStateVector(x=self.x.copy(), P=self.P.copy())

    # ──────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────

    def get_state(self) -> TrackStateVector:
        """获取当前状态"""
        return TrackStateVector(x=self.x.copy(), P=self.P.copy())

    def get_predicted_measurement(self) -> np.ndarray:
        """获取当前状态对应的预测观测值 z_pred = H * x"""
        return self.H @ self.x

    def get_innovation_covariance(self, R: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算新息协方差 S = H * P * H^T + R。
        用于 Mahalanobis 距离计算和门控。
        """
        if R is None:
            R = build_measurement_noise_R(
                std_cx=self._std_cx,
                std_cy=self._std_cy,
                std_w=self._std_w,
                std_h=self._std_h,
            )
        return self.H @ self.P @ self.H.T + R

    # ──────────────────────────────────────────────────────────
    # 运动学设置（bootstrap 接口）
    # ──────────────────────────────────────────────────────────

    def set_kinematics(
        self,
        v: Optional[float] = None,
        theta: Optional[float] = None,
        omega: Optional[float] = None,
        inflate_cov: bool = False,
        v_var_scale: float = 4.0,
        theta_var_scale: float = 3.0,
        omega_var_scale: float = 5.0,
    ) -> None:
        """
        直接设置运动学状态（用于 bootstrap 初始化）。

        inflate_cov=True 时同步放大对应协方差对角元素，
        避免"状态改了但 P 还很自信"导致 EKF 拒绝后续观测修正。

        Args:
            v:              速度模长（像素/秒），None 则不修改
            theta:          航向角（弧度），None 则不修改
            omega:          角速度（弧度/秒），None 则不修改
            inflate_cov:    是否同步放大对应协方差
            v_var_scale:    P[IDX_V, IDX_V] 放大倍数
            theta_var_scale:P[IDX_THETA, IDX_THETA] 放大倍数
            omega_var_scale:P[IDX_OMEGA, IDX_OMEGA] 放大倍数
        """
        if not self._initialized:
            return
        if v is not None:
            self.x[IDX_V] = float(v)
            if inflate_cov:
                self.P[IDX_V, IDX_V] = max(self.P[IDX_V, IDX_V], 1e-4) * v_var_scale
        if theta is not None:
            self.x[IDX_THETA] = self._normalize_angle(float(theta))
            if inflate_cov:
                self.P[IDX_THETA, IDX_THETA] = max(self.P[IDX_THETA, IDX_THETA], 1e-4) * theta_var_scale
        if omega is not None:
            self.x[IDX_OMEGA] = float(omega)
            if inflate_cov:
                self.P[IDX_OMEGA, IDX_OMEGA] = max(self.P[IDX_OMEGA, IDX_OMEGA], 1e-4) * omega_var_scale

    # ──────────────────────────────────────────────────────────
    # 位置不确定性接口
    # ──────────────────────────────────────────────────────────

    def get_position_covariance(self) -> np.ndarray:
        """返回 2x2 位置（cx, cy）协方差子矩阵"""
        return self.P[:2, :2].copy()

    def get_position_uncertainty_trace(self) -> float:
        """返回位置协方差矩阵的迹（cx + cy 方差之和），作为位置不确定性标量指标"""
        return float(self.P[0, 0] + self.P[1, 1])

    def predict_n_steps(self, n: int, dt: Optional[float] = None) -> list:
        """
        从当前状态递推预测未来 n 步，不修改内部状态。

        Args:
            n: 预测步数
            dt: 时间步长

        Returns:
            长度为 n 的 TrackStateVector 列表
        """
        _dt = dt if dt is not None else self.dt

        # 保存当前状态
        x_save = self.x.copy()
        P_save = self.P.copy()

        results = []
        for _ in range(n):
            state = self.predict(_dt)
            results.append(TrackStateVector(x=state.x.copy(), P=state.P.copy()))

        # 恢复状态（预测不改变内部状态）
        self.x = x_save
        self.P = P_save

        return results

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
