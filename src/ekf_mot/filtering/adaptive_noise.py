"""
基于新息统计的分层自适应噪声调度模块

创新点核心实现：
  1. NIS (Normalized Innovation Squared) 异常检测
  2. 测量噪声 R 的对角型在线自适应（含回退机制）
  3. 过程噪声 Q 的机动感知子块缩放调度

每条 track 维护一个 TrackAdaptiveState，由 AdaptiveNoiseController 无状态地驱动更新。
当 enabled=False 时，所有接口均直接返回原始 R/Q，开销为零。
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from ..core.constants import IDX_CX, IDX_CY, IDX_V, IDX_THETA, IDX_OMEGA


# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────

@dataclass
class AdaptiveNoiseConfig:
    """自适应噪声调度器配置，与 ekf_ctrv.yaml 的 adaptive_noise 块一一对应。"""
    enabled: bool = False

    # NIS 阈值
    nis_threshold: float = 9.4877   # chi2(0.99, df=4)，普通异常
    drop_threshold: float = 20.0    # 极端异常，触发 skip update

    # R 自适应参数
    lambda_r: float = 0.6           # R 放大强度
    beta: float = 0.85              # 新息均值遗忘因子
    delta_max: float = 400.0        # 新息偏差逐元素截断上限（像素²）
    recover_alpha_r: float = 0.8    # R 回退衰减（越小回退越快）

    # Q 机动调度参数
    lambda_q: float = 0.3           # Q 缩放强度
    q_max_scale: float = 4.0        # Q 最大缩放倍数
    maneuver_cap: float = 3.0       # NIS 项对机动得分的截断上限
    maneuver_w_nis: float = 1.0     # 机动得分 NIS 项权重
    maneuver_w_omega: float = 0.8   # 机动得分角速度变化权重
    maneuver_w_theta: float = 0.5   # 机动得分航向角速率权重

    # 鲁棒更新
    low_score: float = 0.35         # 极端异常时触发 skip 的检测分数门限
    use_robust_update: bool = True  # 是否启用鲁棒裁剪更新
    robust_clip_delta: float = 25.0 # 新息裁剪阈值（像素）

    # 消融开关（仅 enabled=True 时生效）
    only_r_adapt: bool = False      # 只开 R 自适应
    only_q_schedule: bool = False   # 只开 Q 调度

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveNoiseConfig":
        """从 YAML 加载的字典构造配置。"""
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @property
    def r_adapt_on(self) -> bool:
        return self.enabled and not self.only_q_schedule

    @property
    def q_adapt_on(self) -> bool:
        return self.enabled and not self.only_r_adapt

    @property
    def robust_on(self) -> bool:
        return self.enabled and self.use_robust_update and not self.only_r_adapt and not self.only_q_schedule


# ──────────────────────────────────────────────────────────────
# 每条 Track 的自适应状态
# ──────────────────────────────────────────────────────────────

@dataclass
class TrackAdaptiveState:
    """
    单条 track 的自适应噪声运行时状态。
    由 Track 对象持有，经由 AdaptiveNoiseController 逐帧更新。
    """
    # R 自适应状态
    innov_mean: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    prev_R_diag: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))

    # 机动状态
    last_delta_theta: float = 0.0   # 上一帧航向角变化量（弧度/帧）
    last_delta_omega: float = 0.0   # 上一帧角速度变化量（弧度/秒/帧）
    maneuver_memory: float = 0.0    # 平滑后的机动得分

    # NIS 统计
    prev_nis: float = 0.0
    nis_ema: float = 0.0            # NIS 的指数移动平均（用于 avg_nis 诊断）
    nis_over_threshold_count: int = 0

    # 更新计数
    total_updates: int = 0
    skipped_update_count: int = 0
    abnormal_update_count: int = 0  # NIS 超普通阈值的次数（含 skip 次数）

    def reset(self) -> None:
        """重置为初始状态（新轨迹初始化时调用）。"""
        self.innov_mean[:] = 0.0
        self.prev_R_diag[:] = 0.0
        self.last_delta_theta = 0.0
        self.last_delta_omega = 0.0
        self.maneuver_memory = 0.0
        self.prev_nis = 0.0
        self.nis_ema = 0.0
        self.nis_over_threshold_count = 0
        self.total_updates = 0
        self.skipped_update_count = 0
        self.abnormal_update_count = 0

    @property
    def avg_nis(self) -> float:
        return self.nis_ema

    @property
    def nis_over_threshold_rate(self) -> float:
        if self.total_updates == 0:
            return 0.0
        return self.nis_over_threshold_count / self.total_updates

    @property
    def skipped_update_rate(self) -> float:
        if self.total_updates == 0:
            return 0.0
        return self.skipped_update_count / self.total_updates

    @property
    def abnormal_update_rate(self) -> float:
        if self.total_updates == 0:
            return 0.0
        return self.abnormal_update_count / self.total_updates

    def get_diagnostics(self) -> dict:
        return {
            "avg_nis": round(self.avg_nis, 4),
            "nis_over_threshold_rate": round(self.nis_over_threshold_rate, 4),
            "skipped_update_count": self.skipped_update_count,
            "skipped_update_rate": round(self.skipped_update_rate, 4),
            "abnormal_update_count": self.abnormal_update_count,
            "abnormal_update_rate": round(self.abnormal_update_rate, 4),
            "total_updates": self.total_updates,
            "maneuver_memory": round(self.maneuver_memory, 4),
            "prev_nis": round(self.prev_nis, 4),
        }


# ──────────────────────────────────────────────────────────────
# 控制器（无状态，所有状态通过 TrackAdaptiveState 传入传出）
# ──────────────────────────────────────────────────────────────

class AdaptiveNoiseController:
    """
    自适应噪声调度控制器。

    设计为无状态工具类：不持有任何 track 级别状态，
    通过传入 TrackAdaptiveState 并返回更新后的副本来驱动计算。
    多条 track 可共享同一个 Controller 实例。
    """

    def __init__(self, cfg: AdaptiveNoiseConfig) -> None:
        self.cfg = cfg

    # ──────────────────────────────────────────────────────────
    # 核心 1：NIS 计算
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_nis(innov: np.ndarray, S: np.ndarray) -> float:
        """
        计算 NIS = e^T * S^{-1} * e

        Args:
            innov: 新息向量 e_k，shape (4,)
            S: 新息协方差矩阵，shape (4, 4)

        Returns:
            NIS 标量（非负）
        """
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        nis = float(innov @ S_inv @ innov)
        return max(nis, 0.0)

    # ──────────────────────────────────────────────────────────
    # 核心 2：R 自适应
    # ──────────────────────────────────────────────────────────

    def adapt_R(
        self,
        R_base: np.ndarray,
        innov: np.ndarray,
        nis: float,
        state: TrackAdaptiveState,
    ) -> Tuple[np.ndarray, TrackAdaptiveState]:
        """
        计算自适应测量噪声 R_adapt。

        逻辑：
          - 更新新息均值：innov_mean = beta * innov_mean + (1-beta) * e
          - 计算偏差：delta = clip((e - innov_mean)^2, 0, delta_max)
          - 放大：R_adapt = R_base + lambda_r * diag(delta)
          - 恢复：当 NIS 正常时，R_adapt 指数衰减回 R_base

        Args:
            R_base: 基础测量噪声矩阵 (4x4)
            innov:  当前新息向量 (4,)
            nis:    当前 NIS 值
            state:  当前 track 的自适应状态

        Returns:
            (R_adapt, updated_state)
        """
        cfg = self.cfg
        R_base_diag = np.diag(R_base).copy()

        # 新息均值递推
        state.innov_mean = cfg.beta * state.innov_mean + (1.0 - cfg.beta) * innov

        # 偏差量（逐元素平方后截断）
        delta = np.clip((innov - state.innov_mean) ** 2, 0.0, cfg.delta_max)

        # 自适应 R 对角
        if nis > cfg.nis_threshold:
            # 异常：放大 R
            R_adapt_diag = R_base_diag + cfg.lambda_r * delta
            state.prev_R_diag = R_adapt_diag
        else:
            # 正常：逐步回退到 R_base
            if np.any(state.prev_R_diag > 0.0):
                R_adapt_diag = (
                    cfg.recover_alpha_r * state.prev_R_diag
                    + (1.0 - cfg.recover_alpha_r) * R_base_diag
                )
                # 防止低于 R_base
                R_adapt_diag = np.maximum(R_adapt_diag, R_base_diag)
                state.prev_R_diag = R_adapt_diag
            else:
                R_adapt_diag = R_base_diag

        R_adapt = np.diag(R_adapt_diag).astype(np.float64)
        return R_adapt, state

    # ──────────────────────────────────────────────────────────
    # 核心 3：Q 机动感知调度
    # ──────────────────────────────────────────────────────────

    def adapt_Q(
        self,
        Q_base: np.ndarray,
        nis: float,
        state: TrackAdaptiveState,
        dt: float,
        delta_theta: Optional[float] = None,
        delta_omega: Optional[float] = None,
    ) -> Tuple[np.ndarray, TrackAdaptiveState]:
        """
        计算机动感知 Q_adapt（关键子块缩放，w/h 不放大）。

        机动得分：
          m = w_nis * clip(NIS/nis_threshold, 0, maneuver_cap)
            + w_omega * |delta_omega|
            + w_theta * |delta_theta| / dt

        缩放因子：
          scale = min(1 + lambda_q * m, q_max_scale)

        Q 缩放只作用于位置/速度/航向/角速度维度，尺寸维度保持不变。
        delta_theta / delta_omega 由调用方（Track）计算并传入；
        None 时使用 state.last_delta_theta / last_delta_omega（上一步存储值）。

        Args:
            Q_base:       基础过程噪声矩阵 (7x7)
            nis:          当前（上一步）NIS 值
            state:        当前 track 自适应状态
            dt:           时间步长（秒）
            delta_theta:  航向角变化量（弧度）；None 则用 state 中缓存值
            delta_omega:  角速度变化量（弧度/秒）；None 则用 state 中缓存值

        Returns:
            (Q_adapt, updated_state)
        """
        cfg = self.cfg

        # 使用传入值或上一步缓存的 delta
        d_theta = abs(delta_theta) if delta_theta is not None else abs(state.last_delta_theta)
        d_omega = abs(delta_omega) if delta_omega is not None else abs(state.last_delta_omega)
        _dt = abs(float(dt)) if abs(float(dt)) > 1e-9 else 1e-9

        # 机动得分
        nis_term = min(nis / max(cfg.nis_threshold, 1e-6), cfg.maneuver_cap)
        theta_rate = d_theta / _dt
        m = (
            cfg.maneuver_w_nis * nis_term
            + cfg.maneuver_w_omega * d_omega
            + cfg.maneuver_w_theta * theta_rate
        )

        # 平滑机动得分（EMA，防止单帧噪声）
        state.maneuver_memory = 0.7 * state.maneuver_memory + 0.3 * m

        scale = min(1.0 + cfg.lambda_q * state.maneuver_memory, cfg.q_max_scale)

        # 构造对角缩放矩阵 D（只缩放运动维度，w/h 保持 1.0）
        # 状态向量：[cx, cy, v, theta, omega, w, h]
        d_diag = np.ones(7, dtype=np.float64)
        sqrt_scale = math.sqrt(scale)
        for i in (IDX_CX, IDX_CY, IDX_V, IDX_THETA, IDX_OMEGA):
            d_diag[i] = sqrt_scale

        D = np.diag(d_diag)
        Q_adapt = D @ Q_base @ D  # = D * Q_base * D^T（D 对称）

        return Q_adapt, state

    # ──────────────────────────────────────────────────────────
    # 统计更新（每次 update 调用后调用）
    # ──────────────────────────────────────────────────────────

    def record_update(
        self,
        state: TrackAdaptiveState,
        nis: float,
        skipped: bool,
    ) -> TrackAdaptiveState:
        """更新 NIS 统计计数，在每次 EKF update 后调用。"""
        cfg = self.cfg
        state.total_updates += 1
        state.prev_nis = nis

        # NIS EMA（遗忘因子 0.9，适合长序列在线统计）
        if state.total_updates == 1:
            state.nis_ema = nis
        else:
            state.nis_ema = 0.9 * state.nis_ema + 0.1 * nis

        if skipped:
            state.skipped_update_count += 1
            state.abnormal_update_count += 1
            state.nis_over_threshold_count += 1
        elif nis > cfg.nis_threshold:
            state.abnormal_update_count += 1
            state.nis_over_threshold_count += 1

        return state


# ──────────────────────────────────────────────────────────────
# 工厂方法
# ──────────────────────────────────────────────────────────────

def make_adaptive_controller(cfg_dict: Optional[dict]) -> AdaptiveNoiseController:
    """从配置字典构造控制器。cfg_dict 为 None 或 enabled=False 时返回 disabled 控制器。"""
    if cfg_dict is None:
        return AdaptiveNoiseController(AdaptiveNoiseConfig(enabled=False))
    return AdaptiveNoiseController(AdaptiveNoiseConfig.from_dict(cfg_dict))


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _normalize_angle_diff(diff: float) -> float:
    """将角度差归一化到 [-pi, pi]"""
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff
