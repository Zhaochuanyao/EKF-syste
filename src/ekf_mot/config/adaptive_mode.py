"""
EKF 自适应噪声策略模式枚举与配置映射。
所有 mode 定义、label、adaptive_noise_cfg 映射集中在此文件，供 API 和 tracker 共用。
"""

from typing import Dict, Optional

# ── 模式枚举（字符串常量）──────────────────────────────────────
MODE_CURRENT_EKF = "current_ekf"
MODE_R_ADAPT     = "r_adapt"
MODE_Q_SCHED     = "q_sched"
MODE_RQ_ADAPT    = "rq_adapt"
MODE_FULL_ADPT   = "full_adpt"

VALID_MODES = {MODE_CURRENT_EKF, MODE_R_ADAPT, MODE_Q_SCHED, MODE_RQ_ADAPT, MODE_FULL_ADPT}

# ── 前端展示标签 ───────────────────────────────────────────────
MODE_LABELS: Dict[str, str] = {
    MODE_CURRENT_EKF: "Current EKF",
    MODE_R_ADAPT:     "+R-adapt",
    MODE_Q_SCHED:     "+Q-sched",
    MODE_RQ_ADAPT:    "+RQ-adapt",
    MODE_FULL_ADPT:   "Full Adpt",
}

# ── 基础自适应参数（与 run_adaptive_ablation.py 保持一致）──────
_ADAPTIVE_BASE = {
    "enabled": True,
    "nis_threshold": 9.4877,
    "drop_threshold": 20.0,
    "lambda_r": 0.3,
    "beta": 0.85,
    "delta_max": 400.0,
    "recover_alpha_r": 0.65,
    "lambda_q": 0.3,
    "q_max_scale": 4.0,
    "maneuver_cap": 3.0,
    "maneuver_w_nis": 1.0,
    "maneuver_w_omega": 0.8,
    "maneuver_w_theta": 0.5,
    "low_score": 0.35,
    "use_robust_update": True,
    "robust_clip_delta": 25.0,
    "only_r_adapt": False,
    "only_q_schedule": False,
}


def resolve_adaptive_mode(mode: str) -> Optional[dict]:
    """
    将前端 mode 字符串映射为 adaptive_noise_cfg dict（传给 make_adaptive_controller）。
    current_ekf 返回 None（禁用自适应）。
    """
    if mode == MODE_CURRENT_EKF:
        return None
    if mode == MODE_R_ADAPT:
        return {**_ADAPTIVE_BASE, "only_r_adapt": True, "only_q_schedule": False,
                "use_robust_update": False}
    if mode == MODE_Q_SCHED:
        return {**_ADAPTIVE_BASE, "only_r_adapt": False, "only_q_schedule": True,
                "use_robust_update": False}
    if mode == MODE_RQ_ADAPT:
        return {**_ADAPTIVE_BASE, "only_r_adapt": False, "only_q_schedule": False,
                "use_robust_update": False}
    if mode == MODE_FULL_ADPT:
        return {**_ADAPTIVE_BASE, "only_r_adapt": False, "only_q_schedule": False,
                "use_robust_update": True}
    raise ValueError(f"未知 mode: {mode}")
