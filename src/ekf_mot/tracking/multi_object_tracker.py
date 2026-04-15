"""
多目标跟踪器主类 - 串联检测、EKF、关联、轨迹管理
"""

from typing import Dict, List, Optional, Any
import numpy as np

from .track import Track
from .track_manager import TrackManager
from .association import associate
from ..core.types import Detection, PredictionResult
from ..core.constants import IDX_CX, IDX_CY
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.tracking.mot")


class MultiObjectTracker:
    """
    多目标跟踪器（MOT）。

    每帧处理流程:
    1. 对所有活跃轨迹执行 EKF 预测
    2. 三阶段数据关联（A: Confirmed+Lost×高置信, B: 未匹配×低置信, C: Tentative×高置信）
    3. 更新已匹配轨迹（同时 bootstrap 运动学状态）
    4. 标记未匹配轨迹为丢失
    5. 为未匹配检测框创建新轨迹（受 min_create_score 门控）
    6. 清理已删除轨迹
    7. 返回活跃轨迹结果
    """

    def __init__(
        self,
        n_init: int = 3,
        max_age: int = 20,
        dt: float = 0.04,
        # ── 三阶段关联参数 ──────────────────────────────────────
        high_conf_threshold: float = 0.5,
        low_conf_threshold: float = 0.1,
        gating_threshold_confirmed: float = 9.4877,
        iou_weight: float = 0.4,
        mahal_weight: float = 0.4,
        center_weight: float = 0.2,
        cost_threshold_a: float = 0.8,
        center_norm: float = 200.0,
        iou_threshold_b: float = 0.4,
        iou_threshold_c: float = 0.3,
        second_stage_match: bool = True,
        # ── Stage A2：Lost 轨迹专项恢复（车辆场景关键参数）─────
        lost_recovery_stage: bool = True,
        cost_threshold_a2: float = 0.9,
        # ── 新轨迹创建门限 ─────────────────────────────────────
        min_create_score: float = 0.0,
        anchor_mode: str = "center",
        # ── EKF 参数 ────────────────────────────────────────────
        std_acc: float = 2.0,
        std_yaw_rate: float = 0.5,
        std_size: float = 0.1,
        std_cx: float = 5.0,
        std_cy: float = 5.0,
        std_w: float = 10.0,
        std_h: float = 10.0,
        score_adaptive: bool = True,
        omega_threshold: float = 0.001,
        # ── 向后兼容参数 ────────────────────────────────────────
        iou_threshold: float = None,          # 兼容旧配置，映射到 iou_threshold_c
        gating_threshold: float = None,        # 兼容旧配置
        second_stage_conf: float = None,       # 兼容旧配置，映射到 low_conf_threshold
        second_stage_iou_threshold: float = None,
    ) -> None:
        self.dt = dt

        # 向后兼容映射
        if iou_threshold is not None:
            iou_threshold_c = iou_threshold
        if gating_threshold is not None:
            gating_threshold_confirmed = gating_threshold
        if second_stage_conf is not None:
            low_conf_threshold = second_stage_conf
        if second_stage_iou_threshold is not None:
            iou_threshold_b = second_stage_iou_threshold

        self._assoc_params = dict(
            high_conf_threshold=high_conf_threshold,
            low_conf_threshold=low_conf_threshold,
            gating_threshold_confirmed=gating_threshold_confirmed,
            iou_weight=iou_weight,
            mahal_weight=mahal_weight,
            center_weight=center_weight,
            cost_threshold_a=cost_threshold_a,
            center_norm=center_norm,
            lost_recovery_stage=lost_recovery_stage,
            cost_threshold_a2=cost_threshold_a2,
            iou_threshold_b=iou_threshold_b,
            iou_threshold_c=iou_threshold_c,
            second_stage=second_stage_match,
        )

        self.manager = TrackManager(
            n_init=n_init,
            max_age=max_age,
            dt=dt,
            min_create_score=min_create_score,
            anchor_mode=anchor_mode,
            std_acc=std_acc,
            std_yaw_rate=std_yaw_rate,
            std_size=std_size,
            std_cx=std_cx,
            std_cy=std_cy,
            std_w=std_w,
            std_h=std_h,
            score_adaptive=score_adaptive,
            omega_threshold=omega_threshold,
        )

        self._frame_count = 0

    def step(
        self,
        detections: List[Detection],
        frame_id: int,
        dt: Optional[float] = None,
    ) -> List[Track]:
        """
        处理单帧，返回当前活跃轨迹列表。

        Args:
            detections: 当前帧检测结果
            frame_id: 帧ID
            dt: 时间步长（None 则使用默认值）

        Returns:
            活跃轨迹列表（Confirmed + Tentative）
        """
        self._frame_count += 1
        _dt = dt if dt is not None else self.dt

        # ── Step 1: EKF 预测 ──────────────────────────────────
        self.manager.predict_all(_dt)

        # ── Step 2: 三阶段数据关联 ────────────────────────────
        tracks = self.manager.tracks
        matches, unmatched_tracks, unmatched_dets = associate(
            tracks=tracks,
            detections=detections,
            **self._assoc_params,
        )

        logger.debug(
            f"帧 {frame_id}: 检测={len(detections)} 轨迹={len(tracks)} "
            f"匹配={len(matches)} 未匹配轨迹={len(unmatched_tracks)} "
            f"未匹配检测={len(unmatched_dets)}"
        )

        # ── Step 3: 更新已匹配轨迹（含 bootstrap）────────────
        self.manager.update_matched(matches, tracks, detections, frame_id, dt=_dt)

        # ── Step 4: 标记未匹配轨迹 ───────────────────────────
        self.manager.mark_unmatched_missed(unmatched_tracks)

        # ── Step 5: 创建新轨迹 ───────────────────────────────
        self.manager.create_new_tracks(unmatched_dets, detections, frame_id)

        # ── Step 6: 清理已删除轨迹 ───────────────────────────
        self.manager.cleanup()

        # ── Step 7: 返回活跃轨迹 ─────────────────────────────
        return self.manager.get_active()

    def get_confirmed_tracks(self) -> List[Track]:
        return self.manager.get_confirmed()

    def reset(self) -> None:
        self.manager.reset()
        self._frame_count = 0

    @classmethod
    def from_config(cls, cfg: Any) -> "MultiObjectTracker":
        """从配置对象构建跟踪器"""
        tracker_cfg = cfg.tracker if hasattr(cfg, "tracker") else cfg.get("tracker", {})
        ekf_cfg = cfg.ekf if hasattr(cfg, "ekf") else cfg.get("ekf", {})

        def _get(obj, key, default):
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default

        pn = _get(ekf_cfg, "process_noise", {})
        mn = _get(ekf_cfg, "measurement_noise", {})

        def _g(d, k, v):
            return d.get(k, v) if isinstance(d, dict) else getattr(d, k, v)

        return cls(
            n_init=_get(tracker_cfg, "n_init", 3),
            max_age=_get(tracker_cfg, "max_age", 20),
            dt=_get(tracker_cfg, "dt", 0.04),
            # 三阶段关联参数（优先读新 key，fallback 到旧 key）
            high_conf_threshold=_get(tracker_cfg, "high_conf_threshold", 0.5),
            low_conf_threshold=_get(tracker_cfg, "low_conf_threshold",
                                    _get(tracker_cfg, "second_stage_conf", 0.1)),
            gating_threshold_confirmed=_get(tracker_cfg, "gating_threshold_confirmed",
                                            _get(tracker_cfg, "gating_threshold", 9.4877)),
            iou_weight=_get(tracker_cfg, "cost_iou_weight", 0.4),
            mahal_weight=_get(tracker_cfg, "cost_mahal_weight", 0.4),
            center_weight=_get(tracker_cfg, "cost_center_weight_confirmed", 0.2),
            cost_threshold_a=_get(tracker_cfg, "cost_threshold_a", 0.8),
            center_norm=_get(tracker_cfg, "center_norm", 200.0),
            iou_threshold_b=_get(tracker_cfg, "iou_threshold_b",
                                  _get(tracker_cfg, "second_stage_iou_threshold", 0.4)),
            iou_threshold_c=_get(tracker_cfg, "iou_threshold_c",
                                  _get(tracker_cfg, "iou_threshold", 0.3)),
            second_stage_match=_get(tracker_cfg, "second_stage_match", True),
            lost_recovery_stage=_get(tracker_cfg, "lost_recovery_stage", True),
            cost_threshold_a2=_get(tracker_cfg, "cost_threshold_a2", 0.9),
            min_create_score=_get(tracker_cfg, "min_create_score", 0.0),
            anchor_mode=_get(tracker_cfg, "anchor_mode", "center"),
            std_acc=_g(pn, "std_acc", 2.0),
            std_yaw_rate=_g(pn, "std_yaw_rate", 0.5),
            std_size=_g(pn, "std_size", 0.1),
            std_cx=_g(mn, "std_cx", 5.0),
            std_cy=_g(mn, "std_cy", 5.0),
            std_w=_g(mn, "std_w", 10.0),
            std_h=_g(mn, "std_h", 10.0),
            score_adaptive=_g(mn, "score_adaptive", True),
        )
