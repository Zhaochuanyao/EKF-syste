"""
轨迹管理器 - 统一管理所有轨迹的创建、更新、删除
"""

import math
from typing import Dict, List, Optional
import numpy as np

from .track import Track
from .track_state import TrackState
from .lifecycle import apply_lifecycle
from ..core.types import Detection, Measurement
from ..filtering.ekf import ExtendedKalmanFilter
from ..filtering.adaptive_noise import AdaptiveNoiseController, make_adaptive_controller
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.tracking.manager")


class TrackManager:
    """
    轨迹管理器，负责:
    - 创建新轨迹（支持 min_create_score 门限）
    - 更新已匹配轨迹（支持 bootstrap 时间步 dt）
    - 标记未匹配轨迹为丢失
    - 删除过期轨迹
    - 提供轨迹查询接口
    """

    def __init__(
        self,
        n_init: int = 3,
        max_age: int = 20,
        dt: float = 0.04,
        min_create_score: float = 0.0,   # 创建新轨迹所需的最低置信度
        anchor_mode: str = "center",      # 观测锚点模式
        # EKF 噪声参数
        std_acc: float = 2.0,
        std_yaw_rate: float = 0.5,
        std_size: float = 0.1,
        std_pos: float = 0.0,
        std_vel: float = 0.0,
        std_cx: float = 5.0,
        std_cy: float = 5.0,
        std_w: float = 10.0,
        std_h: float = 10.0,
        score_adaptive: bool = True,
        size_adaptive: bool = False,
        aspect_adaptive: bool = False,
        lost_age_q_scale: float = 1.3,
        omega_threshold: float = 0.001,
        # 初始协方差参数
        init_std_cx: float = 10.0,
        init_std_cy: float = 10.0,
        init_std_v: float = 5.0,
        init_std_theta: float = 0.5,
        init_std_omega: float = 0.2,
        init_std_w: float = 20.0,
        init_std_h: float = 20.0,
        # 自适应噪声配置（dict 或 None）
        adaptive_noise_cfg: Optional[dict] = None,
    ) -> None:
        self.n_init = n_init
        self.max_age = max_age
        self.dt = dt
        self.min_create_score = min_create_score
        self.anchor_mode = anchor_mode

        # EKF 参数（用于创建新轨迹时）
        self._ekf_params = dict(
            dt=dt,
            std_acc=std_acc,
            std_yaw_rate=std_yaw_rate,
            std_size=std_size,
            std_pos=std_pos,
            std_vel=std_vel,
            std_cx=std_cx,
            std_cy=std_cy,
            std_w=std_w,
            std_h=std_h,
            score_adaptive=score_adaptive,
            size_adaptive=size_adaptive,
            aspect_adaptive=aspect_adaptive,
            lost_age_q_scale=lost_age_q_scale,
            omega_threshold=omega_threshold,
        )
        self._init_cov_params = dict(
            std_cx=init_std_cx,
            std_cy=init_std_cy,
            std_v=init_std_v,
            std_theta=init_std_theta,
            std_omega=init_std_omega,
            std_w=init_std_w,
            std_h=init_std_h,
        )

        # 自适应噪声调度器（共享，无状态；None 表示 disabled）
        self._adaptive_ctrl: Optional[AdaptiveNoiseController] = make_adaptive_controller(adaptive_noise_cfg)
        if self._adaptive_ctrl.cfg.enabled:
            logger.info("自适应噪声调度已启用 (AdaptiveNoiseController)")

        self._tracks: List[Track] = []
        self._pending_births: Dict[tuple, Dict[str, float]] = {}

    # ──────────────────────────────────────────────────────────
    # 核心操作
    # ──────────────────────────────────────────────────────────

    def predict_all(self, dt: Optional[float] = None) -> None:
        """对所有活跃轨迹执行 EKF 预测步骤"""
        for track in self._tracks:
            track.predict(dt)

    def update_matched(
        self,
        matches: List[tuple],
        tracks: List[Track],
        detections: List[Detection],
        frame_id: int,
        dt: Optional[float] = None,
    ) -> None:
        """
        更新已匹配的轨迹。

        Args:
            matches: [(track_idx, det_idx), ...] 匹配对
            tracks: 轨迹列表（索引参照）
            detections: 检测列表（索引参照）
            frame_id: 当前帧ID
            dt: 时间步长（传给 track.update 用于 bootstrap）
        """
        for track_idx, det_idx in matches:
            tracks[track_idx].update(detections[det_idx], frame_id, dt=dt)

    def mark_unmatched_missed(self, unmatched_track_indices: List[int]) -> None:
        """
        标记未匹配轨迹为丢失。

        注意：此函数接收相对于 self._tracks 的索引。
        调用前不应修改 self._tracks 的内容，确保索引安全。
        """
        for idx in unmatched_track_indices:
            if 0 <= idx < len(self._tracks):
                self._tracks[idx].mark_missed()

    def _birth_key(self, det: Detection) -> tuple:
        return (det.class_id, round(det.cx / 30), round(det.cy / 30))

    def _has_nearby_recoverable_lost(self, det: Detection) -> bool:
        """附近是否存在短时 Lost 同类轨迹（窗口 4 帧，距离阈值 max(45, 0.65*diag)）"""
        det_diag = math.sqrt(det.w * det.w + det.h * det.h)
        dist_threshold = max(45.0, 0.65 * det_diag)
        for track in self._tracks:
            if not track.is_lost:
                continue
            if track.class_id != det.class_id:
                continue
            if track.time_since_update > 4:
                continue
            track_cx, track_cy = track.get_center()
            dx = track_cx - det.cx
            dy = track_cy - det.cy
            if math.sqrt(dx * dx + dy * dy) < dist_threshold:
                return True
        return False

    def create_new_tracks(
        self,
        unmatched_det_indices: List[int],
        detections: List[Detection],
        frame_id: int,
    ) -> None:
        """
        三段式出生逻辑：
          - score < min_create_score：丢弃
          - score >= 0.78：立即出生
          - 其余：连续两帧确认后出生；附近有 Lost 时额外要求 best_score >= 0.66
        """
        seen_keys: set = set()

        for idx in unmatched_det_indices:
            det = detections[idx]

            # 第一段：低置信丢弃
            if det.score < self.min_create_score:
                key = self._birth_key(det)
                self._pending_births.pop(key, None)
                continue

            # 第二段：高置信立即出生
            if det.score >= 0.78:
                key = self._birth_key(det)
                self._pending_births.pop(key, None)
                self._create_track(det, frame_id)
                continue

            # 第三段：中等置信两帧确认
            key = self._birth_key(det)
            seen_keys.add(key)

            entry = self._pending_births.get(key)
            if entry is None:
                self._pending_births[key] = {
                    "count": 1,
                    "last_frame": float(frame_id),
                    "best_score": det.score,
                }
                continue

            # 更新 pending
            if frame_id - entry["last_frame"] <= 1:
                entry["count"] += 1
            else:
                entry["count"] = 1
            entry["last_frame"] = float(frame_id)
            entry["best_score"] = max(entry["best_score"], det.score)

            if entry["count"] < 2:
                continue

            # 附近有 Lost 时额外要求 best_score >= 0.66
            if self._has_nearby_recoverable_lost(det) and entry["best_score"] < 0.66:
                logger.debug(
                    f"[延迟出生] key={key} nearby_lost=True best_score={entry['best_score']:.2f} 继续等待"
                )
                continue

            del self._pending_births[key]
            self._create_track(det, frame_id)

        # 清理过期 pending（超过 2 帧未出现）
        stale = [k for k, v in self._pending_births.items()
                 if frame_id - v["last_frame"] > 2]
        for k in stale:
            del self._pending_births[k]

    def cleanup(self) -> None:
        """删除 Removed 状态的轨迹"""
        before = len(self._tracks)
        self._tracks = apply_lifecycle(self._tracks)
        removed = before - len(self._tracks)
        if removed > 0:
            logger.debug(f"清理了 {removed} 条已删除轨迹，当前活跃: {len(self._tracks)}")

    # ──────────────────────────────────────────────────────────
    # 查询接口
    # ──────────────────────────────────────────────────────────

    @property
    def tracks(self) -> List[Track]:
        """所有活跃轨迹（不含 Removed）"""
        return self._tracks

    def get_confirmed(self) -> List[Track]:
        return [t for t in self._tracks if t.is_confirmed]

    def get_tentative(self) -> List[Track]:
        return [t for t in self._tracks if t.is_tentative]

    def get_lost(self) -> List[Track]:
        return [t for t in self._tracks if t.is_lost]

    def get_active(self) -> List[Track]:
        """返回 Confirmed + Tentative 轨迹（用于可视化输出）"""
        return [t for t in self._tracks if not t.is_lost and not t.is_deleted]

    def reset(self) -> None:
        """重置所有轨迹"""
        self._tracks.clear()
        Track.reset_id_counter()

    def update_adaptive_controller(self, adaptive_noise_cfg: Optional[dict]) -> None:
        """
        热切换自适应噪声策略。
        替换共享 controller，并将新 controller 传播到所有现存 track。
        已有 track 的 adaptive_state（NIS 历史等）保留不清空。
        """
        new_ctrl = make_adaptive_controller(adaptive_noise_cfg)
        self._adaptive_ctrl = new_ctrl
        for track in self._tracks:
            track._adaptive_ctrl = new_ctrl
        logger.info(
            f"自适应噪声策略已热切换: enabled={new_ctrl.cfg.enabled} "
            f"r_adapt={new_ctrl.cfg.r_adapt_on} q_adapt={new_ctrl.cfg.q_adapt_on} "
            f"robust={new_ctrl.cfg.robust_on}"
        )

    # ──────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────

    def _create_track(self, detection: Detection, frame_id: int) -> Track:
        """创建新轨迹并初始化 EKF"""
        ekf = ExtendedKalmanFilter(**self._ekf_params)
        z = detection.to_measurement(anchor_mode=self.anchor_mode)
        ekf.initialize(z, score=detection.score, **self._init_cov_params)

        track = Track(
            detection=detection,
            ekf=ekf,
            n_init=self.n_init,
            max_age=self.max_age,
            frame_id=frame_id,
            anchor_mode=self.anchor_mode,
            adaptive_controller=self._adaptive_ctrl,
        )
        self._tracks.append(track)
        logger.debug(
            f"创建新轨迹 ID={track.track_id} | 类别={detection.class_name} | 帧={frame_id}"
        )
        return track
