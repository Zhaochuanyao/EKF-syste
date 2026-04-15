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

        self._tracks: List[Track] = []

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

    def _has_nearby_recoverable_lost(self, det: Detection) -> bool:
        """
        判断是否存在"短时 Lost 且同类别"的轨迹位于检测框附近。

        若存在，本帧先不新建轨迹，让 Stage A2 在下一帧有机会恢复。
        避免旧轨迹还未恢复就被新 ID 抢占。

        判定条件（同时满足）：
          - track.is_lost
          - track.class_id == det.class_id
          - track.time_since_update <= 5
          - 欧氏距离 < max(40.0, 0.6 * diag(det))
        """
        det_diag = math.sqrt(det.w * det.w + det.h * det.h)
        dist_threshold = max(40.0, 0.6 * det_diag)
        for track in self._tracks:
            if not track.is_lost:
                continue
            if track.class_id != det.class_id:
                continue
            if track.time_since_update > 5:
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
        为未匹配的检测框创建新轨迹。

        新增"延迟出生"保护：
          - 低于 min_create_score 的检测框跳过（噪声抑制）
          - 附近存在短时 Lost 同类轨迹时，本帧暂不新建（防止旧轨迹
            尚未恢复就被新 ID 抢占，减少轨迹碎片化）
        """
        for idx in unmatched_det_indices:
            det = detections[idx]
            if det.score < self.min_create_score:
                continue
            if self._has_nearby_recoverable_lost(det):
                logger.debug(
                    f"[延迟出生] 检测 score={det.score:.2f} class={det.class_name} "
                    f"附近有 Lost 轨迹，本帧跳过新建"
                )
                continue
            self._create_track(det, frame_id)

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
        )
        self._tracks.append(track)
        logger.debug(
            f"创建新轨迹 ID={track.track_id} | 类别={detection.class_name} | 帧={frame_id}"
        )
        return track
