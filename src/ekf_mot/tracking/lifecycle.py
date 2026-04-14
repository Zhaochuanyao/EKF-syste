"""
轨迹生命周期管理规则
"""

from typing import List
from .track import Track
from .track_state import TrackState


def apply_lifecycle(tracks: List[Track]) -> List[Track]:
    """
    应用生命周期规则，返回仍然活跃的轨迹列表（过滤掉 Removed）。

    规则:
    - Tentative 且 time_since_update > 0 → Removed（未命中即删除，在 mark_missed 中处理）
    - Confirmed/Lost 且 time_since_update > max_age → Removed（在 mark_missed 中处理）
    - 此函数只负责最终过滤 Removed 轨迹
    """
    return [t for t in tracks if t.state != TrackState.Removed]


def get_active_tracks(tracks: List[Track]) -> List[Track]:
    """返回 Confirmed 和 Tentative 轨迹（用于输出/可视化）"""
    return [t for t in tracks if t.state in (TrackState.Confirmed, TrackState.Tentative)]


def get_confirmed_tracks(tracks: List[Track]) -> List[Track]:
    """返回已确认轨迹"""
    return [t for t in tracks if t.state == TrackState.Confirmed]


def get_visible_tracks(tracks: List[Track]) -> List[Track]:
    """
    返回可见轨迹：
    - Confirmed 轨迹（所有，包含 Lost）
    - 最近一帧有命中的 Tentative 轨迹（time_since_update == 0）

    可视化时使用，确保只显示有意义的轨迹。
    """
    result = []
    for t in tracks:
        if t.state == TrackState.Confirmed:
            result.append(t)
        elif t.state == TrackState.Tentative and t.time_since_update == 0:
            result.append(t)
    return result


def get_recoverable_tracks(tracks: List[Track]) -> List[Track]:
    """
    返回可恢复的 Lost 轨迹：
    - 状态为 Lost
    - 丢失帧数 <= max_age（还没超时）

    用于关联阶段，在丢失超时前尝试重新匹配。
    """
    return [
        t for t in tracks
        if t.state == TrackState.Lost and t.time_since_update <= t.max_age
    ]


def get_prediction_eligible_tracks(
    tracks: List[Track],
    min_hits: int = 3,
    max_position_cov_trace: float = 1e6,
) -> List[Track]:
    """
    返回满足预测质量门限的轨迹：
    - 状态为 Confirmed
    - 命中次数 >= min_hits（有足够历史）
    - 本帧刚更新（time_since_update == 0，表示当前帧有检测命中）
    - 位置协方差迹 < max_position_cov_trace（EKF 位置估计稳定）

    仅对满足条件的轨迹做未来预测，确保预测可信度。

    Args:
        tracks: 轨迹列表
        min_hits: 最少命中帧数
        max_position_cov_trace: 位置协方差迹最大允许值（越小越确定）
    """
    result = []
    for t in tracks:
        if not t.is_confirmed:
            continue
        if t.hits < min_hits:
            continue
        if t.time_since_update != 0:
            continue
        pos_cov_trace = t.ekf.get_position_uncertainty_trace()
        if pos_cov_trace > max_position_cov_trace:
            continue
        result.append(t)
    return result
