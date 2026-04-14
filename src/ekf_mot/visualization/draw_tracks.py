"""
绘制历史轨迹线
"""

from typing import List, Tuple
import cv2
import numpy as np

from ..core.constants import COLOR_CONFIRMED, COLOR_TENTATIVE, COLOR_LOST
from ..tracking.track import Track
from ..tracking.track_state import TrackState


def _track_color(track: Track) -> Tuple[int, int, int]:
    """根据轨迹状态返回颜色"""
    if track.state == TrackState.Confirmed:
        # 用 track_id 生成固定颜色，便于区分不同目标
        hue = (track.track_id * 37) % 180
        hsv = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    elif track.state == TrackState.Tentative:
        return COLOR_TENTATIVE
    else:
        return COLOR_LOST


def draw_track_history(
    frame: np.ndarray,
    track: Track,
    max_len: int = 30,
    thickness: int = 2,
) -> np.ndarray:
    """
    绘制单条轨迹的历史轨迹线。

    Args:
        frame: BGR 图像
        track: 轨迹对象
        max_len: 最多显示的历史点数
        thickness: 线宽

    Returns:
        绘制后的图像
    """
    history = track.history[-max_len:]
    if len(history) < 2:
        return frame

    color = _track_color(track)

    for i in range(1, len(history)):
        pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
        pt2 = (int(history[i][0]), int(history[i][1]))
        # 越近的点越粗
        alpha = i / len(history)
        t = max(1, int(thickness * alpha))
        cv2.line(frame, pt1, pt2, color, t)

    return frame


def draw_all_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    max_len: int = 30,
    thickness: int = 2,
    draw_bbox: bool = True,
    draw_id: bool = True,
    draw_score: bool = True,
    font_scale: float = 0.5,
) -> np.ndarray:
    """绘制所有轨迹的历史线和检测框"""
    from .draw_bbox import draw_track_bbox

    for track in tracks:
        color = _track_color(track)
        draw_track_history(frame, track, max_len, thickness)
        if draw_bbox:
            draw_track_bbox(
                frame,
                track.get_bbox(),
                track.track_id,
                track.class_name,
                track.score,
                color=color,
                thickness=thickness,
                draw_id=draw_id,
                draw_score=draw_score,
                font_scale=font_scale,
            )
    return frame
