"""
测试数据关联模块
"""

import numpy as np
import pytest

from src.ekf_mot.tracking.track import Track
from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
from src.ekf_mot.tracking.association import associate, hungarian_match
from src.ekf_mot.core.types import Detection


def make_detection(x1, y1, x2, y2, score=0.9, class_id=0):
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        score=score,
        class_id=class_id,
        class_name="person",
    )


def make_tracker_with_tracks(n_frames=3):
    """创建一个已有确认轨迹的跟踪器"""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(n_init=2, max_age=5, dt=0.04)
    for frame_id in range(n_frames):
        dets = [
            make_detection(100, 100, 150, 150),
            make_detection(300, 200, 360, 260),
        ]
        tracker.step(dets, frame_id)
    return tracker


class TestHungarianMatch:
    def test_perfect_match(self):
        """完美匹配：代价矩阵对角线最小"""
        cost = np.array([[0.1, 0.9], [0.9, 0.1]])
        matches, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.5)
        assert (0, 0) in matches
        assert (1, 1) in matches
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 0

    def test_threshold_filtering(self):
        """超过阈值的匹配应被过滤"""
        cost = np.array([[0.8, 0.9], [0.9, 0.8]])
        matches, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.5)
        assert len(matches) == 0
        assert len(unmatched_t) == 2
        assert len(unmatched_d) == 2

    def test_empty_cost_matrix(self):
        cost = np.empty((0, 3))
        matches, unmatched_t, unmatched_d = hungarian_match(cost)
        assert matches == []
        assert unmatched_t == []
        assert unmatched_d == [0, 1, 2]

    def test_more_tracks_than_dets(self):
        cost = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
        matches, unmatched_t, unmatched_d = hungarian_match(cost, threshold=0.5)
        assert len(matches) == 2
        assert len(unmatched_t) == 1


class TestAssociate:
    def test_basic_association(self):
        """基础关联：相同位置的检测应匹配"""
        tracker = make_tracker_with_tracks(n_frames=3)
        tracks = tracker.manager.tracks

        # 检测框与轨迹位置相同
        dets = [
            make_detection(100, 100, 150, 150),
            make_detection(300, 200, 360, 260),
        ]
        matches, unmatched_t, unmatched_d = associate(
            tracks, dets, iou_threshold=0.3
        )
        assert len(matches) == 2
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 0

    def test_no_overlap_no_match(self):
        """完全不重叠的检测框不应匹配"""
        tracker = make_tracker_with_tracks(n_frames=3)
        tracks = tracker.manager.tracks

        # 检测框在完全不同的位置
        dets = [
            make_detection(500, 500, 550, 550),
            make_detection(600, 600, 660, 660),
        ]
        matches, unmatched_t, unmatched_d = associate(
            tracks, dets, iou_threshold=0.3, gating_threshold=9.4877
        )
        assert len(matches) == 0

    def test_empty_detections(self):
        tracker = make_tracker_with_tracks(n_frames=3)
        tracks = tracker.manager.tracks
        matches, unmatched_t, unmatched_d = associate(tracks, [])
        assert matches == []
        assert len(unmatched_t) == len(tracks)
        assert unmatched_d == []
