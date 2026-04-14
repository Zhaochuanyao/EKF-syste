"""
Baseline 跟踪器测试 — BaselineTracker / BaselineTrack / 轨迹质量函数
"""

import sys
import math
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.prediction.baseline import (
    BaselineTrack,
    BaselineTracker,
    _compute_jitter,
    _compute_smoothness,
    compute_track_quality,
    _iou,
)


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

class TestIou:
    def test_perfect_overlap(self):
        box = [0.0, 0.0, 10.0, 10.0]
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)


class TestJitterSmoothness:
    def test_zero_jitter_constant_speed(self):
        # 匀速：每帧移动 (1, 0) → 位移全为 1 → std=0
        history = [(float(i), 0.0) for i in range(5)]
        assert _compute_jitter(history) == pytest.approx(0.0, abs=1e-6)

    def test_jitter_variable_speed(self):
        # 速度变化：1, 2, 1, 2 → std > 0
        history = [(0, 0), (1, 0), (3, 0), (4, 0), (6, 0)]
        jitter = _compute_jitter(history)
        assert jitter > 0.0

    def test_zero_smoothness_constant_speed(self):
        history = [(float(i), 0.0) for i in range(5)]
        assert _compute_smoothness(history) == pytest.approx(0.0, abs=1e-6)

    def test_smoothness_with_acceleration(self):
        # 加速：1, 2, 4, 8 → 加速度 1, 2, 4 → mean=2.33
        history = [(0, 0), (1, 0), (3, 0), (7, 0), (15, 0)]
        smooth = _compute_smoothness(history)
        assert smooth > 0.0


class TestComputeTrackQuality:
    def test_empty_history(self):
        q = compute_track_quality([])
        assert q["length"] == 0

    def test_single_point(self):
        q = compute_track_quality([(0.0, 0.0)])
        assert q["length"] == 1
        assert q["jitter"] == 0.0

    def test_two_points(self):
        q = compute_track_quality([(0.0, 0.0), (3.0, 4.0)])
        assert q["avg_speed"] == pytest.approx(5.0, abs=1e-4)

    def test_keys_present(self):
        q = compute_track_quality([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
        for key in ("jitter", "smoothness", "length", "avg_speed"):
            assert key in q


# ══════════════════════════════════════════════════════════════
# BaselineTrack
# ══════════════════════════════════════════════════════════════

class TestBaselineTrack:
    def setup_method(self):
        BaselineTrack.reset_id_counter()

    def test_creation(self):
        t = BaselineTrack([0.0, 0.0, 10.0, 10.0], frame_id=1)
        assert t.track_id == 1
        assert t.hits == 1
        assert len(t.history) == 1

    def test_get_center(self):
        t = BaselineTrack([0.0, 0.0, 10.0, 10.0], frame_id=1)
        cx, cy = t.get_center()
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_update_appends_history(self):
        t = BaselineTrack([0.0, 0.0, 10.0, 10.0], frame_id=1)
        t.update([2.0, 2.0, 12.0, 12.0], frame_id=2)
        assert len(t.history) == 2
        assert t.hits == 2

    def test_predict_linear_two_points(self):
        t = BaselineTrack([0.0, 0.0, 10.0, 10.0], frame_id=1)
        t.update([10.0, 0.0, 20.0, 10.0], frame_id=2)
        # dx=10, dy=0 → step=1: (15+10=25,5), step=2: (25+20=?,5)
        pred = t.predict_linear([1, 2])
        assert len(pred) == 2
        # step=1: cx(15) + 10*1 = 25
        assert pred[1][0] == pytest.approx(25.0, abs=1e-4)

    def test_predict_linear_single_point(self):
        t = BaselineTrack([0.0, 0.0, 10.0, 10.0], frame_id=1)
        pred = t.predict_linear([1, 5])
        # 不足两点 → 返回当前位置的复制
        cx, cy = t.get_center()
        for step in [1, 5]:
            assert pred[step][0] == pytest.approx(cx, abs=1e-4)


# ══════════════════════════════════════════════════════════════
# BaselineTracker
# ══════════════════════════════════════════════════════════════

def _make_detection(bbox, class_id=0, class_name="car", score=0.9):
    """创建模拟 Detection 对象"""
    det = MagicMock()
    det.bbox = np.array(bbox, dtype=np.float64)
    det.class_id = class_id
    det.class_name = class_name
    det.score = score
    return det


class TestBaselineTracker:
    def setup_method(self):
        BaselineTrack.reset_id_counter()

    def test_new_detection_creates_track(self):
        tracker = BaselineTracker(iou_threshold=0.3, max_age=5, min_hits=1)
        dets = [_make_detection([0.0, 0.0, 10.0, 10.0])]
        tracks = tracker.step(dets, frame_id=1)
        assert len(tracks) == 1

    def test_track_confirmed_after_min_hits(self):
        tracker = BaselineTracker(iou_threshold=0.3, max_age=5, min_hits=2)
        det = _make_detection([0.0, 0.0, 10.0, 10.0])
        # 第一帧不确认
        tracks = tracker.step([det], frame_id=1)
        assert not tracks[0].is_confirmed
        # 第二帧确认
        tracks = tracker.step([det], frame_id=2)
        assert tracks[0].is_confirmed

    def test_track_removed_after_max_age(self):
        tracker = BaselineTracker(iou_threshold=0.3, max_age=2, min_hits=1)
        det = _make_detection([0.0, 0.0, 10.0, 10.0])
        tracker.step([det], frame_id=1)
        # 之后 3 帧无检测
        for i in range(3):
            tracks = tracker.step([], frame_id=2 + i)
        assert len(tracks) == 0

    def test_same_detection_matched_to_same_track(self):
        tracker = BaselineTracker(iou_threshold=0.3, max_age=5, min_hits=1)
        det = _make_detection([0.0, 0.0, 10.0, 10.0])
        t1 = tracker.step([det], frame_id=1)[0]
        t2 = tracker.step([det], frame_id=2)[0]
        assert t1.track_id == t2.track_id

    def test_disjoint_detections_create_separate_tracks(self):
        tracker = BaselineTracker(iou_threshold=0.5, max_age=5, min_hits=1)
        d1 = _make_detection([0.0, 0.0, 10.0, 10.0])
        d2 = _make_detection([100.0, 100.0, 110.0, 110.0])
        tracks = tracker.step([d1, d2], frame_id=1)
        assert len(tracks) == 2
        assert tracks[0].track_id != tracks[1].track_id

    def test_get_summary_keys(self):
        tracker = BaselineTracker(iou_threshold=0.3, max_age=5, min_hits=1)
        det = _make_detection([0.0, 0.0, 10.0, 10.0])
        for i in range(3):
            tracker.step([det], frame_id=i + 1)
        s = tracker.get_summary()
        for key in ("num_tracks", "avg_track_length", "avg_jitter", "avg_smoothness"):
            assert key in s

    def test_empty_tracker_summary(self):
        tracker = BaselineTracker()
        s = tracker.get_summary()
        assert s["num_tracks"] == 0
