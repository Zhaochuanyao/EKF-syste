"""
测试轨迹管理器
"""

import numpy as np
import pytest

from src.ekf_mot.tracking.track import Track
from src.ekf_mot.tracking.track_manager import TrackManager
from src.ekf_mot.tracking.track_state import TrackState
from src.ekf_mot.core.types import Detection


def make_det(x1=100, y1=100, x2=150, y2=150, score=0.9):
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        score=score, class_id=0, class_name="person",
    )


@pytest.fixture(autouse=True)
def reset_id():
    Track.reset_id_counter()
    yield


class TestTrackManager:
    def test_create_track(self):
        mgr = TrackManager(n_init=3, max_age=10)
        det = make_det()
        mgr.create_new_tracks([0], [det], frame_id=0)
        assert len(mgr.tracks) == 1
        assert mgr.tracks[0].state == TrackState.Tentative

    def test_track_confirmed_after_n_init(self):
        """连续命中 n_init 帧后轨迹应变为 Confirmed"""
        mgr = TrackManager(n_init=3, max_age=10)
        det = make_det()
        mgr.create_new_tracks([0], [det], frame_id=0)
        track = mgr.tracks[0]

        for frame_id in range(1, 4):
            mgr.predict_all()
            mgr.update_matched([(0, 0)], mgr.tracks, [det], frame_id)

        assert track.state == TrackState.Confirmed

    def test_track_removed_after_max_age(self):
        """超过 max_age 帧未命中后轨迹应被删除"""
        mgr = TrackManager(n_init=1, max_age=3)
        det = make_det()
        mgr.create_new_tracks([0], [det], frame_id=0)
        track = mgr.tracks[0]
        # 先确认
        mgr.predict_all()
        mgr.update_matched([(0, 0)], mgr.tracks, [det], frame_id=1)

        # 连续未命中：每次 cleanup 后重新获取索引
        for _ in range(5):
            mgr.predict_all()
            if mgr.tracks:
                mgr.mark_unmatched_missed(list(range(len(mgr.tracks))))
            mgr.cleanup()

        assert len(mgr.tracks) == 0 or all(t.track_id != track.track_id for t in mgr.tracks)

    def test_tentative_removed_on_miss(self):
        """Tentative 轨迹未命中应立即删除"""
        mgr = TrackManager(n_init=3, max_age=10)
        det = make_det()
        mgr.create_new_tracks([0], [det], frame_id=0)
        assert mgr.tracks[0].state == TrackState.Tentative

        mgr.predict_all()
        mgr.mark_unmatched_missed([0])
        mgr.cleanup()

        assert len(mgr.tracks) == 0

    def test_multiple_tracks(self):
        """多目标同时跟踪"""
        mgr = TrackManager(n_init=2, max_age=10)
        dets = [make_det(100, 100, 150, 150), make_det(300, 300, 350, 350)]
        mgr.create_new_tracks([0, 1], dets, frame_id=0)
        assert len(mgr.tracks) == 2

    def test_reset(self):
        mgr = TrackManager(n_init=2, max_age=10)
        mgr.create_new_tracks([0], [make_det()], frame_id=0)
        mgr.reset()
        assert len(mgr.tracks) == 0
