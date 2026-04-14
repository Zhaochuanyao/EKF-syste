"""
端到端冒烟测试 - 用伪造数据跑通完整流程
"""

import numpy as np
import pytest

from src.ekf_mot.tracking.track import Track
from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
from src.ekf_mot.tracking.track_state import TrackState
from src.ekf_mot.prediction.trajectory_predictor import TrajectoryPredictor
from src.ekf_mot.core.types import Detection


def make_det(x1, y1, x2, y2, score=0.9, class_id=0):
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        score=score, class_id=class_id, class_name="car",
    )


@pytest.fixture(autouse=True)
def reset_ids():
    Track.reset_id_counter()
    yield


class TestSmokePipeline:
    def test_full_pipeline_runs(self):
        """完整流程不报错"""
        tracker = MultiObjectTracker(n_init=2, max_age=5, dt=0.04)
        predictor = TrajectoryPredictor(future_steps=[1, 5, 10])

        for frame_id in range(15):
            dets = [
                make_det(100 + frame_id * 3, 100, 160 + frame_id * 3, 150),
                make_det(300 + frame_id * 2, 200, 360 + frame_id * 2, 260),
            ]
            active = tracker.step(dets, frame_id)

            for track in active:
                if track.is_confirmed:
                    future = predictor.predict_track(track)
                    assert isinstance(future, dict)

    def test_tracks_confirmed_after_n_init(self):
        """n_init 帧后应有确认轨迹"""
        tracker = MultiObjectTracker(n_init=3, max_age=10, dt=0.04)
        for frame_id in range(5):
            dets = [make_det(100, 100, 150, 150)]
            tracker.step(dets, frame_id)

        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) >= 1

    def test_prediction_steps(self):
        """预测步数应与配置一致"""
        tracker = MultiObjectTracker(n_init=2, max_age=10, dt=0.04)
        predictor = TrajectoryPredictor(future_steps=[1, 5, 10])

        for frame_id in range(5):
            dets = [make_det(100 + frame_id * 5, 100, 160 + frame_id * 5, 150)]
            tracker.step(dets, frame_id)

        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) >= 1

        future = predictor.predict_track(confirmed[0])
        assert set(future.keys()) == {1, 5, 10}

    def test_track_id_unique(self):
        """每条轨迹 ID 应唯一"""
        tracker = MultiObjectTracker(n_init=1, max_age=10, dt=0.04)
        for frame_id in range(5):
            dets = [
                make_det(100, 100, 150, 150),
                make_det(300, 200, 360, 260),
                make_det(500, 300, 560, 360),
            ]
            tracker.step(dets, frame_id)

        all_tracks = tracker.manager.tracks
        ids = [t.track_id for t in all_tracks]
        assert len(ids) == len(set(ids))

    def test_lost_track_recovery(self):
        """丢失后重新出现的目标应能恢复"""
        tracker = MultiObjectTracker(n_init=2, max_age=5, dt=0.04)

        # 建立轨迹
        for frame_id in range(4):
            dets = [make_det(100, 100, 150, 150)]
            tracker.step(dets, frame_id)

        # 丢失 2 帧
        for frame_id in range(4, 6):
            tracker.step([], frame_id)

        # 重新出现
        dets = [make_det(115, 100, 165, 150)]  # 稍微移动
        active = tracker.step(dets, 6)
        assert len(active) >= 1

    def test_no_detections_tracks_age(self):
        """无检测时轨迹应正常老化"""
        tracker = MultiObjectTracker(n_init=2, max_age=3, dt=0.04)

        for frame_id in range(4):
            dets = [make_det(100, 100, 150, 150)]
            tracker.step(dets, frame_id)

        # 无检测，轨迹应逐渐老化直到删除
        for frame_id in range(4, 10):
            tracker.step([], frame_id)

        # max_age=3，超过后应被删除
        assert len(tracker.manager.tracks) == 0

    def test_future_points_are_valid(self):
        """预测点坐标应为有限数值"""
        tracker = MultiObjectTracker(n_init=2, max_age=10, dt=0.04)
        predictor = TrajectoryPredictor(future_steps=[1, 5, 10])

        for frame_id in range(5):
            dets = [make_det(100 + frame_id * 5, 100, 160 + frame_id * 5, 150)]
            tracker.step(dets, frame_id)

        for track in tracker.get_confirmed_tracks():
            future = predictor.predict_track(track)
            for step, (cx, cy) in future.items():
                assert np.isfinite(cx), f"step={step} cx={cx} 不是有限数"
                assert np.isfinite(cy), f"step={step} cy={cy} 不是有限数"
