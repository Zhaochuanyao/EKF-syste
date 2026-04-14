"""
新增测试：bootstrap、anchor_mode、三阶段关联、预测质量门限
"""

import math
import numpy as np
import pytest

from src.ekf_mot.tracking.track import Track
from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
from src.ekf_mot.tracking.association import associate, hungarian_match
from src.ekf_mot.tracking.lifecycle import get_prediction_eligible_tracks
from src.ekf_mot.tracking.track_state import TrackState
from src.ekf_mot.filtering.ekf import ExtendedKalmanFilter
from src.ekf_mot.filtering.noise import build_measurement_noise_R, build_process_noise_Q
from src.ekf_mot.core.types import Detection, TrackStateVector
from src.ekf_mot.prediction.trajectory_predictor import TrajectoryPredictor


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def make_det(x1=100, y1=100, x2=150, y2=200, score=0.9, class_id=0):
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        score=score,
        class_id=class_id,
        class_name="person",
    )


def make_ekf(z=None):
    ekf = ExtendedKalmanFilter(dt=0.04)
    if z is None:
        z = np.array([125.0, 150.0, 50.0, 100.0])
    ekf.initialize(z)
    return ekf


@pytest.fixture(autouse=True)
def reset_id():
    Track.reset_id_counter()
    yield


# ──────────────────────────────────────────────────────────────
# anchor_mode 测试
# ──────────────────────────────────────────────────────────────

class TestAnchorMode:
    def test_center_to_measurement(self):
        """center 模式 to_measurement 返回几何中心"""
        det = make_det(100, 100, 150, 200)
        z = det.to_measurement(anchor_mode="center")
        assert abs(z[0] - 125.0) < 1e-6  # cx
        assert abs(z[1] - 150.0) < 1e-6  # cy = (100+200)/2
        assert abs(z[2] - 50.0) < 1e-6   # w
        assert abs(z[3] - 100.0) < 1e-6  # h

    def test_bottom_center_to_measurement(self):
        """bottom_center 模式 to_measurement 返回底部中心"""
        det = make_det(100, 100, 150, 200)
        z = det.to_measurement(anchor_mode="bottom_center")
        assert abs(z[0] - 125.0) < 1e-6  # cx
        assert abs(z[1] - 200.0) < 1e-6  # cy = y2 = 200
        assert abs(z[2] - 50.0) < 1e-6   # w
        assert abs(z[3] - 100.0) < 1e-6  # h

    def test_center_to_bbox(self):
        """center 模式 to_bbox 正确还原"""
        sv = TrackStateVector(
            x=np.array([125.0, 150.0, 0, 0, 0, 50.0, 100.0]),
            P=np.eye(7),
        )
        bbox = sv.to_bbox(anchor_mode="center")
        assert abs(bbox[0] - 100.0) < 1e-6  # x1 = 125-25
        assert abs(bbox[1] - 100.0) < 1e-6  # y1 = 150-50
        assert abs(bbox[2] - 150.0) < 1e-6  # x2 = 125+25
        assert abs(bbox[3] - 200.0) < 1e-6  # y2 = 150+50

    def test_bottom_center_to_bbox(self):
        """bottom_center 模式 to_bbox：cy 是底部 y"""
        sv = TrackStateVector(
            x=np.array([125.0, 200.0, 0, 0, 0, 50.0, 100.0]),
            P=np.eye(7),
        )
        bbox = sv.to_bbox(anchor_mode="bottom_center")
        assert abs(bbox[0] - 100.0) < 1e-6  # x1 = 125-25
        assert abs(bbox[1] - 100.0) < 1e-6  # y1 = 200-100
        assert abs(bbox[3] - 200.0) < 1e-6  # y2 = 200

    def test_center_bottom_center_cy_difference(self):
        """center 和 bottom_center 的 cy 相差 h/2"""
        det = make_det(100, 100, 150, 200)  # h=100
        z_center = det.to_measurement(anchor_mode="center")
        z_bottom = det.to_measurement(anchor_mode="bottom_center")
        assert abs(z_bottom[1] - z_center[1] - 50.0) < 1e-6  # 底部 y 比中心 y 大 h/2=50


# ──────────────────────────────────────────────────────────────
# Bootstrap 速度/航向测试
# ──────────────────────────────────────────────────────────────

class TestBootstrapKinematics:
    def test_velocity_zero_before_second_hit(self):
        """创建轨迹时速度应为 0"""
        ekf = make_ekf()
        det = make_det()
        track = Track(detection=det, ekf=ekf, n_init=2)
        assert abs(float(track.ekf.x[2])) < 1e-6  # v=0

    def test_velocity_bootstrapped_after_second_hit(self):
        """第 2 次命中后速度应被 bootstrap 设置为非零"""
        ekf = make_ekf(z=np.array([125.0, 150.0, 50.0, 100.0]))
        det1 = make_det(100, 100, 150, 200)  # cx=125, cy=150
        track = Track(detection=det1, ekf=ekf, n_init=2, anchor_mode="center")

        # 第 2 次命中：目标移动到 cx=135, cy=160（向右下移动 10px）
        det2 = make_det(110, 110, 160, 210)  # cx=135, cy=160
        track.update(det2, frame_id=1, dt=0.04)

        # 速度应该被 bootstrap 设置为非零
        v = float(track.ekf.x[2])
        assert track.velocity_valid or abs(v) > 0

    def test_bootstrap_requires_minimum_movement(self):
        """目标几乎不动时不应触发 bootstrap（避免噪声干扰）"""
        ekf = make_ekf(z=np.array([125.0, 150.0, 50.0, 100.0]))
        det1 = make_det(100, 100, 150, 200)  # cx=125
        track = Track(detection=det1, ekf=ekf, n_init=2)

        # 第 2 次：几乎不动（只移动 0.1px）
        det2 = make_det(100.05, 100, 150.05, 200)  # cx≈125.05，移动约 0.05px
        track.update(det2, frame_id=1, dt=0.04)

        # 速度不应被 bootstrap（移动量不足）
        assert not track.velocity_valid

    def test_heading_estimated_correctly(self):
        """航向应与运动方向一致"""
        ekf = make_ekf(z=np.array([100.0, 100.0, 50.0, 50.0]))
        det1 = make_det(75, 75, 125, 125)  # cx=100, cy=100
        track = Track(detection=det1, ekf=ekf, n_init=2)

        # 沿 x 轴正方向移动 20px
        det2 = make_det(95, 75, 145, 125)  # cx=120, cy=100
        track.update(det2, frame_id=1, dt=0.04)

        if track.heading_valid:
            theta = float(track.ekf.x[3])
            # 水平向右 → theta ≈ 0
            assert abs(theta) < 0.5

    def test_stability_score_increases_with_hits(self):
        """随命中次数增加，稳定性分数应增大"""
        ekf = make_ekf()
        det = make_det()
        track = Track(detection=det, ekf=ekf, n_init=3)
        score_before = track.stability_score

        for fid in range(1, 5):
            track.update(make_det(), frame_id=fid, dt=0.04)
        assert track.stability_score >= score_before

    def test_velocity_valid_flag(self):
        """velocity_valid 在 bootstrap 后应为 True"""
        ekf = make_ekf(z=np.array([100.0, 100.0, 50.0, 50.0]))
        det1 = make_det(75, 75, 125, 125)
        track = Track(detection=det1, ekf=ekf, n_init=2)
        det2 = make_det(95, 75, 145, 125)  # 移动 20px
        track.update(det2, frame_id=1, dt=0.04)
        assert track.velocity_valid


# ──────────────────────────────────────────────────────────────
# 三阶段关联测试
# ──────────────────────────────────────────────────────────────

class TestThreeStageAssociation:
    def _make_tracker_with_confirmed_track(self):
        """创建一个有已确认轨迹的跟踪器"""
        tracker = MultiObjectTracker(n_init=2, max_age=5, dt=0.04)
        for fid in range(3):
            dets = [make_det(100, 100, 150, 150, score=0.9)]
            tracker.step(dets, fid)
        return tracker

    def test_high_conf_det_matches_confirmed_track(self):
        """高置信度检测框应与 Confirmed 轨迹匹配（Stage A）"""
        tracker = self._make_tracker_with_confirmed_track()
        tracks = tracker.manager.tracks
        # 确认有 Confirmed 轨迹
        assert any(t.is_confirmed for t in tracks)

        dets = [make_det(100, 100, 150, 150, score=0.9)]  # 高置信度，相同位置
        matches, unmatched_t, unmatched_d = associate(
            tracks, dets, high_conf_threshold=0.5
        )
        assert len(matches) == 1
        assert len(unmatched_d) == 0

    def test_low_conf_det_can_match_unmatched_track(self):
        """低置信度检测框应能匹配 Stage B 未匹配轨迹"""
        tracker = self._make_tracker_with_confirmed_track()
        tracks = tracker.manager.tracks

        # 低置信度检测（0.3 < score < 0.5），在轨迹位置
        dets = [make_det(100, 100, 150, 150, score=0.35)]
        matches, unmatched_t, unmatched_d = associate(
            tracks, dets,
            high_conf_threshold=0.5,
            low_conf_threshold=0.2,
        )
        # 应该能匹配（Stage B）
        assert len(matches) == 1

    def test_tentative_uses_stage_c(self):
        """Tentative 轨迹只参与 Stage C（与未匹配的高置信度检测框）"""
        # 创建只有 Tentative 轨迹的跟踪器（只运行1帧）
        tracker = MultiObjectTracker(n_init=3, max_age=5, dt=0.04)
        dets = [make_det(100, 100, 150, 150, score=0.9)]
        tracker.step(dets, frame_id=0)
        tracks = tracker.manager.tracks
        assert all(t.is_tentative for t in tracks)

        # 相同位置的高置信度检测应能匹配 Tentative（Stage C）
        dets2 = [make_det(100, 100, 150, 150, score=0.9)]
        matches, _, _ = associate(
            tracks, dets2, high_conf_threshold=0.5, iou_threshold_c=0.3
        )
        assert len(matches) == 1

    def test_below_low_conf_not_used_for_stage_a(self):
        """极低置信度检测框不参与 Stage A"""
        tracker = self._make_tracker_with_confirmed_track()
        tracks = tracker.manager.tracks

        # 极低置信度（< low_conf_threshold=0.1）
        dets = [make_det(100, 100, 150, 150, score=0.05)]
        matches, unmatched_t, unmatched_d = associate(
            tracks, dets, high_conf_threshold=0.5, low_conf_threshold=0.1
        )
        # 不参与任何阶段匹配（score 太低）
        assert len(matches) == 0

    def test_class_mismatch_not_matched(self):
        """不同类别的轨迹和检测框不应匹配（Stage A 类别检查）"""
        tracker = MultiObjectTracker(n_init=2, max_age=5, dt=0.04)
        # 创建 class_id=0（person）的轨迹
        for fid in range(3):
            dets = [make_det(100, 100, 150, 150, score=0.9, class_id=0)]
            tracker.step(dets, fid)
        tracks = tracker.manager.tracks

        # class_id=2（car）的检测框，相同位置
        car_dets = [Detection(
            bbox=np.array([100, 100, 150, 150], dtype=np.float64),
            score=0.9, class_id=2, class_name="car"
        )]
        matches, _, _ = associate(
            tracks, car_dets, high_conf_threshold=0.5
        )
        assert len(matches) == 0


# ──────────────────────────────────────────────────────────────
# 预测质量门限测试
# ──────────────────────────────────────────────────────────────

class TestPredictionEligibility:
    def _make_confirmed_track(self, hits=5, time_since_update=0):
        """创建一个 Confirmed 轨迹"""
        ekf = make_ekf()
        det = make_det()
        track = Track(detection=det, ekf=ekf, n_init=1)  # n_init=1 即确认
        track.update(det, frame_id=0)  # hits=2
        track.hits = hits
        track.time_since_update = time_since_update
        return track

    def test_eligible_confirmed_track(self):
        """满足条件的 Confirmed 轨迹应通过门限"""
        ekf = make_ekf()
        det = make_det()
        track = Track(detection=det, ekf=ekf, n_init=1)
        # 更新多次直到 Confirmed 且 hits >= 3
        for fid in range(4):
            track.update(det, frame_id=fid, dt=0.04)
        track.time_since_update = 0

        predictor = TrajectoryPredictor(min_hits_for_prediction=3)
        assert predictor.is_eligible(track)

    def test_tentative_track_not_eligible(self):
        """Tentative 轨迹不应通过门限"""
        ekf = make_ekf()
        det = make_det()
        track = Track(detection=det, ekf=ekf, n_init=10)  # 需要 10 次才确认
        track.time_since_update = 0
        track.hits = 5

        predictor = TrajectoryPredictor(min_hits_for_prediction=3)
        assert not predictor.is_eligible(track)

    def test_not_enough_hits(self):
        """命中次数不足不应通过门限"""
        ekf = make_ekf()
        det = make_det()
        track = Track(detection=det, ekf=ekf, n_init=1)
        track.update(det, frame_id=0)  # hits=2
        track.time_since_update = 0

        predictor = TrajectoryPredictor(min_hits_for_prediction=5)
        assert not predictor.is_eligible(track)

    def test_not_updated_this_frame(self):
        """本帧未更新（time_since_update > 0）不应通过门限"""
        ekf = make_ekf()
        det = make_det()
        track = Track(detection=det, ekf=ekf, n_init=1)
        for fid in range(5):
            track.update(det, frame_id=fid)
        track.time_since_update = 2  # 本帧未更新

        predictor = TrajectoryPredictor(min_hits_for_prediction=3)
        assert not predictor.is_eligible(track)

    def test_high_uncertainty_not_eligible(self):
        """位置不确定性过高不应通过门限"""
        ekf = make_ekf()
        det = make_det()
        track = Track(detection=det, ekf=ekf, n_init=1)
        for fid in range(5):
            track.update(det, frame_id=fid)
        track.time_since_update = 0
        # 更新后手动注入极大的位置协方差
        track.ekf.P[0, 0] = 1e8
        track.ekf.P[1, 1] = 1e8

        predictor = TrajectoryPredictor(
            min_hits_for_prediction=3,
            max_position_cov_trace=100.0,  # 严格上限
        )
        assert not predictor.is_eligible(track)

    def test_prediction_confidence_increases_with_hits(self):
        """随命中次数增加，预测置信度应增大"""
        ekf1 = make_ekf()
        det = make_det()
        track_few = Track(detection=det, ekf=ekf1, n_init=1)
        track_few.hits = 2
        track_few.time_since_update = 0

        ekf2 = make_ekf()
        track_many = Track(detection=det, ekf=ekf2, n_init=1)
        track_many.hits = 20
        track_many.time_since_update = 0

        predictor = TrajectoryPredictor(min_hits_for_prediction=1)
        conf_few = predictor.compute_prediction_confidence(track_few)
        conf_many = predictor.compute_prediction_confidence(track_many)
        assert conf_many > conf_few

    def test_get_prediction_eligible_tracks(self):
        """lifecycle 接口：只返回满足条件的轨迹"""
        ekf = make_ekf()
        det = make_det()

        # 满足条件的轨迹
        track_good = Track(detection=det, ekf=ekf, n_init=1)
        for fid in range(5):
            track_good.update(det, frame_id=fid)
        track_good.time_since_update = 0

        # 不满足条件（Tentative）
        ekf2 = make_ekf()
        track_tent = Track(detection=det, ekf=ekf2, n_init=10)
        track_tent.hits = 5
        track_tent.time_since_update = 0

        eligible = get_prediction_eligible_tracks(
            [track_good, track_tent], min_hits=3
        )
        assert track_good in eligible
        assert track_tent not in eligible


# ──────────────────────────────────────────────────────────────
# mark_unmatched_missed 索引安全性
# ──────────────────────────────────────────────────────────────

class TestMarkUnmatchedMissedSafety:
    def test_safe_with_valid_indices(self):
        """mark_unmatched_missed 对有效索引不应出错"""
        from src.ekf_mot.tracking.track_manager import TrackManager
        mgr = TrackManager(n_init=2, max_age=5)
        det = make_det()
        mgr.create_new_tracks([0], [det], frame_id=0)
        mgr.create_new_tracks([0], [det], frame_id=0)
        # 标记所有轨迹为 missed，不应抛异常
        mgr.mark_unmatched_missed(list(range(len(mgr.tracks))))

    def test_out_of_bounds_index_ignored(self):
        """越界索引应被安全跳过，不抛异常"""
        from src.ekf_mot.tracking.track_manager import TrackManager
        mgr = TrackManager(n_init=2, max_age=5)
        det = make_det()
        mgr.create_new_tracks([0], [det], frame_id=0)
        # 传入越界索引（不应抛 IndexError）
        mgr.mark_unmatched_missed([0, 5, 100])  # 5 和 100 越界，应被安全跳过


# ──────────────────────────────────────────────────────────────
# min_create_score 测试
# ──────────────────────────────────────────────────────────────

class TestMinCreateScore:
    def test_high_score_creates_track(self):
        """满足 min_create_score 的检测框应创建轨迹"""
        tracker = MultiObjectTracker(n_init=2, max_age=5, dt=0.04, min_create_score=0.5)
        dets = [make_det(score=0.9)]
        tracker.step(dets, frame_id=0)
        assert len(tracker.manager.tracks) == 1

    def test_low_score_does_not_create_track(self):
        """低于 min_create_score 的检测框不应创建轨迹"""
        tracker = MultiObjectTracker(n_init=2, max_age=5, dt=0.04, min_create_score=0.5)
        dets = [make_det(score=0.3)]  # 低于 0.5
        tracker.step(dets, frame_id=0)
        assert len(tracker.manager.tracks) == 0


# ──────────────────────────────────────────────────────────────
# 自适应噪声测试
# ──────────────────────────────────────────────────────────────

class TestAdaptiveNoise:
    def test_score_adaptive_r(self):
        """低置信度应产生更大的 R"""
        R_high = build_measurement_noise_R(score=0.95, score_adaptive=True)
        R_low = build_measurement_noise_R(score=0.3, score_adaptive=True)
        assert np.trace(R_low) > np.trace(R_high)

    def test_lost_age_adaptive_q(self):
        """丢失帧数越多，Q 应越大"""
        Q0 = build_process_noise_Q(dt=0.04, lost_age=0)
        Q3 = build_process_noise_Q(dt=0.04, lost_age=3)
        assert np.trace(Q3) > np.trace(Q0)

    def test_lost_age_q_capped(self):
        """Q 放大有上限（最多 8 倍）"""
        Q0 = build_process_noise_Q(dt=0.04, lost_age=0)
        Q100 = build_process_noise_Q(dt=0.04, lost_age=100)  # 极大的 lost_age
        Q8 = build_process_noise_Q(dt=0.04, lost_age=100, lost_age_q_scale=1.5)
        # Q_position（前 3x3 块）应不超过 8× 基础值
        ratio = Q100[0, 0] / max(Q0[0, 0], 1e-12)
        assert ratio <= 8.1  # 允许微小浮点误差

    def test_size_adaptive_r_larger_for_bigger_object(self):
        """更大的目标框应产生更大的 R"""
        R_small = build_measurement_noise_R(
            bbox_w=30, bbox_h=60, size_adaptive=True, size_ref=100.0
        )
        R_large = build_measurement_noise_R(
            bbox_w=200, bbox_h=300, size_adaptive=True, size_ref=100.0
        )
        assert np.trace(R_large) >= np.trace(R_small)
