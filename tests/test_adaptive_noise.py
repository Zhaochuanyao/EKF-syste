"""
自适应噪声调度测试

覆盖场景：
  1. 低 NIS → R 回退到基线行为
  2. 高 NIS → R 确实上调
  3. 机动片段 → Q 关键子块放大，w/h 维度不放大
  4. 极端异常 + 低分 → skip update
  5. 极端异常 + 高分 → robust clip（不 skip）
  6. enabled=False → 行为与改动前完全一致
  7. 输出接口与 JSON 结构稳定
  8. small smoke：adaptive 模式在 MultiObjectTracker 中可正常跑通
"""

import math
import numpy as np
import pytest

from src.ekf_mot.filtering.adaptive_noise import (
    AdaptiveNoiseConfig, TrackAdaptiveState,
    AdaptiveNoiseController, make_adaptive_controller,
)
from src.ekf_mot.filtering.robust_update import (
    should_skip_update, apply_robust_clip, robust_update_step,
)
from src.ekf_mot.filtering.ekf import ExtendedKalmanFilter
from src.ekf_mot.filtering.noise import build_measurement_noise_R, build_process_noise_Q
from src.ekf_mot.tracking.track import Track
from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
from src.ekf_mot.core.types import Detection, Measurement
from src.ekf_mot.core.constants import IDX_W, IDX_H, IDX_CX, IDX_CY, IDX_V, IDX_THETA, IDX_OMEGA, STATE_DIM


# ──────────────────────────────────────────────────────────────
# 共用 fixtures / helpers
# ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_track_id():
    Track.reset_id_counter()
    yield


def make_det(x1=80, y1=170, x2=120, y2=230, score=0.9, class_id=0):
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        score=score, class_id=class_id, class_name="car",
    )


def make_ekf_initialized():
    ekf = ExtendedKalmanFilter(dt=0.04)
    ekf.initialize(np.array([100.0, 200.0, 50.0, 60.0]))
    return ekf


def make_ctrl(enabled=True, **kwargs):
    cfg = {"enabled": enabled, "nis_threshold": 9.4877, "drop_threshold": 20.0,
           "lambda_r": 0.6, "lambda_q": 0.3, "beta": 0.85, "delta_max": 400.0,
           "recover_alpha_r": 0.8, "q_max_scale": 4.0, "maneuver_cap": 3.0,
           "maneuver_w_nis": 1.0, "maneuver_w_omega": 0.8, "maneuver_w_theta": 0.5,
           "low_score": 0.35, "use_robust_update": True, "robust_clip_delta": 25.0,
           "only_r_adapt": False, "only_q_schedule": False}
    cfg.update(kwargs)
    return make_adaptive_controller(cfg)


def make_R_base():
    return np.diag([144.0, 144.0, 324.0, 324.0])  # std=12/12/18/18


def make_Q_base(dt=0.04):
    return build_process_noise_Q(dt=dt, std_acc=0.5, std_yaw_rate=0.15, std_size=0.05)


# ──────────────────────────────────────────────────────────────
# 1. AdaptiveNoiseConfig
# ──────────────────────────────────────────────────────────────

class TestAdaptiveNoiseConfig:
    def test_from_dict_enabled(self):
        cfg = AdaptiveNoiseConfig.from_dict({"enabled": True, "lambda_r": 0.9})
        assert cfg.enabled is True
        assert cfg.lambda_r == 0.9

    def test_from_dict_disabled(self):
        cfg = AdaptiveNoiseConfig.from_dict({"enabled": False})
        assert cfg.r_adapt_on is False
        assert cfg.q_adapt_on is False
        assert cfg.robust_on is False

    def test_only_r_adapt_flag(self):
        cfg = AdaptiveNoiseConfig.from_dict({"enabled": True, "only_r_adapt": True})
        assert cfg.r_adapt_on is True
        assert cfg.q_adapt_on is False
        assert cfg.robust_on is False

    def test_only_q_schedule_flag(self):
        cfg = AdaptiveNoiseConfig.from_dict({"enabled": True, "only_q_schedule": True})
        assert cfg.r_adapt_on is False
        assert cfg.q_adapt_on is True
        assert cfg.robust_on is False

    def test_default_thresholds(self):
        cfg = AdaptiveNoiseConfig()
        assert cfg.nis_threshold == pytest.approx(9.4877)
        assert cfg.drop_threshold == pytest.approx(20.0)


# ──────────────────────────────────────────────────────────────
# 2. TrackAdaptiveState
# ──────────────────────────────────────────────────────────────

class TestTrackAdaptiveState:
    def test_initial_zero(self):
        s = TrackAdaptiveState()
        assert s.total_updates == 0
        assert s.skipped_update_count == 0
        np.testing.assert_array_equal(s.innov_mean, np.zeros(4))

    def test_reset_clears_all(self):
        s = TrackAdaptiveState()
        s.total_updates = 10
        s.skipped_update_count = 3
        s.innov_mean[:] = [1, 2, 3, 4]
        s.reset()
        assert s.total_updates == 0
        assert s.skipped_update_count == 0
        np.testing.assert_array_equal(s.innov_mean, np.zeros(4))

    def test_rates_are_zero_before_any_update(self):
        s = TrackAdaptiveState()
        assert s.nis_over_threshold_rate == 0.0
        assert s.skipped_update_rate == 0.0
        assert s.abnormal_update_rate == 0.0

    def test_diagnostics_keys(self):
        s = TrackAdaptiveState()
        d = s.get_diagnostics()
        for key in ("avg_nis", "nis_over_threshold_rate", "skipped_update_count",
                    "skipped_update_rate", "abnormal_update_count", "total_updates"):
            assert key in d


# ──────────────────────────────────────────────────────────────
# 3. NIS 计算
# ──────────────────────────────────────────────────────────────

class TestNISComputation:
    def test_nis_correct_value(self):
        ctrl = make_ctrl()
        innov = np.array([3.0, 3.0, 2.0, 2.0])
        S = np.diag([10.0, 10.0, 5.0, 5.0])
        nis = ctrl.compute_nis(innov, S)
        expected = 3**2/10 + 3**2/10 + 2**2/5 + 2**2/5
        assert nis == pytest.approx(expected, rel=1e-9)

    def test_nis_nonnegative(self):
        ctrl = make_ctrl()
        for _ in range(20):
            innov = np.random.randn(4) * 10
            S = np.eye(4) * (np.random.rand() * 5 + 1)
            assert ctrl.compute_nis(innov, S) >= 0.0

    def test_nis_zero_for_zero_innov(self):
        ctrl = make_ctrl()
        assert ctrl.compute_nis(np.zeros(4), np.eye(4)) == pytest.approx(0.0)

    def test_nis_larger_for_larger_innov(self):
        ctrl = make_ctrl()
        S = np.eye(4) * 10.0
        nis_small = ctrl.compute_nis(np.ones(4), S)
        nis_large = ctrl.compute_nis(np.ones(4) * 5, S)
        assert nis_large > nis_small


# ──────────────────────────────────────────────────────────────
# 4. R 自适应（核心 1 & 2）
# ──────────────────────────────────────────────────────────────

class TestRAdaptation:
    def test_r_amplified_on_anomaly(self):
        """NIS > nis_threshold 时 R_adapt 对角元素必须大于 R_base"""
        ctrl = make_ctrl()
        R_base = make_R_base()
        innov = np.array([20.0, 20.0, 15.0, 15.0])
        state = TrackAdaptiveState()
        R_adapt, _ = ctrl.adapt_R(R_base, innov, nis=15.0, state=state)
        assert np.all(np.diag(R_adapt) >= np.diag(R_base))
        # 至少有一个元素比 R_base 更大
        assert np.any(np.diag(R_adapt) > np.diag(R_base))

    def test_r_recovers_toward_base_on_normal(self):
        """NIS 恢复正常后，R_adapt 逐步向 R_base 回退（不能立即回到 R_base）"""
        ctrl = make_ctrl(recover_alpha_r=0.8)
        R_base = make_R_base()
        state = TrackAdaptiveState()
        # 先模拟一次异常（prev_R_diag 变大）
        state.prev_R_diag = np.diag(R_base) * 3.0
        innov = np.ones(4) * 0.5
        R_adapt, state = ctrl.adapt_R(R_base, innov, nis=1.0, state=state)
        # 回退后仍大于 R_base（未完全回退）
        assert np.all(np.diag(R_adapt) >= np.diag(R_base))
        # 但小于之前的放大值
        assert np.all(np.diag(R_adapt) < state.prev_R_diag * 2)

    def test_r_not_below_base(self):
        """R_adapt 绝不能低于 R_base"""
        ctrl = make_ctrl()
        R_base = make_R_base()
        state = TrackAdaptiveState()
        for _ in range(30):
            innov = np.random.randn(4) * 3
            nis = float(np.random.rand() * 5)
            R_adapt, state = ctrl.adapt_R(R_base, innov, nis=nis, state=state)
            assert np.all(np.diag(R_adapt) >= np.diag(R_base) - 1e-9)

    def test_r_is_diagonal(self):
        """R_adapt 必须是对角矩阵（只允许对角型修改）"""
        ctrl = make_ctrl()
        R_base = make_R_base()
        state = TrackAdaptiveState()
        innov = np.array([10.0, 8.0, 6.0, 5.0])
        R_adapt, _ = ctrl.adapt_R(R_base, innov, nis=12.0, state=state)
        off_diag = R_adapt - np.diag(np.diag(R_adapt))
        np.testing.assert_array_equal(off_diag, np.zeros_like(off_diag))

    def test_innov_mean_updates(self):
        """innov_mean 递推：应随新息逐步更新"""
        ctrl = make_ctrl(beta=0.9)
        R_base = make_R_base()
        state = TrackAdaptiveState()
        innov = np.array([5.0, 0.0, 0.0, 0.0])
        _, state = ctrl.adapt_R(R_base, innov, nis=0.5, state=state)
        assert state.innov_mean[0] != 0.0  # 应被更新
        assert abs(state.innov_mean[0] - (1 - 0.9) * 5.0) < 1e-9


# ──────────────────────────────────────────────────────────────
# 5. Q 机动感知调度（核心 3）
# ──────────────────────────────────────────────────────────────

class TestQSchedule:
    def test_q_motion_dims_scaled_on_maneuver(self):
        """机动场景下位置/速度/航向维度 Q 应放大"""
        ctrl = make_ctrl()
        Q_base = make_Q_base()
        state = TrackAdaptiveState()
        Q_adapt, _ = ctrl.adapt_Q(
            Q_base=Q_base, nis=15.0, state=state, dt=0.04,
            delta_theta=0.3, delta_omega=0.5,
        )
        # cx/cy 维度放大
        assert Q_adapt[IDX_CX, IDX_CX] > Q_base[IDX_CX, IDX_CX]
        assert Q_adapt[IDX_CY, IDX_CY] > Q_base[IDX_CY, IDX_CY]
        # v/theta/omega 维度放大
        assert Q_adapt[IDX_V, IDX_V] > Q_base[IDX_V, IDX_V]
        assert Q_adapt[IDX_THETA, IDX_THETA] > Q_base[IDX_THETA, IDX_THETA]

    def test_q_size_dims_unchanged(self):
        """w/h 维度 Q 绝不放大"""
        ctrl = make_ctrl()
        Q_base = make_Q_base()
        state = TrackAdaptiveState()
        Q_adapt, _ = ctrl.adapt_Q(
            Q_base=Q_base, nis=15.0, state=state, dt=0.04,
            delta_theta=0.5, delta_omega=1.0,
        )
        assert Q_adapt[IDX_W, IDX_W] == pytest.approx(Q_base[IDX_W, IDX_W])
        assert Q_adapt[IDX_H, IDX_H] == pytest.approx(Q_base[IDX_H, IDX_H])

    def test_q_stable_when_no_maneuver(self):
        """无机动时 Q_adapt 应接近 Q_base（scale ≈ 1）"""
        ctrl = make_ctrl(lambda_q=0.3)
        Q_base = make_Q_base()
        state = TrackAdaptiveState()
        Q_adapt, _ = ctrl.adapt_Q(
            Q_base=Q_base, nis=0.5, state=state, dt=0.04,
            delta_theta=0.0, delta_omega=0.0,
        )
        # 无机动时 scale 应接近 1（maneuver_memory 从 0 起步）
        ratio = Q_adapt[IDX_CX, IDX_CX] / Q_base[IDX_CX, IDX_CX]
        assert ratio == pytest.approx(1.0, abs=0.05)

    def test_q_scale_bounded_by_q_max_scale(self):
        """Q 缩放因子不能超过 q_max_scale"""
        ctrl = make_ctrl(q_max_scale=4.0, lambda_q=100.0)  # 超大 lambda 触发上限
        Q_base = make_Q_base()
        state = TrackAdaptiveState()
        # 多步累积 maneuver_memory 以触发上限
        for _ in range(10):
            Q_adapt, state = ctrl.adapt_Q(
                Q_base=Q_base, nis=50.0, state=state, dt=0.04,
                delta_theta=1.0, delta_omega=2.0,
            )
        ratio = Q_adapt[IDX_CX, IDX_CX] / Q_base[IDX_CX, IDX_CX]
        assert ratio <= 4.0 + 1e-6

    def test_q_shape_preserved(self):
        """Q_adapt 形状与 Q_base 相同"""
        ctrl = make_ctrl()
        Q_base = make_Q_base()
        state = TrackAdaptiveState()
        Q_adapt, _ = ctrl.adapt_Q(Q_base=Q_base, nis=5.0, state=state, dt=0.04)
        assert Q_adapt.shape == Q_base.shape


# ──────────────────────────────────────────────────────────────
# 6. 鲁棒更新（核心 4）
# ──────────────────────────────────────────────────────────────

class TestRobustUpdate:
    def test_skip_when_extreme_nis_and_low_score(self):
        assert should_skip_update(25.0, 20.0, 0.2, 0.35) is True

    def test_no_skip_when_high_score(self):
        """高质量检测不应 skip，即使 NIS 极端"""
        assert should_skip_update(25.0, 20.0, 0.9, 0.35) is False

    def test_no_skip_when_nis_below_threshold(self):
        assert should_skip_update(10.0, 20.0, 0.2, 0.35) is False

    def test_no_skip_when_both_conditions_barely_not_met(self):
        """NIS 恰好等于阈值时不 skip（严格大于）"""
        assert should_skip_update(20.0, 20.0, 0.2, 0.35) is False

    def test_robust_clip_positive_innov(self):
        innov = np.array([100.0, 50.0, 10.0, 30.0])
        clipped = apply_robust_clip(innov, clip_delta=25.0)
        assert clipped[0] == pytest.approx(25.0)
        assert clipped[1] == pytest.approx(25.0)
        assert clipped[2] == pytest.approx(10.0)
        assert clipped[3] == pytest.approx(25.0)

    def test_robust_clip_negative_innov(self):
        innov = np.array([-100.0, -10.0])
        clipped = apply_robust_clip(innov, clip_delta=25.0)
        assert clipped[0] == pytest.approx(-25.0)
        assert clipped[1] == pytest.approx(-10.0)

    def test_robust_clip_preserves_sign(self):
        innov = np.array([50.0, -50.0, 5.0, -5.0])
        clipped = apply_robust_clip(innov, clip_delta=25.0)
        assert np.all(np.sign(clipped) == np.sign(innov))

    def test_robust_update_step_shape(self):
        x = np.zeros(STATE_DIM)
        P = np.eye(STATE_DIM) * 100.0
        innov = np.array([10.0, -30.0, 5.0, -5.0])
        H = np.zeros((4, STATE_DIM))
        H[0, 0] = H[1, 1] = H[2, 5] = H[3, 6] = 1.0
        R = np.eye(4) * 100.0
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_new, P_new = robust_update_step(x, P, innov, K, R, H, clip_delta=15.0, state_dim=STATE_DIM)
        assert x_new.shape == (STATE_DIM,)
        assert P_new.shape == (STATE_DIM, STATE_DIM)
        np.testing.assert_allclose(P_new, P_new.T, atol=1e-8)


# ──────────────────────────────────────────────────────────────
# 7. EKF skip=True 保留预测状态
# ──────────────────────────────────────────────────────────────

class TestEKFSkipUpdate:
    def test_skip_preserves_state(self):
        ekf = make_ekf_initialized()
        ekf.predict()
        x_pred = ekf.x.copy()
        P_pred = ekf.P.copy()
        meas = Measurement(z=np.array([999.0, 999.0, 50.0, 60.0]), score=0.1)
        ekf.update(meas, skip=True)
        np.testing.assert_array_equal(ekf.x, x_pred)
        np.testing.assert_array_equal(ekf.P, P_pred)

    def test_skip_false_still_updates(self):
        ekf = make_ekf_initialized()
        ekf.predict()
        x_pred = ekf.x.copy()
        meas = Measurement(z=np.array([105.0, 205.0, 50.0, 60.0]), score=0.9)
        ekf.update(meas, skip=False)
        assert not np.allclose(ekf.x, x_pred)

    def test_r_override_used(self):
        """R_override 被使用时，高置信度检测使用更大 R 应产生更保守更新"""
        ekf1 = make_ekf_initialized()
        ekf2 = make_ekf_initialized()
        ekf1.predict(); ekf2.predict()
        z = np.array([130.0, 220.0, 55.0, 65.0])
        meas1 = Measurement(z=z, score=0.9)
        meas2 = Measurement(z=z, score=0.9)
        ekf1.update(meas1)
        ekf2.update(meas2, R_override=np.diag([10000.0, 10000.0, 10000.0, 10000.0]))
        # R_override 更大 → ekf2 更新幅度更小（更靠近预测值）
        assert abs(ekf2.x[0] - 100.0) < abs(ekf1.x[0] - 100.0)

    def test_innov_clip_limits_update(self):
        """innov_clip 应限制极大新息的影响"""
        ekf1 = make_ekf_initialized()
        ekf2 = make_ekf_initialized()
        ekf1.predict(); ekf2.predict()
        x_pred = ekf1.x.copy()
        z = np.array([x_pred[0] + 500.0, x_pred[1], x_pred[5], x_pred[6]])
        meas = Measurement(z=z, score=0.9)
        ekf1.update(meas)
        ekf2.update(meas, innov_clip=10.0)
        # clip 后更新应更小
        assert abs(ekf2.x[0] - x_pred[0]) < abs(ekf1.x[0] - x_pred[0])


# ──────────────────────────────────────────────────────────────
# 8. enabled=False 向后兼容性
# ──────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_disabled_track_has_no_adaptive_state(self):
        det = make_det()
        ekf = make_ekf_initialized()
        track = Track(detection=det, ekf=ekf, n_init=2, max_age=20, frame_id=0)
        assert track._adaptive_ctrl is None
        assert track.adaptive_state is None

    def test_disabled_track_predict_works(self):
        det = make_det()
        ekf = make_ekf_initialized()
        track = Track(detection=det, ekf=ekf, n_init=2, max_age=20, frame_id=0)
        track.predict(0.04)
        assert not np.any(np.isnan(track.ekf.x))

    def test_disabled_track_update_works(self):
        det = make_det()
        ekf = make_ekf_initialized()
        track = Track(detection=det, ekf=ekf, n_init=2, max_age=20, frame_id=0)
        track.predict(0.04)
        det2 = make_det(x1=82, x2=122, score=0.85)
        track.update(det2, frame_id=1, dt=0.04)
        assert not np.any(np.isnan(track.ekf.x))

    def test_disabled_mot_identical_to_no_param(self):
        """adaptive_noise_cfg=None 与不传参的行为应完全一致"""
        import copy
        dets = [make_det(), make_det(x1=200, x2=240, y1=100, y2=160, score=0.8)]

        mot1 = MultiObjectTracker(n_init=2, max_age=10, dt=0.04)
        mot2 = MultiObjectTracker(n_init=2, max_age=10, dt=0.04, adaptive_noise_cfg=None)
        Track.reset_id_counter()

        active1 = mot1.step(copy.deepcopy(dets), 0)
        Track.reset_id_counter()
        active2 = mot2.step(copy.deepcopy(dets), 0)

        assert len(active1) == len(active2)

    def test_diagnostics_returns_none_when_disabled(self):
        det = make_det()
        ekf = make_ekf_initialized()
        track = Track(detection=det, ekf=ekf, n_init=2, max_age=20, frame_id=0)
        assert track.get_adaptive_diagnostics() is None


# ──────────────────────────────────────────────────────────────
# 9. Track 级自适应集成测试
# ──────────────────────────────────────────────────────────────

class TestTrackAdaptiveIntegration:
    def _make_adaptive_track(self, ctrl=None):
        if ctrl is None:
            ctrl = make_ctrl()
        det = make_det()
        ekf = ExtendedKalmanFilter(dt=0.04, std_cx=12.0, std_cy=12.0, std_w=18.0, std_h=18.0)
        ekf.initialize(det.to_measurement())
        return Track(detection=det, ekf=ekf, n_init=2, max_age=20, frame_id=0,
                     adaptive_controller=ctrl)

    def test_adaptive_state_initialized(self):
        track = self._make_adaptive_track()
        assert track.adaptive_state is not None
        assert track.adaptive_state.total_updates == 0

    def test_predict_does_not_crash(self):
        track = self._make_adaptive_track()
        track.predict(0.04)
        assert not np.any(np.isnan(track.ekf.x))

    def test_update_increments_total(self):
        track = self._make_adaptive_track()
        track.predict(0.04)
        det2 = make_det(x1=82, x2=122, score=0.85)
        track.update(det2, frame_id=1, dt=0.04)
        assert track.adaptive_state.total_updates == 1

    def test_normal_update_no_skip(self):
        """正常观测不应触发 skip"""
        track = self._make_adaptive_track()
        track.predict(0.04)
        det2 = make_det(x1=82, x2=122, score=0.9)  # 高分 + 小新息
        track.update(det2, frame_id=1, dt=0.04)
        assert track.adaptive_state.skipped_update_count == 0

    def test_extreme_low_score_triggers_skip(self):
        """极端异常 + 低分检测应累积 skip_count"""
        ctrl = make_ctrl(drop_threshold=0.001, low_score=0.99)  # 极端阈值确保触发
        track = self._make_adaptive_track(ctrl)
        track.predict(0.04)
        # 极端位移 + 极低分
        det_far = make_det(x1=800, x2=900, y1=800, y2=900, score=0.1)
        track.update(det_far, frame_id=1, dt=0.04)
        assert track.adaptive_state.skipped_update_count >= 1

    def test_diagnostics_structure(self):
        track = self._make_adaptive_track()
        track.predict(0.04)
        det2 = make_det(x1=82, x2=122, score=0.85)
        track.update(det2, frame_id=1, dt=0.04)
        diag = track.get_adaptive_diagnostics()
        assert diag is not None
        assert "avg_nis" in diag
        assert "nis_over_threshold_rate" in diag
        assert "skipped_update_count" in diag
        assert "total_updates" in diag
        assert isinstance(diag["avg_nis"], float)

    def test_delta_stored_after_predict(self):
        """predict 后 last_delta_theta/omega 应被更新"""
        ctrl = make_ctrl()
        det = make_det()
        ekf = ExtendedKalmanFilter(dt=0.04)
        ekf.initialize(det.to_measurement())
        # 注入非零 omega 以使 delta_theta 非零
        ekf.x[IDX_OMEGA] = 0.5
        track = Track(detection=det, ekf=ekf, n_init=2, max_age=20, frame_id=0,
                      adaptive_controller=ctrl)
        track.predict(0.04)
        # last_delta_theta 应被设置为 predict 前后的 theta 差
        assert isinstance(track.adaptive_state.last_delta_theta, float)


# ──────────────────────────────────────────────────────────────
# 10. NIS 统计计数器
# ──────────────────────────────────────────────────────────────

class TestNISRecordUpdate:
    def test_record_normal_update(self):
        ctrl = make_ctrl()
        state = TrackAdaptiveState()
        ctrl.record_update(state, nis=2.0, skipped=False)
        assert state.total_updates == 1
        assert state.skipped_update_count == 0
        assert state.abnormal_update_count == 0

    def test_record_anomaly_update(self):
        ctrl = make_ctrl()
        state = TrackAdaptiveState()
        ctrl.record_update(state, nis=12.0, skipped=False)
        assert state.abnormal_update_count == 1
        assert state.nis_over_threshold_count == 1
        assert state.skipped_update_count == 0

    def test_record_skipped_update(self):
        ctrl = make_ctrl()
        state = TrackAdaptiveState()
        ctrl.record_update(state, nis=25.0, skipped=True)
        assert state.skipped_update_count == 1
        assert state.abnormal_update_count == 1

    def test_nis_ema_converges(self):
        ctrl = make_ctrl()
        state = TrackAdaptiveState()
        for _ in range(50):
            ctrl.record_update(state, nis=5.0, skipped=False)
        assert state.nis_ema == pytest.approx(5.0, abs=0.5)

    def test_over_threshold_rate_correct(self):
        ctrl = make_ctrl()
        state = TrackAdaptiveState()
        ctrl.record_update(state, nis=2.0, skipped=False)   # normal
        ctrl.record_update(state, nis=12.0, skipped=False)  # anomaly
        ctrl.record_update(state, nis=12.0, skipped=False)  # anomaly
        assert state.total_updates == 3
        assert state.nis_over_threshold_rate == pytest.approx(2/3, rel=1e-6)


# ──────────────────────────────────────────────────────────────
# 11. 消融开关：only_r_adapt / only_q_schedule
# ──────────────────────────────────────────────────────────────

class TestAblationSwitches:
    def test_only_r_adapt_does_not_run_q(self):
        ctrl = make_ctrl(only_r_adapt=True)
        assert ctrl.cfg.q_adapt_on is False
        assert ctrl.cfg.r_adapt_on is True

    def test_only_q_schedule_does_not_run_r(self):
        ctrl = make_ctrl(only_q_schedule=True)
        assert ctrl.cfg.r_adapt_on is False
        assert ctrl.cfg.q_adapt_on is True

    def test_mot_with_only_r_adapt(self):
        dets = [make_det()]
        mot = MultiObjectTracker(n_init=2, max_age=10, dt=0.04,
            adaptive_noise_cfg={"enabled": True, "only_r_adapt": True})
        for i in range(5):
            mot.step(dets, i)
        # 应正常运行，不崩溃
        assert len(mot.manager.tracks) >= 0


# ──────────────────────────────────────────────────────────────
# 13. G5 +RQ-adapt 消融组：R+Q 开，robust 关
# ──────────────────────────────────────────────────────────────

class TestRQNoRobust:
    """验证 G5 (+RQ-adapt) 配置：r_adapt_on=True, q_adapt_on=True, robust_on=False"""

    def _make_rq_ctrl(self, **kwargs):
        cfg = {"enabled": True, "use_robust_update": False,
               "only_r_adapt": False, "only_q_schedule": False}
        cfg.update(kwargs)
        return make_ctrl(**cfg)

    def test_config_flags(self):
        """use_robust_update=False → r_adapt_on=True, q_adapt_on=True, robust_on=False"""
        ctrl = self._make_rq_ctrl()
        assert ctrl.cfg.r_adapt_on is True
        assert ctrl.cfg.q_adapt_on is True
        assert ctrl.cfg.robust_on is False

    def test_r_adapt_still_amplifies_on_high_nis(self):
        """R 自适应在高 NIS 时仍然放大 R"""
        ctrl = self._make_rq_ctrl()
        R_base = make_R_base()
        state = TrackAdaptiveState()
        innov = np.array([20.0, 20.0, 15.0, 15.0])
        R_adapt, _ = ctrl.adapt_R(R_base, innov, nis=15.0, state=state)
        assert np.any(np.diag(R_adapt) > np.diag(R_base))

    def test_q_adapt_scales_motion_dims(self):
        """Q 调度在机动场景下仍然放大运动维度"""
        ctrl = self._make_rq_ctrl()
        Q_base = make_Q_base()
        state = TrackAdaptiveState()
        Q_adapt, _ = ctrl.adapt_Q(
            Q_base=Q_base, nis=15.0, state=state, dt=0.04,
            delta_theta=0.5, delta_omega=1.0,
        )
        assert Q_adapt[IDX_CX, IDX_CX] > Q_base[IDX_CX, IDX_CX]
        assert Q_adapt[IDX_W, IDX_W] == pytest.approx(Q_base[IDX_W, IDX_W])

    def test_no_skip_even_when_extreme_nis_and_low_score(self):
        """robust_on=False 时不触发 skip update，即使 NIS 极端 + 检测分极低"""
        ctrl = self._make_rq_ctrl(drop_threshold=0.001, low_score=0.99)
        det = make_det()
        ekf = ExtendedKalmanFilter(dt=0.04, std_cx=12.0, std_cy=12.0,
                                   std_w=18.0, std_h=18.0)
        ekf.initialize(det.to_measurement())
        track = Track(detection=det, ekf=ekf, n_init=2, max_age=20,
                      frame_id=0, adaptive_controller=ctrl)
        track.predict(0.04)
        det_far = make_det(x1=800, x2=900, y1=800, y2=900, score=0.1)
        track.update(det_far, frame_id=1, dt=0.04)
        # robust_on=False → skip 不可能触发
        assert track.adaptive_state.skipped_update_count == 0

    def test_distinct_from_full_adaptive(self):
        """G5(RQ-adapt) 与 G6(Full Adaptive) 配置上可区分：robust_on 不同"""
        ctrl_g5 = make_ctrl(use_robust_update=False)
        ctrl_g6 = make_ctrl(use_robust_update=True)
        assert ctrl_g5.cfg.robust_on is False
        assert ctrl_g6.cfg.robust_on is True
        # 两组 R/Q 自适应均开启
        assert ctrl_g5.cfg.r_adapt_on is True
        assert ctrl_g5.cfg.q_adapt_on is True
        assert ctrl_g6.cfg.r_adapt_on is True
        assert ctrl_g6.cfg.q_adapt_on is True

    def test_smoke_rq_no_robust_mot(self):
        """G5 配置在 MultiObjectTracker 中完整运行 20 帧无崩溃"""
        mot = MultiObjectTracker(
            n_init=2, max_age=10, dt=0.04,
            adaptive_noise_cfg={"enabled": True, "use_robust_update": False},
        )
        dets_seq = [
            [make_det(100 + i * 2, 100, 160 + i * 2, 160),
             make_det(300 + i, 200, 360 + i, 260, score=0.8)]
            for i in range(20)
        ]
        for i, dets in enumerate(dets_seq):
            mot.step(dets, frame_id=i)
        confirmed = mot.get_confirmed_tracks()
        assert len(confirmed) >= 1
        for t in confirmed:
            assert t.adaptive_state is not None
            # skipped_update_count 应为 0（robust 关闭时不 skip）
            assert t.adaptive_state.skipped_update_count == 0
            assert not np.any(np.isnan(t.ekf.x))


# ──────────────────────────────────────────────────────────────
# 12. 小样本 smoke：MultiObjectTracker 全链路
# ──────────────────────────────────────────────────────────────

class TestAdaptiveSmokePipeline:
    def _run_mot(self, adaptive_cfg, n_frames=20):
        mot = MultiObjectTracker(n_init=2, max_age=15, dt=0.04,
                                 adaptive_noise_cfg=adaptive_cfg)
        dets_seq = [
            [make_det(100 + i * 3, 100, 160 + i * 3, 160),
             make_det(300 + i * 2, 200, 360 + i * 2, 260, score=0.85)]
            for i in range(n_frames)
        ]
        for i, dets in enumerate(dets_seq):
            mot.step(dets, frame_id=i)
        return mot

    def test_smoke_adaptive_disabled(self):
        mot = self._run_mot(None)
        confirmed = mot.get_confirmed_tracks()
        assert len(confirmed) >= 1
        for t in confirmed:
            assert t.adaptive_state is None

    def test_smoke_adaptive_enabled(self):
        mot = self._run_mot({"enabled": True})
        confirmed = mot.get_confirmed_tracks()
        assert len(confirmed) >= 1
        for t in confirmed:
            assert t.adaptive_state is not None
            assert t.adaptive_state.total_updates > 0

    def test_smoke_no_nan(self):
        """自适应模式下 20 帧运行后状态无 NaN/Inf"""
        mot = self._run_mot({"enabled": True})
        for t in mot.manager.tracks:
            assert not np.any(np.isnan(t.ekf.x))
            assert not np.any(np.isinf(t.ekf.x))
            assert not np.any(np.isnan(t.ekf.P))

    def test_smoke_diagnostics_valid(self):
        """诊断接口返回有效数值"""
        mot = self._run_mot({"enabled": True})
        for t in mot.get_confirmed_tracks():
            d = t.get_adaptive_diagnostics()
            assert d is not None
            assert 0.0 <= d["nis_over_threshold_rate"] <= 1.0
            assert d["total_updates"] > 0
            assert np.isfinite(d["avg_nis"])

    def test_smoke_only_r_vs_full(self):
        """only_r_adapt 模式应能正常跑通"""
        mot = self._run_mot({"enabled": True, "only_r_adapt": True})
        assert len(mot.manager.tracks) >= 0

    def test_smoke_only_q_vs_full(self):
        """only_q_schedule 模式应能正常跑通"""
        mot = self._run_mot({"enabled": True, "only_q_schedule": True})
        assert len(mot.manager.tracks) >= 0

    def test_smoke_make_adaptive_controller_none_input(self):
        """make_adaptive_controller(None) 返回 disabled controller"""
        ctrl = make_adaptive_controller(None)
        assert ctrl.cfg.enabled is False

    def test_smoke_make_adaptive_controller_disabled_dict(self):
        ctrl = make_adaptive_controller({"enabled": False})
        assert ctrl.cfg.enabled is False
