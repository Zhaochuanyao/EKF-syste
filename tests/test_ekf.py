"""
测试 EKF 滤波器
"""

import numpy as np
import pytest

from src.ekf_mot.filtering.ekf import ExtendedKalmanFilter
from src.ekf_mot.core.types import Measurement
from src.ekf_mot.core.constants import STATE_DIM, MEAS_DIM


@pytest.fixture
def ekf():
    f = ExtendedKalmanFilter(dt=0.04)
    z0 = np.array([320.0, 240.0, 80.0, 60.0])
    f.initialize(z0)
    return f


class TestEKFInit:
    def test_state_shape(self, ekf):
        assert ekf.x.shape == (STATE_DIM,)

    def test_covariance_shape(self, ekf):
        assert ekf.P.shape == (STATE_DIM, STATE_DIM)

    def test_covariance_symmetric(self, ekf):
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-10)

    def test_covariance_positive_definite(self, ekf):
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > 0)

    def test_initial_position(self, ekf):
        assert abs(ekf.x[0] - 320.0) < 1e-6
        assert abs(ekf.x[1] - 240.0) < 1e-6

    def test_initial_velocity_zero(self, ekf):
        assert abs(ekf.x[2]) < 1e-6  # v=0
        assert abs(ekf.x[4]) < 1e-6  # omega=0


class TestEKFPredict:
    def test_predict_state_shape(self, ekf):
        state = ekf.predict()
        assert state.x.shape == (STATE_DIM,)
        assert state.P.shape == (STATE_DIM, STATE_DIM)

    def test_predict_covariance_grows(self, ekf):
        """预测后协方差应增大（不确定性增加）"""
        P_before = ekf.P.copy()
        ekf.predict()
        # 迹（总不确定性）应增大
        assert np.trace(ekf.P) >= np.trace(P_before)

    def test_predict_covariance_symmetric(self, ekf):
        ekf.predict()
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-8)

    def test_predict_n_steps_no_state_change(self, ekf):
        """predict_n_steps 不应修改内部状态"""
        x_before = ekf.x.copy()
        P_before = ekf.P.copy()
        ekf.predict_n_steps(10)
        np.testing.assert_array_equal(ekf.x, x_before)
        np.testing.assert_array_equal(ekf.P, P_before)

    def test_predict_n_steps_returns_correct_count(self, ekf):
        results = ekf.predict_n_steps(5)
        assert len(results) == 5

    def test_predict_custom_dt(self, ekf):
        # 初始速度为 0 时，不同 dt 的状态均值相同，但协方差不同（Q 依赖 dt）
        state1 = ekf.predict(dt=0.04)
        ekf2 = ExtendedKalmanFilter(dt=0.04)
        ekf2.initialize(np.array([320.0, 240.0, 80.0, 60.0]))
        state2 = ekf2.predict(dt=0.08)
        # 不同 dt 应产生不同的协方差（过程噪声 Q 依赖 dt）
        assert not np.allclose(state1.P, state2.P)


class TestEKFUpdate:
    def test_update_state_shape(self, ekf):
        ekf.predict()
        meas = Measurement(z=np.array([322.0, 241.0, 81.0, 61.0]), score=0.9)
        state = ekf.update(meas)
        assert state.x.shape == (STATE_DIM,)

    def test_update_covariance_decreases(self, ekf):
        """更新后协方差应减小（信息增加）"""
        ekf.predict()
        P_after_predict = ekf.P.copy()
        meas = Measurement(z=np.array([322.0, 241.0, 81.0, 61.0]), score=0.9)
        ekf.update(meas)
        assert np.trace(ekf.P) <= np.trace(P_after_predict)

    def test_update_covariance_symmetric(self, ekf):
        ekf.predict()
        meas = Measurement(z=np.array([322.0, 241.0, 81.0, 61.0]), score=0.9)
        ekf.update(meas)
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-8)

    def test_update_moves_toward_measurement(self, ekf):
        """更新后状态应向观测值靠近"""
        ekf.predict()
        cx_before = ekf.x[0]
        z = np.array([cx_before + 20, ekf.x[1], ekf.x[5], ekf.x[6]])
        meas = Measurement(z=z, score=0.9)
        ekf.update(meas)
        # 更新后 cx 应向 z[0] 靠近
        assert ekf.x[0] > cx_before

    def test_score_adaptive_r(self, ekf):
        """低置信度应产生更大的 R（更保守的更新）"""
        ekf_high = ExtendedKalmanFilter(dt=0.04, score_adaptive=True)
        ekf_high.initialize(np.array([320.0, 240.0, 80.0, 60.0]))
        ekf_low = ExtendedKalmanFilter(dt=0.04, score_adaptive=True)
        ekf_low.initialize(np.array([320.0, 240.0, 80.0, 60.0]))

        ekf_high.predict()
        ekf_low.predict()

        z = np.array([340.0, 250.0, 85.0, 65.0])
        ekf_high.update(Measurement(z=z, score=0.95))
        ekf_low.update(Measurement(z=z, score=0.3))

        # 高置信度更新后应更靠近观测值
        assert abs(ekf_high.x[0] - 340.0) < abs(ekf_low.x[0] - 340.0)


class TestEKFNumericalStability:
    def test_long_run_stability(self):
        """长时间运行不应出现 NaN 或 Inf"""
        ekf = ExtendedKalmanFilter(dt=0.04)
        ekf.initialize(np.array([320.0, 240.0, 80.0, 60.0]))

        for i in range(100):
            ekf.predict()
            if i % 3 == 0:
                z = np.array([320.0 + i * 0.5, 240.0, 80.0, 60.0])
                ekf.update(Measurement(z=z, score=0.8))

        assert not np.any(np.isnan(ekf.x))
        assert not np.any(np.isinf(ekf.x))
        assert not np.any(np.isnan(ekf.P))
        assert not np.any(np.isinf(ekf.P))
