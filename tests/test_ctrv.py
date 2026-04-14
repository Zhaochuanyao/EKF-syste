"""
测试 CTRV 运动模型
"""

import math
import numpy as np
import pytest

from src.ekf_mot.filtering.models.ctrv import ctrv_predict, ctrv_jacobian
from src.ekf_mot.core.constants import IDX_CX, IDX_CY, IDX_THETA, IDX_OMEGA


def make_state(cx=100, cy=100, v=10, theta=0, omega=0, w=50, h=40):
    return np.array([cx, cy, v, theta, omega, w, h], dtype=np.float64)


class TestCTRVPredict:
    def test_straight_line_omega_zero(self):
        """omega=0 时应退化为直线运动"""
        x = make_state(cx=0, cy=0, v=10, theta=0, omega=0)
        dt = 1.0
        x_new = ctrv_predict(x, dt, omega_threshold=0.001)
        # 沿 x 轴方向运动
        assert abs(x_new[IDX_CX] - 10.0) < 1e-6
        assert abs(x_new[IDX_CY] - 0.0) < 1e-6

    def test_straight_line_small_omega(self):
        """omega 接近 0 时也应使用直线近似"""
        x = make_state(cx=0, cy=0, v=10, theta=0, omega=1e-5)
        dt = 1.0
        x_new = ctrv_predict(x, dt, omega_threshold=0.001)
        assert abs(x_new[IDX_CX] - 10.0) < 0.1

    def test_circular_motion(self):
        """omega != 0 时应做圆弧运动"""
        omega = math.pi / 2  # 每秒转 90 度
        x = make_state(cx=0, cy=0, v=10, theta=0, omega=omega)
        dt = 1.0
        x_new = ctrv_predict(x, dt)
        # 圆弧运动，cx 和 cy 都应有变化
        assert abs(x_new[IDX_CX]) > 0 or abs(x_new[IDX_CY]) > 0

    def test_theta_updated(self):
        """omega != 0 时 theta 应更新"""
        omega = 0.5
        x = make_state(theta=0, omega=omega)
        dt = 1.0
        x_new = ctrv_predict(x, dt)
        assert abs(x_new[IDX_THETA] - omega * dt) < 1e-6

    def test_size_unchanged(self):
        """w, h 应保持不变"""
        x = make_state(w=60, h=45)
        x_new = ctrv_predict(x, dt=0.04)
        assert x_new[5] == 60.0
        assert x_new[6] == 45.0

    def test_output_shape(self):
        x = make_state()
        x_new = ctrv_predict(x, dt=0.04)
        assert x_new.shape == (7,)


class TestCTRVJacobian:
    def test_jacobian_shape(self):
        x = make_state()
        F = ctrv_jacobian(x, dt=0.04)
        assert F.shape == (7, 7)

    def test_jacobian_diagonal_ones(self):
        """对角线上 v, omega, w, h 对自身偏导应为 1"""
        x = make_state()
        F = ctrv_jacobian(x, dt=0.04)
        assert F[2, 2] == 1.0  # v
        assert F[4, 4] == 1.0  # omega
        assert F[5, 5] == 1.0  # w
        assert F[6, 6] == 1.0  # h

    def test_numerical_jacobian(self):
        """数值验证雅可比矩阵（有限差分）"""
        x = make_state(cx=100, cy=100, v=5, theta=0.3, omega=0.2)
        dt = 0.04
        eps = 1e-5

        F_analytical = ctrv_jacobian(x, dt)
        F_numerical = np.zeros((7, 7))

        for j in range(7):
            x_plus = x.copy()
            x_plus[j] += eps
            x_minus = x.copy()
            x_minus[j] -= eps
            F_numerical[:, j] = (ctrv_predict(x_plus, dt) - ctrv_predict(x_minus, dt)) / (2 * eps)

        # 允许一定误差（数值微分精度限制）
        np.testing.assert_allclose(F_analytical, F_numerical, atol=1e-4)
