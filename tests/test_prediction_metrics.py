"""
预测指标测试 — PredictionMetrics (ADE / FDE / RMSE) + compute_ade_fde()
"""

import sys
import math
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.metrics.prediction_metrics import PredictionMetrics, compute_ade_fde


class TestComputeAdeFde:
    """独立函数 compute_ade_fde 测试"""

    def test_perfect_prediction(self):
        pred = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        gt = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        ade, fde = compute_ade_fde(pred, gt)
        assert ade == pytest.approx(0.0)
        assert fde == pytest.approx(0.0)

    def test_constant_error(self):
        # 每步误差均为 sqrt(2)
        pred = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        gt = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]
        ade, fde = compute_ade_fde(pred, gt)
        assert ade == pytest.approx(math.sqrt(2), abs=1e-4)
        assert fde == pytest.approx(math.sqrt(2), abs=1e-4)

    def test_increasing_error(self):
        pred = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        gt = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        ade, fde = compute_ade_fde(pred, gt)
        assert ade == pytest.approx(2.0, abs=1e-4)   # (1+2+3)/3
        assert fde == pytest.approx(3.0, abs=1e-4)    # last step

    def test_empty_input(self):
        ade, fde = compute_ade_fde([], [])
        assert ade == 0.0
        assert fde == 0.0

    def test_single_step(self):
        pred = [(3.0, 4.0)]
        gt = [(0.0, 0.0)]
        ade, fde = compute_ade_fde(pred, gt)
        assert ade == pytest.approx(5.0, abs=1e-4)
        assert fde == pytest.approx(5.0, abs=1e-4)


class TestPredictionMetrics:
    """PredictionMetrics 累积器测试"""

    def test_perfect_predictions_all_zero(self):
        m = PredictionMetrics()
        pred = {1: (0.0, 0.0), 5: (0.0, 0.0)}
        gt = {1: (0.0, 0.0), 5: (0.0, 0.0)}
        m.update(pred, gt)
        r = m.compute()
        assert r["ADE"] == pytest.approx(0.0)
        assert r["FDE"] == pytest.approx(0.0)
        assert r["RMSE"] == pytest.approx(0.0)
        assert r["num_samples"] == 1

    def test_ade_fde_basic(self):
        m = PredictionMetrics()
        # step=1: error=3, step=5: error=4 → ADE=(3+4)/2=3.5, FDE=4
        pred = {1: (3.0, 0.0), 5: (4.0, 0.0)}
        gt = {1: (0.0, 0.0), 5: (0.0, 0.0)}
        m.update(pred, gt)
        r = m.compute()
        assert r["ADE"] == pytest.approx(3.5, abs=1e-4)
        assert r["FDE"] == pytest.approx(4.0, abs=1e-4)

    def test_rmse_formula(self):
        m = PredictionMetrics()
        # errors: [3, 4] → RMSE = sqrt((9+16)/2) = sqrt(12.5)
        pred = {1: (3.0, 0.0), 5: (4.0, 0.0)}
        gt = {1: (0.0, 0.0), 5: (0.0, 0.0)}
        m.update(pred, gt)
        r = m.compute()
        assert r["RMSE"] == pytest.approx(math.sqrt(12.5), abs=1e-4)

    def test_per_step_ade(self):
        m = PredictionMetrics()
        pred1 = {1: (1.0, 0.0), 5: (2.0, 0.0)}
        gt1 = {1: (0.0, 0.0), 5: (0.0, 0.0)}
        m.update(pred1, gt1)
        pred2 = {1: (3.0, 0.0), 5: (4.0, 0.0)}
        gt2 = {1: (0.0, 0.0), 5: (0.0, 0.0)}
        m.update(pred2, gt2)
        r = m.compute()
        # step 1: (1+3)/2=2.0, step 5: (2+4)/2=3.0
        assert r["per_step_ADE"]["1"] == pytest.approx(2.0, abs=1e-4)
        assert r["per_step_ADE"]["5"] == pytest.approx(3.0, abs=1e-4)

    def test_empty_returns_zeros(self):
        m = PredictionMetrics()
        r = m.compute()
        assert r["ADE"] == 0.0
        assert r["num_samples"] == 0

    def test_no_matching_steps(self):
        m = PredictionMetrics()
        # pred has step=1, gt has step=5 → no match → no update
        m.update({1: (1.0, 0.0)}, {5: (0.0, 0.0)})
        assert m.compute()["num_samples"] == 0

    def test_reset_clears_state(self):
        m = PredictionMetrics()
        m.update({1: (1.0, 0.0)}, {1: (0.0, 0.0)})
        m.reset()
        r = m.compute()
        assert r["num_samples"] == 0
        assert r["ADE"] == 0.0

    def test_multiple_samples_average(self):
        m = PredictionMetrics()
        # Sample 1: err=0, Sample 2: err=4
        m.update({1: (0.0, 0.0)}, {1: (0.0, 0.0)})
        m.update({1: (4.0, 0.0)}, {1: (0.0, 0.0)})
        r = m.compute()
        assert r["ADE"] == pytest.approx(2.0, abs=1e-4)
        assert r["num_samples"] == 2
