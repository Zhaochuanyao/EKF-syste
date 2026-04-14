"""
跟踪指标测试 — TrackingEvaluator (MOTA/MOTP/ID-Switch)
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.metrics.tracking_metrics import (
    TrackingEvaluator,
    TrackingMetrics,
    _compute_iou,
)


class TestComputeIou:
    def test_perfect_overlap(self):
        box = [0.0, 0.0, 10.0, 10.0]
        assert _compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _compute_iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_half_overlap(self):
        # 两个 10×10 框，水平偏移 5px → 交集 5×10=50，并集 150
        iou = _compute_iou([0, 0, 10, 10], [5, 0, 15, 10])
        assert iou == pytest.approx(50.0 / 150.0, abs=1e-4)

    def test_contained_box(self):
        # 内框 IoU = inner_area / outer_area
        iou = _compute_iou([2, 2, 8, 8], [0, 0, 10, 10])
        assert iou == pytest.approx(36.0 / 100.0, abs=1e-4)


class TestTrackingEvaluator:
    """TrackingEvaluator 单元测试"""

    def _make_box(self, x1, y1, x2, y2):
        return [float(x1), float(y1), float(x2), float(y2)]

    def test_perfect_match_single_frame(self):
        """预测框和 GT 完全重合 → TP=1, FP=0, FN=0, IDSW=0"""
        ev = TrackingEvaluator(iou_threshold=0.5)
        pred = [(1, self._make_box(0, 0, 10, 10))]
        gt = [(100, self._make_box(0, 0, 10, 10))]
        stats = ev.update(pred, gt)
        assert stats["tp"] == 1
        assert stats["fp"] == 0
        assert stats["fn"] == 0
        assert stats["id_switches"] == 0

    def test_no_pred_all_fn(self):
        """无预测框 → 全 FN"""
        ev = TrackingEvaluator()
        stats = ev.update([], [(1, self._make_box(0, 0, 10, 10))])
        assert stats["fn"] == 1
        assert stats["tp"] == 0

    def test_no_gt_all_fp(self):
        """无 GT → 全 FP"""
        ev = TrackingEvaluator()
        stats = ev.update([(1, self._make_box(0, 0, 10, 10))], [])
        assert stats["fp"] == 1
        assert stats["tp"] == 0

    def test_iou_below_threshold(self):
        """IoU 低于阈值 → FP + FN"""
        ev = TrackingEvaluator(iou_threshold=0.5)
        # 仅有 5px 重叠，IoU << 0.5
        pred = [(1, self._make_box(0, 0, 10, 10))]
        gt = [(100, self._make_box(9, 9, 19, 19))]
        stats = ev.update(pred, gt)
        assert stats["tp"] == 0
        assert stats["fp"] == 1
        assert stats["fn"] == 1

    def test_id_switch_detected(self):
        """同一 GT 目标匹配到不同 track_id → ID Switch"""
        ev = TrackingEvaluator(iou_threshold=0.5)
        box = self._make_box(0, 0, 10, 10)
        # 帧1：gt_id=100 与 track_id=1 匹配
        ev.update([(1, box)], [(100, box)])
        # 帧2：gt_id=100 与 track_id=2 匹配（ID Switch）
        ev.update([(2, box)], [(100, box)])
        report = ev.compute()
        assert report["ID_Switch"] == 1

    def test_no_id_switch_same_assignment(self):
        """同一 GT 始终匹配同一 track_id → 无 ID Switch"""
        ev = TrackingEvaluator(iou_threshold=0.5)
        box = self._make_box(0, 0, 10, 10)
        for _ in range(5):
            ev.update([(1, box)], [(100, box)])
        report = ev.compute()
        assert report["ID_Switch"] == 0

    def test_mota_formula(self):
        """MOTA = 1 - (FN + FP + IDSW) / GT"""
        ev = TrackingEvaluator(iou_threshold=0.5)
        box = self._make_box(0, 0, 10, 10)
        # 帧1: perfect match
        ev.update([(1, box)], [(100, box)])
        # 帧2: FN (no pred)
        ev.update([], [(100, box)])
        report = ev.compute()
        # GT=2, FN=1, FP=0, IDSW=0 → MOTA = 1 - 1/2 = 0.5
        assert report["MOTA"] == pytest.approx(0.5, abs=1e-4)

    def test_motp_is_iou_mean(self):
        """MOTP = 匹配对 IoU 均值"""
        ev = TrackingEvaluator(iou_threshold=0.5)
        box = self._make_box(0, 0, 10, 10)
        ev.update([(1, box)], [(100, box)])  # perfect match IoU=1.0
        report = ev.compute()
        assert report["MOTP"] == pytest.approx(1.0, abs=1e-4)

    def test_track_length_accumulation(self):
        """track_lengths 正确累积"""
        ev = TrackingEvaluator(iou_threshold=0.5)
        box = self._make_box(0, 0, 10, 10)
        for _ in range(3):
            ev.update([(1, box)], [(100, box)])
        report = ev.compute()
        assert report["avg_track_length"] == pytest.approx(3.0, abs=1e-4)

    def test_reset_clears_state(self):
        """reset() 后所有累积清零"""
        ev = TrackingEvaluator()
        box = self._make_box(0, 0, 10, 10)
        ev.update([(1, box)], [(100, box)])
        ev.reset()
        report = ev.compute()
        assert report["TP"] == 0
        assert report["num_frames"] == 0


class TestTrackingMetrics:
    """TrackingMetrics 轻量级累积器测试"""

    def test_basic_compute(self):
        m = TrackingMetrics()
        m.update(tp=10, fp=2, fn=3, id_switches=1, dist_sum=8.0, matched=10)
        r = m.compute()
        # GT = 10+3 = 13
        # MOTA = 1 - (3+2+1)/13 = 1 - 6/13
        assert r["MOTA"] == pytest.approx(1.0 - 6.0 / 13.0, abs=1e-4)
        assert r["MOTP"] == pytest.approx(8.0 / 10.0, abs=1e-4)
        assert r["TP"] == 10
        assert r["FP"] == 2
        assert r["FN"] == 3
        assert r["ID_Switch"] == 1

    def test_reset(self):
        m = TrackingMetrics()
        m.update(tp=5, fp=1, fn=1)
        m.reset()
        r = m.compute()
        assert r["TP"] == 0
