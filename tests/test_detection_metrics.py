"""
检测评估指标测试

覆盖：
  - IoU 计算正确性
  - 单帧 TP/FP/FN 统计
  - 多帧累积后 Precision/Recall/F1
  - AP50 在典型场景下的计算
  - DetectionEvaluator 高层接口
"""

import numpy as np
import pytest
from src.ekf_mot.core.types import Detection
from src.ekf_mot.metrics.detection_metrics import (
    _compute_iou,
    _iou_matrix,
    match_detections,
    compute_ap,
    DetectionMetrics,
)
from src.ekf_mot.detection.evaluator import DetectionEvaluator


def make_det(x1, y1, x2, y2, score=0.9, class_id=0, frame_id=0):
    return Detection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        score=score,
        class_id=class_id,
        class_name="car",
        frame_id=frame_id,
    )


# ══════════════════════════════════════════════════════════════
# IoU 计算
# ══════════════════════════════════════════════════════════════

class TestComputeIoU:
    def test_perfect_overlap(self):
        """完全重叠时 IoU=1"""
        iou = _compute_iou(
            np.array([0, 0, 100, 100]),
            np.array([0, 0, 100, 100]),
        )
        assert abs(iou - 1.0) < 1e-6

    def test_no_overlap(self):
        """不相交时 IoU=0"""
        iou = _compute_iou(
            np.array([0, 0, 50, 50]),
            np.array([100, 100, 200, 200]),
        )
        assert abs(iou) < 1e-6

    def test_half_overlap(self):
        """一个框在另一个框的一半"""
        # box_a: [0,0,100,100] area=10000
        # box_b: [50,0,150,100] area=10000
        # intersection: [50,0,100,100] area=5000
        # union: 10000+10000-5000=15000
        # IoU = 5000/15000 = 1/3
        iou = _compute_iou(
            np.array([0, 0, 100, 100]),
            np.array([50, 0, 150, 100]),
        )
        assert abs(iou - 1.0 / 3.0) < 1e-4

    def test_contained_box(self):
        """小框完全在大框内"""
        # box_a: [0,0,100,100] area=10000
        # box_b: [25,25,75,75] area=2500
        # intersection=2500, union=10000
        # IoU=2500/10000=0.25
        iou = _compute_iou(
            np.array([0, 0, 100, 100]),
            np.array([25, 25, 75, 75]),
        )
        assert abs(iou - 0.25) < 1e-4

    def test_degenerate_box(self):
        """面积为零的框 IoU=0"""
        iou = _compute_iou(
            np.array([50, 50, 50, 50]),  # 零面积
            np.array([0, 0, 100, 100]),
        )
        assert iou == 0.0


# ══════════════════════════════════════════════════════════════
# 单帧匹配
# ══════════════════════════════════════════════════════════════

class TestMatchDetections:
    def test_perfect_match(self):
        """预测框与 GT 完全重叠：所有都是 TP"""
        preds = [make_det(0, 0, 100, 100, score=0.9)]
        gts = [make_det(0, 0, 100, 100)]
        tp, fp, fn, pairs = match_detections(preds, gts, iou_threshold=0.5)
        assert tp == 1
        assert fp == 0
        assert fn == 0
        assert len(pairs) == 1

    def test_no_match_low_iou(self):
        """IoU 不满足阈值：FP + FN"""
        preds = [make_det(0, 0, 50, 50, score=0.9)]
        gts = [make_det(200, 200, 300, 300)]
        tp, fp, fn, pairs = match_detections(preds, gts, iou_threshold=0.5)
        assert tp == 0
        assert fp == 1
        assert fn == 1
        assert len(pairs) == 0

    def test_extra_preds_are_fp(self):
        """多余的预测框是 FP"""
        preds = [
            make_det(0, 0, 100, 100, score=0.9),
            make_det(0, 0, 100, 100, score=0.8),  # 重复检测
        ]
        gts = [make_det(0, 0, 100, 100)]
        tp, fp, fn, pairs = match_detections(preds, gts, iou_threshold=0.5)
        assert tp == 1
        assert fp == 1
        assert fn == 0

    def test_missing_gts_are_fn(self):
        """漏检的 GT 是 FN"""
        preds = [make_det(0, 0, 100, 100, score=0.9)]
        gts = [
            make_det(0, 0, 100, 100),
            make_det(200, 200, 300, 300),  # 未检测到
        ]
        tp, fp, fn, pairs = match_detections(preds, gts, iou_threshold=0.5)
        assert tp == 1
        assert fp == 0
        assert fn == 1

    def test_empty_preds(self):
        """无预测：FN = |GT|"""
        gts = [make_det(0, 0, 100, 100), make_det(200, 200, 300, 300)]
        tp, fp, fn, pairs = match_detections([], gts, iou_threshold=0.5)
        assert tp == 0
        assert fp == 0
        assert fn == 2

    def test_empty_gts(self):
        """无 GT：FP = |preds|"""
        preds = [make_det(0, 0, 100, 100), make_det(200, 200, 300, 300)]
        tp, fp, fn, pairs = match_detections(preds, [], iou_threshold=0.5)
        assert tp == 0
        assert fp == 2
        assert fn == 0

    def test_empty_both(self):
        """均为空：全 0"""
        tp, fp, fn, pairs = match_detections([], [], iou_threshold=0.5)
        assert tp == 0 and fp == 0 and fn == 0

    def test_high_confidence_matched_first(self):
        """高置信度预测框应优先匹配（贪婪策略）"""
        # 两个重叠框，高分框应匹配 GT
        preds = [
            make_det(0, 0, 100, 100, score=0.3),
            make_det(0, 0, 100, 100, score=0.9),
        ]
        gts = [make_det(0, 0, 100, 100)]
        tp, fp, fn, pairs = match_detections(preds, gts, iou_threshold=0.5)
        assert tp == 1
        assert fp == 1
        # 高分框（索引1）应该匹配
        matched_pred_indices = [p[0] for p in pairs]
        assert 1 in matched_pred_indices  # score=0.9 的框应匹配

    def test_two_to_two_perfect(self):
        """2对2完美匹配"""
        preds = [
            make_det(0, 0, 100, 100, score=0.9),
            make_det(200, 200, 300, 300, score=0.8),
        ]
        gts = [
            make_det(0, 0, 100, 100),
            make_det(200, 200, 300, 300),
        ]
        tp, fp, fn, _ = match_detections(preds, gts, iou_threshold=0.5)
        assert tp == 2 and fp == 0 and fn == 0


# ══════════════════════════════════════════════════════════════
# DetectionMetrics 多帧累积
# ══════════════════════════════════════════════════════════════

class TestDetectionMetrics:
    def test_perfect_detection(self):
        """完美检测：Precision=Recall=F1=1"""
        m = DetectionMetrics(iou_threshold=0.5)
        for _ in range(5):
            preds = [make_det(0, 0, 100, 100, score=0.9)]
            gts = [make_det(0, 0, 100, 100)]
            m.update(preds, gts)

        r = m.compute()
        assert r["precision"] > 0.99
        assert r["recall"] > 0.99
        assert r["f1"] > 0.99

    def test_no_detection(self):
        """无检测：Precision=0, Recall=0"""
        m = DetectionMetrics(iou_threshold=0.5)
        for _ in range(3):
            gts = [make_det(0, 0, 100, 100)]
            m.update([], gts)

        r = m.compute()
        assert r["tp"] == 0
        assert r["fn"] == 3
        assert r["recall"] < 0.01

    def test_all_fp(self):
        """全误检：Precision=0"""
        m = DetectionMetrics(iou_threshold=0.5)
        for _ in range(3):
            preds = [make_det(500, 500, 600, 600, score=0.9)]
            gts = [make_det(0, 0, 100, 100)]
            m.update(preds, gts)

        r = m.compute()
        assert r["tp"] == 0
        assert r["fp"] == 3
        assert r["fn"] == 3
        assert r["precision"] < 0.01

    def test_reset(self):
        """reset 后重新统计"""
        m = DetectionMetrics()
        m.update([make_det(0, 0, 100, 100)], [make_det(0, 0, 100, 100)])
        assert m._tp == 1
        m.reset()
        assert m._tp == 0
        assert m._frame_counter == 0

    def test_partial_detection(self):
        """部分检测：Precision 和 Recall 均在合理范围内"""
        m = DetectionMetrics(iou_threshold=0.5)
        # 帧1：2个GT，检测到1个
        m.update(
            [make_det(0, 0, 100, 100, score=0.9)],
            [make_det(0, 0, 100, 100), make_det(200, 200, 300, 300)],
        )
        # 帧2：1个GT，检测到1个（正确）
        m.update(
            [make_det(0, 0, 100, 100, score=0.9)],
            [make_det(0, 0, 100, 100)],
        )
        r = m.compute()
        assert r["tp"] == 2
        assert r["fn"] == 1
        assert r["fp"] == 0
        # Precision = 2/(2+0) = 1.0
        assert r["precision"] > 0.99
        # Recall = 2/(2+1) ≈ 0.667
        assert 0.6 < r["recall"] < 0.8

    def test_summary_string(self):
        """summary() 方法返回字符串"""
        m = DetectionMetrics()
        m.update([make_det(0, 0, 100, 100)], [make_det(0, 0, 100, 100)])
        s = m.summary()
        assert "Precision=" in s
        assert "Recall=" in s
        assert "F1=" in s
        assert "AP50=" in s


# ══════════════════════════════════════════════════════════════
# AP50 计算
# ══════════════════════════════════════════════════════════════

class TestComputeAP:
    def test_perfect_ap(self):
        """完美检测器 AP 应接近 1"""
        preds = [make_det(0, 0, 100, 100, score=0.9, frame_id=0)]
        gts = [make_det(0, 0, 100, 100, score=1.0, frame_id=0)]
        ap = compute_ap(preds, gts, iou_threshold=0.5)
        assert ap > 0.9

    def test_no_pred_ap_zero(self):
        """无预测 AP=0"""
        gts = [make_det(0, 0, 100, 100, frame_id=0)]
        ap = compute_ap([], gts, iou_threshold=0.5)
        assert ap == 0.0

    def test_no_gt_ap_zero(self):
        """无 GT AP=0"""
        preds = [make_det(0, 0, 100, 100, score=0.9, frame_id=0)]
        ap = compute_ap(preds, [], iou_threshold=0.5)
        assert ap == 0.0

    def test_ap_range(self):
        """AP 值应在 [0, 1] 范围内"""
        preds = [
            make_det(0, 0, 100, 100, score=0.9, frame_id=0),
            make_det(200, 200, 300, 300, score=0.5, frame_id=0),
        ]
        gts = [make_det(0, 0, 100, 100, frame_id=0)]
        ap = compute_ap(preds, gts, iou_threshold=0.5)
        assert 0.0 <= ap <= 1.0


# ══════════════════════════════════════════════════════════════
# DetectionEvaluator 高层接口
# ══════════════════════════════════════════════════════════════

class TestDetectionEvaluator:
    def test_basic_report(self):
        """基础报告结构正确"""
        ev = DetectionEvaluator(iou_threshold=0.5)
        ev.update(
            [make_det(0, 0, 100, 100, class_id=0)],
            [make_det(0, 0, 100, 100, class_id=0)],
        )
        report = ev.compute()
        assert "global" in report
        assert "iou_threshold" in report
        assert "num_frames" in report
        g = report["global"]
        assert g["precision"] > 0.99
        assert g["recall"] > 0.99
        assert g["f1"] > 0.99

    def test_per_class_keys(self):
        """per_class 报告包含 class_name"""
        ev = DetectionEvaluator(per_class=True)
        ev.update(
            [make_det(0, 0, 100, 100, class_id=0)],
            [make_det(0, 0, 100, 100, class_id=0)],
        )
        report = ev.compute()
        assert "per_class" in report
        assert "0" in report["per_class"]
        assert "class_name" in report["per_class"]["0"]

    def test_save_report(self, tmp_path):
        """save_report 输出 JSON 文件"""
        ev = DetectionEvaluator()
        ev.update(
            [make_det(0, 0, 100, 100)],
            [make_det(0, 0, 100, 100)],
        )
        out = tmp_path / "metrics.json"
        ev.save_report(str(out))
        assert out.exists()

        import json
        with open(out) as f:
            data = json.load(f)
        assert "global" in data

    def test_reset(self):
        """reset 后评估器恢复初始状态"""
        ev = DetectionEvaluator()
        ev.update([make_det(0, 0, 100, 100)], [make_det(0, 0, 100, 100)])
        ev.reset()
        r = ev.compute()
        assert r["global"]["tp"] == 0
        assert r["num_frames"] == 0
