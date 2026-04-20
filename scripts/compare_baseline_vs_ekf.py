"""
Baseline vs EKF 轨迹质量对比脚本

在同一段视频上分别运行：
  - Baseline 跟踪器（纯 IoU 关联，无滤波，线性外推预测）
  - EKF 跟踪器（CTRV 模型 + 三阶段关联 + 多步递推预测）

主指标（论文主表）：ADE/FDE/RMSE @ 1/3/5/7/10 步
辅助指标：HeadingError / TurningADE / WinRate / Availability
历史轨迹指标（降级为辅助）：jitter / heading_change_std / long_track_ratio

用法
----
  python scripts/compare_baseline_vs_ekf.py \\
      --video assets/samples/demo.mp4 \\
      --config configs/exp/demo_vehicle_accuracy.yaml \\
      --output outputs/comparison/
"""

import sys
import json
import time
import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_baseline_vs_ekf")

FUTURE_STEPS_DEFAULT = [1, 3, 5, 7, 10]
TURNING_THRESHOLD_RAD = 0.1745  # 10°


# ══════════════════════════════════════════════════════════════
# 历史轨迹质量指标（辅助）
# ══════════════════════════════════════════════════════════════

def _jitter(history: List[Tuple[float, float]]) -> float:
    if len(history) < 2:
        return 0.0
    disps = [math.sqrt((history[i][0]-history[i-1][0])**2+(history[i][1]-history[i-1][1])**2)
             for i in range(1, len(history))]
    return float(np.std(disps))


def _heading_change_std(history: List[Tuple[float, float]]) -> float:
    if len(history) < 3:
        return 0.0

    def _wrap(a):
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    headings = [math.atan2(history[i][1]-history[i-1][1], history[i][0]-history[i-1][0])
                for i in range(1, len(history))]
    changes = [_wrap(headings[i]-headings[i-1]) for i in range(1, len(headings))]
    return float(np.std(changes))


def _velocity_variance(history: List[Tuple[float, float]]) -> float:
    if len(history) < 2:
        return 0.0
    speeds = [math.sqrt((history[i][0]-history[i-1][0])**2+(history[i][1]-history[i-1][1])**2)
              for i in range(1, len(history))]
    return float(np.var(speeds))


def _long_track_ratio(track_histories: List[List[Tuple]], threshold: int = 20) -> float:
    if not track_histories:
        return 0.0
    return round(sum(1 for h in track_histories if len(h) >= threshold) / len(track_histories), 4)


def _aggregate_track_quality(track_histories: List[List[Tuple]]) -> Dict:
    if not track_histories:
        return {"num_tracks": 0, "avg_track_length": 0.0, "long_track_ratio": 0.0,
                "avg_jitter": 0.0, "avg_heading_change_std": 0.0, "avg_velocity_variance": 0.0}
    lengths = [len(h) for h in track_histories]
    jitters  = [_jitter(h)             for h in track_histories if len(h) >= 2]
    hd_stds  = [_heading_change_std(h) for h in track_histories if len(h) >= 3]
    vel_vars = [_velocity_variance(h)  for h in track_histories if len(h) >= 2]
    return {
        "num_tracks":             len(track_histories),
        "avg_track_length":       round(float(np.mean(lengths)), 2),
        "long_track_ratio":       _long_track_ratio(track_histories, 20),
        "avg_jitter":             round(float(np.mean(jitters))  if jitters  else 0.0, 4),
        "avg_heading_change_std": round(float(np.mean(hd_stds))  if hd_stds  else 0.0, 4),
        "avg_velocity_variance":  round(float(np.mean(vel_vars)) if vel_vars else 0.0, 4),
    }


# ══════════════════════════════════════════════════════════════
# 预测指标计算
# ══════════════════════════════════════════════════════════════

def _wrap_angle(a: float) -> float:
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


def predict_baseline_future(
    history: List[Tuple[float, float]],
    steps: List[int],
) -> Dict[int, Tuple[float, float]]:
    """最近两帧恒速线性外推"""
    if len(history) < 2:
        return {}
    x0, y0 = history[-1]
    x1, y1 = history[-2]
    vx, vy = x0 - x1, y0 - y1
    return {k: (x0 + k * vx, y0 + k * vy) for k in steps}


def _is_turning(future_gt: List[Tuple[float, float]]) -> bool:
    """判断未来窗口内累计航向变化是否 > TURNING_THRESHOLD_RAD"""
    if len(future_gt) < 2:
        return False
    headings = [math.atan2(future_gt[i][1]-future_gt[i-1][1],
                           future_gt[i][0]-future_gt[i-1][0])
                for i in range(1, len(future_gt))]
    total = sum(abs(_wrap_angle(headings[i]-headings[i-1])) for i in range(1, len(headings)))
    return total > TURNING_THRESHOLD_RAD


def compute_prediction_metrics(
    anchors: List[Dict],  # 每个 anchor: {step: pred_pos, "gt": {step: gt_pos}, "heading_gt": {step: float}}
    steps: List[int],
) -> Dict:
    """
    从 prediction anchors 计算 ADE/FDE/RMSE/HeadingError/TurningADE/Availability。

    anchor 结构:
      {
        "pred": {step: (cx, cy)},
        "gt":   {step: (cx, cy)},
        "heading_gt": {step: float},   # 真实航向角（差分得到）
        "heading_pred": {step: float}, # 预测航向角（可选）
      }
    """
    pos_errors: Dict[int, List[float]] = defaultdict(list)
    heading_errors: Dict[int, List[float]] = defaultdict(list)
    turning_errors: Dict[int, List[float]] = defaultdict(list)
    valid_count: Dict[int, int] = defaultdict(int)
    eligible_count: Dict[int, int] = defaultdict(int)

    for anchor in anchors:
        pred = anchor.get("pred", {})
        gt   = anchor.get("gt", {})
        hgt  = anchor.get("heading_gt", {})
        hpred = anchor.get("heading_pred", {})

        # 判断是否转向（用最大步长的未来真实轨迹）
        max_step = max(steps)
        future_gt_seq = [gt[s] for s in range(1, max_step+1) if s in gt]
        is_turn = _is_turning(future_gt_seq)

        for k in steps:
            eligible_count[k] += 1
            if k not in pred or k not in gt:
                continue
            px, py = pred[k]
            gx, gy = gt[k]
            err = math.sqrt((px-gx)**2 + (py-gy)**2)
            pos_errors[k].append(err)
            valid_count[k] += 1

            if is_turn:
                turning_errors[k].append(err)

            # 航向误差
            if k in hgt:
                if k in hpred:
                    h_err = abs(_wrap_angle(hpred[k] - hgt[k]))
                else:
                    h_err = None
                if h_err is not None:
                    heading_errors[k].append(h_err)

    result = {}
    for k in steps:
        errs = pos_errors[k]
        n = len(errs)
        result[f"ade_{k}"]          = round(float(np.mean(errs)), 4)          if errs else None
        result[f"rmse_{k}"]         = round(float(np.sqrt(np.mean(np.array(errs)**2))), 4) if errs else None
        result[f"availability_{k}"] = round(valid_count[k] / eligible_count[k], 4) if eligible_count[k] else 0.0

        t_errs = turning_errors[k]
        result[f"turning_ade_{k}"]  = round(float(np.mean(t_errs)), 4) if t_errs else None

        h_errs = heading_errors[k]
        result[f"heading_error_{k}"] = round(float(np.mean(h_errs)), 4) if h_errs else None

    max_step = max(steps)
    fde_errs = pos_errors[max_step]
    result[f"fde_{max_step}"] = round(float(np.mean(fde_errs)), 4) if fde_errs else None

    return result


def compute_win_rate(
    ekf_anchors: List[Dict],
    base_anchors: List[Dict],
    steps: List[int],
) -> Dict[int, float]:
    """
    WinRate@k：EKF 误差 < Baseline 误差的比例（公共样本集）。

    两侧 anchors 按 (track_id, frame_id) 对齐。
    """
    # 建立 Baseline 索引
    base_idx: Dict[Tuple, Dict] = {}
    for a in base_anchors:
        key = (a["track_id"], a["frame_id"])
        base_idx[key] = a

    win_counts: Dict[int, int] = defaultdict(int)
    total_counts: Dict[int, int] = defaultdict(int)

    for ea in ekf_anchors:
        key = (ea["track_id"], ea["frame_id"])
        ba = base_idx.get(key)
        if ba is None:
            continue
        for k in steps:
            if k not in ea.get("pred", {}) or k not in ba.get("pred", {}):
                continue
            if k not in ea.get("gt", {}) or k not in ba.get("gt", {}):
                continue
            gx, gy = ea["gt"][k]
            ex, ey = ea["pred"][k]
            bx, by = ba["pred"][k]
            e_err = math.sqrt((ex-gx)**2 + (ey-gy)**2)
            b_err = math.sqrt((bx-gx)**2 + (by-gy)**2)
            total_counts[k] += 1
            if e_err < b_err:
                win_counts[k] += 1

    return {k: round(win_counts[k]/total_counts[k], 4) if total_counts[k] else 0.0
            for k in steps}


def _align_anchors_with_gt(
    anchors_raw: List[Dict],
    track_frame_pos: Dict[int, Dict[int, Tuple[float, float]]],
    steps: List[int],
) -> List[Dict]:
    """
    为每个 anchor 填充未来真实位置 gt 和真实航向 heading_gt。

    track_frame_pos[track_id][frame_id] = (cx, cy)
    """
    result = []
    for a in anchors_raw:
        tid = a["track_id"]
        fid = a["frame_id"]
        pos_map = track_frame_pos.get(tid, {})

        gt = {}
        heading_gt = {}
        for k in steps:
            if (fid + k) in pos_map:
                gt[k] = pos_map[fid + k]
        # 真实航向：差分
        for k in steps:
            if k in gt:
                prev_pos = gt.get(k-1) or pos_map.get(fid + k - 1)
                if prev_pos is None and k == 1:
                    prev_pos = pos_map.get(fid)
                if prev_pos:
                    dx = gt[k][0] - prev_pos[0]
                    dy = gt[k][1] - prev_pos[1]
                    heading_gt[k] = math.atan2(dy, dx)

        a["gt"] = gt
        a["heading_gt"] = heading_gt
        result.append(a)
    return result


# ══════════════════════════════════════════════════════════════
# Baseline 运行
# ══════════════════════════════════════════════════════════════

def run_baseline(
    detector,
    video_path: str,
    max_frames: Optional[int],
    future_steps: List[int],
    vehicle_classes: Optional[List[int]] = None,
) -> Dict:
    import cv2
    from src.ekf_mot.prediction.baseline import BaselineTracker

    tracker = BaselineTracker(
        iou_threshold=0.3,
        max_age=5,
        min_hits=2,
        future_steps=future_steps,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_id = 0
    processed = 0
    elapsed_times = []

    # track_id -> {frame_id -> (cx, cy)}
    track_frame_pos: Dict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    # 原始 anchors（未填 gt）
    anchors_raw: List[Dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if max_frames and processed >= max_frames:
            break

        t0 = time.perf_counter()
        dets = detector.predict(frame)
        if vehicle_classes:
            dets = [d for d in dets if d.class_id in vehicle_classes]
        tracks = tracker.step(dets, frame_id)
        elapsed_times.append((time.perf_counter() - t0) * 1000)

        for t in tracks:
            cx, cy = float(t.history[-1][0]), float(t.history[-1][1])
            track_frame_pos[t.track_id][frame_id] = (cx, cy)

            # 满足预测条件：当前可见 + 至少2帧历史
            if len(t.history) >= 2:
                pred = predict_baseline_future(t.history, future_steps)
                anchors_raw.append({
                    "track_id": t.track_id,
                    "frame_id": frame_id,
                    "pred": pred,
                })

        processed += 1
        if processed % 50 == 0:
            logger.info(f"  [Baseline] 帧 {frame_id}: tracks={len(tracks)}")

    cap.release()

    anchors = _align_anchors_with_gt(anchors_raw, track_frame_pos, future_steps)
    pred_metrics = compute_prediction_metrics(anchors, future_steps)

    all_tracks = tracker._tracks + tracker._removed
    histories = [t.history for t in all_tracks]
    quality = _aggregate_track_quality(histories)
    avg_ms = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    return {
        "method": "baseline_iou",
        "video": video_path,
        "num_frames": processed,
        "fps": round(fps, 1),
        "avg_inference_ms": round(avg_ms, 2),
        "prediction_metrics": pred_metrics,
        "anchors": anchors,
        **quality,
    }


# ══════════════════════════════════════════════════════════════
# EKF 运行
# ══════════════════════════════════════════════════════════════

def run_ekf(
    detector,
    video_path: str,
    config_path: Optional[str],
    max_frames: Optional[int],
    future_steps: List[int],
) -> Dict:
    import cv2
    from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
    from src.ekf_mot.prediction.trajectory_predictor import TrajectoryPredictor
    from src.ekf_mot.main import build_config
    from src.ekf_mot.core.config import Config
    from src.ekf_mot.core.constants import IDX_THETA

    cfg_dict = build_config(config_path)
    cfg = Config.from_dict(cfg_dict)
    tracker_cfg = cfg_dict.get("tracker", {})
    pred_cfg = cfg_dict.get("prediction", {})

    tracker = MultiObjectTracker.from_config(cfg)
    predictor = TrajectoryPredictor(
        future_steps=future_steps,
        dt=tracker_cfg.get("dt", 0.04),
        min_hits_for_prediction=pred_cfg.get("min_hits_for_prediction", 2),
        fixed_lag_smoothing=pred_cfg.get("fixed_lag_smoothing", False),
        smoothing_lag=pred_cfg.get("smoothing_lag", 3),
        smoothing_alpha=pred_cfg.get("smoothing_alpha", None),
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dt = 1.0 / fps
    frame_id = 0
    processed = 0
    elapsed_times = []

    track_frame_pos: Dict[int, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    anchors_raw: List[Dict] = []
    all_track_histories_raw: Dict[int, List[Tuple[float, float]]] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if max_frames and processed >= max_frames:
            break

        t0 = time.perf_counter()
        dets = detector.predict(frame)
        active_tracks = tracker.step(dets, frame_id, dt=dt)
        elapsed_times.append((time.perf_counter() - t0) * 1000)

        for t in active_tracks:
            if t.is_confirmed and t.time_since_update == 0:
                cx, cy = t.get_center()
                track_frame_pos[t.track_id][frame_id] = (cx, cy)

                if t.track_id not in all_track_histories_raw:
                    all_track_histories_raw[t.track_id] = []
                all_track_histories_raw[t.track_id].append((cx, cy))

                predictor.update_smooth(t.track_id, float(cx), float(cy))

                # 预测锚点
                if predictor.is_eligible(t):
                    pred_pos = predictor.predict_track(t, dt)
                    # 从 EKF 未来状态取预测航向
                    max_s = max(future_steps)
                    future_states = t.ekf.predict_n_steps(max_s, dt)
                    heading_pred = {}
                    for k in future_steps:
                        if k <= len(future_states):
                            heading_pred[k] = float(future_states[k-1].x[IDX_THETA])
                    anchors_raw.append({
                        "track_id": t.track_id,
                        "frame_id": frame_id,
                        "pred": pred_pos,
                        "heading_pred": heading_pred,
                    })

        processed += 1
        if processed % 50 == 0:
            logger.info(f"  [EKF] 帧 {frame_id}: tracks={len(active_tracks)}")

    cap.release()

    anchors = _align_anchors_with_gt(anchors_raw, track_frame_pos, future_steps)
    pred_metrics = compute_prediction_metrics(anchors, future_steps)

    raw_histories = list(all_track_histories_raw.values())
    quality = _aggregate_track_quality(raw_histories)
    avg_ms = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    return {
        "method": "ekf_ctrv",
        "video": video_path,
        "num_frames": processed,
        "fps": round(fps, 1),
        "avg_inference_ms": round(avg_ms, 2),
        "prediction_metrics": pred_metrics,
        "anchors": anchors,
        **quality,
    }


# ══════════════════════════════════════════════════════════════
# 对比摘要
# ══════════════════════════════════════════════════════════════

def build_compare_summary(baseline: Dict, ekf: Dict, win_rates: Dict[int, float], steps: List[int]) -> Dict:
    bp = baseline["prediction_metrics"]
    ep = ekf["prediction_metrics"]

    def _fmt(v):
        return round(v, 4) if v is not None else "N/A"

    def _imprv(bv, ev, lower_is_better=True):
        if bv is None or ev is None or bv == 0:
            return "N/A"
        pct = (bv - ev) / abs(bv) * 100 if lower_is_better else (ev - bv) / abs(bv) * 100
        return f"{pct:+.1f}%"

    # 主预测指标
    primary = {}
    for k in steps:
        primary[f"ade_{k}"]  = {"baseline": _fmt(bp.get(f"ade_{k}")),  "ekf": _fmt(ep.get(f"ade_{k}")),
                                 "improvement": _imprv(bp.get(f"ade_{k}"), ep.get(f"ade_{k}"))}
        primary[f"rmse_{k}"] = {"baseline": _fmt(bp.get(f"rmse_{k}")), "ekf": _fmt(ep.get(f"rmse_{k}")),
                                 "improvement": _imprv(bp.get(f"rmse_{k}"), ep.get(f"rmse_{k}"))}
    max_s = max(steps)
    primary[f"fde_{max_s}"] = {"baseline": _fmt(bp.get(f"fde_{max_s}")), "ekf": _fmt(ep.get(f"fde_{max_s}")),
                                "improvement": _imprv(bp.get(f"fde_{max_s}"), ep.get(f"fde_{max_s}"))}

    # EKF 优势指标
    advantage = {}
    for k in [3, 5, 10]:
        if k in steps:
            advantage[f"heading_error_{k}"]  = {"ekf": _fmt(ep.get(f"heading_error_{k}")),
                                                  "baseline": _fmt(bp.get(f"heading_error_{k}")),
                                                  "improvement": _imprv(bp.get(f"heading_error_{k}"), ep.get(f"heading_error_{k}"))}
            advantage[f"turning_ade_{k}"]    = {"ekf": _fmt(ep.get(f"turning_ade_{k}")),
                                                  "baseline": _fmt(bp.get(f"turning_ade_{k}")),
                                                  "improvement": _imprv(bp.get(f"turning_ade_{k}"), ep.get(f"turning_ade_{k}"))}
            advantage[f"win_rate_{k}"]       = {"ekf": _fmt(win_rates.get(k)), "note": "EKF误差<Baseline的比例"}
    for k in steps:
        advantage[f"availability_{k}"] = {"baseline": _fmt(bp.get(f"availability_{k}")),
                                           "ekf": _fmt(ep.get(f"availability_{k}"))}

    # 辅助历史轨迹指标
    auxiliary = {
        "avg_jitter":             {"baseline": baseline["avg_jitter"],             "ekf": ekf["avg_jitter"]},
        "avg_heading_change_std": {"baseline": baseline["avg_heading_change_std"], "ekf": ekf["avg_heading_change_std"]},
        "long_track_ratio":       {"baseline": baseline["long_track_ratio"],       "ekf": ekf["long_track_ratio"]},
        "avg_track_length":       {"baseline": baseline["avg_track_length"],       "ekf": ekf["avg_track_length"]},
        "num_tracks":             {"baseline": baseline["num_tracks"],             "ekf": ekf["num_tracks"]},
    }

    return {
        "video": baseline.get("video", ""),
        "num_frames": baseline.get("num_frames", 0),
        "primary_prediction_metrics": primary,
        "ekf_advantage_metrics": advantage,
        "auxiliary_track_quality": auxiliary,
    }


# ══════════════════════════════════════════════════════════════
# 图表
# ══════════════════════════════════════════════════════════════

def _save_comparison_charts(summary: Dict, steps: List[int], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过图表生成")
        return

    pm = summary["primary_prediction_metrics"]

    # 图1：ADE@k 折线图（EKF vs Baseline）
    ade_steps = [k for k in steps if f"ade_{k}" in pm]
    ekf_ades  = [pm[f"ade_{k}"]["ekf"]      for k in ade_steps]
    base_ades = [pm[f"ade_{k}"]["baseline"] for k in ade_steps]
    ekf_ades_v  = [v if v != "N/A" else None for v in ekf_ades]
    base_ades_v = [v if v != "N/A" else None for v in base_ades]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ade_steps, base_ades_v, marker="s", label="Baseline (CV)", color="#e07b54", linewidth=2)
    ax.plot(ade_steps, ekf_ades_v,  marker="o", label="EKF (CTRV)",    color="#2a7a2a", linewidth=2)
    ax.set_xlabel("Prediction Step (frames)")
    ax.set_ylabel("ADE (px)")
    ax.set_title("ADE vs Prediction Step — EKF vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "prediction_ade_per_step.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  图表已保存: {out_dir / 'prediction_ade_per_step.png'}")

    # 图2：WinRate@k 柱状图
    adv = summary["ekf_advantage_metrics"]
    wr_steps = [k for k in [3, 5, 10] if f"win_rate_{k}" in adv and k in steps]
    wr_vals  = [adv[f"win_rate_{k}"]["ekf"] for k in wr_steps]
    wr_vals_v = [v if v != "N/A" else 0.0 for v in wr_vals]
    if wr_steps:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.bar([f"@{k}" for k in wr_steps], wr_vals_v, color="#2a6099", width=0.4)
        ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% baseline")
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Win Rate")
        ax2.set_title("EKF Win Rate vs Baseline (per step)")
        ax2.legend()
        for bar, val in zip(bars, wr_vals_v):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        fig2.savefig(out_dir / "win_rate_per_step.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)
        logger.info(f"  图表已保存: {out_dir / 'win_rate_per_step.png'}")

    # 图3：RMSE@k 对比
    rmse_steps = [k for k in [1, 5, 10] if f"rmse_{k}" in pm and k in steps]
    if rmse_steps:
        ekf_rmse  = [pm[f"rmse_{k}"]["ekf"]      for k in rmse_steps]
        base_rmse = [pm[f"rmse_{k}"]["baseline"] for k in rmse_steps]
        ekf_rmse_v  = [v if v != "N/A" else None for v in ekf_rmse]
        base_rmse_v = [v if v != "N/A" else None for v in base_rmse]
        x = list(range(len(rmse_steps)))
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar([i-0.2 for i in x], base_rmse_v, width=0.35, label="Baseline", color="#e07b54")
        ax3.bar([i+0.2 for i in x], ekf_rmse_v,  width=0.35, label="EKF",      color="#2a7a2a")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"RMSE@{k}" for k in rmse_steps])
        ax3.set_ylabel("RMSE (px)")
        ax3.set_title("RMSE Comparison — EKF vs Baseline")
        ax3.legend()
        plt.tight_layout()
        fig3.savefig(out_dir / "rmse_comparison.png", dpi=120, bbox_inches="tight")
        plt.close(fig3)
        logger.info(f"  图表已保存: {out_dir / 'rmse_comparison.png'}")


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Baseline vs EKF 预测能力对比")
    parser.add_argument("--video",   default="assets/samples/demo.mp4")
    parser.add_argument("--config",  default="configs/exp/demo_vehicle_accuracy.yaml")
    parser.add_argument("--output",  default="outputs/comparison/")
    parser.add_argument("--weights", default="weights/yolov8n.pt")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--future-steps", nargs="+", type=int, default=FUTURE_STEPS_DEFAULT)
    parser.add_argument("--vehicle-classes", nargs="+", type=int, default=[2, 3, 5, 7])
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"视频文件不存在: {video_path}"); sys.exit(1)
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"权重文件不存在: {weights_path}"); sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = args.future_steps

    logger.info("加载检测器 (YOLOv8n)...")
    from src.ekf_mot.detection import build_detector
    detector = build_detector(backend="ultralytics", weights=str(weights_path),
                              conf=0.35, iou=0.5, imgsz=640, warmup=True)

    logger.info(f"\n{'='*50}\n运行 Baseline...\n{'='*50}")
    baseline_result = run_baseline(detector, str(video_path), args.max_frames, steps, args.vehicle_classes)

    logger.info(f"\n{'='*50}\n运行 EKF...\n{'='*50}")
    ekf_result = run_ekf(detector, str(video_path),
                         args.config if Path(args.config).exists() else None,
                         args.max_frames, steps)

    win_rates = compute_win_rate(ekf_result["anchors"], baseline_result["anchors"], steps)
    summary = build_compare_summary(baseline_result, ekf_result, win_rates, steps)

    # 保存 JSON（不含 anchors 原始数据）
    for name, res in [("baseline_result", baseline_result), ("ekf_result", ekf_result)]:
        res_save = {k: v for k, v in res.items() if k != "anchors"}
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(res_save, f, ensure_ascii=False, indent=2)
    with open(out_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 打印第一部分：预测能力主指标 ──────────────────────────
    logger.info(f"\n{'='*62}")
    logger.info("【第一部分】预测能力主指标（论文主表）")
    logger.info("这部分用于评价未来多步预测能力，是本论文主指标。")
    logger.info(f"{'指标':<20} {'Baseline':>12} {'EKF':>12} {'改善':>10}")
    logger.info("-" * 56)
    pm = summary["primary_prediction_metrics"]
    for k in steps:
        for metric in [f"ade_{k}", f"rmse_{k}"]:
            if metric in pm:
                b, e, imp = pm[metric]["baseline"], pm[metric]["ekf"], pm[metric].get("improvement","")
                logger.info(f"{metric:<20} {str(b):>12} {str(e):>12} {imp:>10}")
    max_s = max(steps)
    fde_key = f"fde_{max_s}"
    if fde_key in pm:
        b, e, imp = pm[fde_key]["baseline"], pm[fde_key]["ekf"], pm[fde_key].get("improvement","")
        logger.info(f"{fde_key:<20} {str(b):>12} {str(e):>12} {imp:>10}")

    # ── 打印第二部分：EKF 优势指标 ────────────────────────────
    logger.info(f"\n{'='*62}")
    logger.info("【第二部分】EKF 优势指标")
    logger.info("这部分用于体现 EKF 相对简单线性外推 Baseline 的结构性优势。")
    adv = summary["ekf_advantage_metrics"]
    for k in [3, 5, 10]:
        if k not in steps:
            continue
        for metric in [f"heading_error_{k}", f"turning_ade_{k}", f"win_rate_{k}", f"availability_{k}"]:
            if metric in adv:
                entry = adv[metric]
                b = entry.get("baseline", "—")
                e = entry.get("ekf", "—")
                imp = entry.get("improvement", "")
                logger.info(f"{metric:<28} baseline={b}  ekf={e}  {imp}")
    for k in [1, 7]:
        if k in steps and f"availability_{k}" in adv:
            entry = adv[f"availability_{k}"]
            logger.info(f"availability_{k:<20} baseline={entry.get('baseline','—')}  ekf={entry.get('ekf','—')}")

    # ── 打印第三部分：辅助历史轨迹指标 ───────────────────────
    logger.info(f"\n{'='*62}")
    logger.info("【第三部分】辅助历史轨迹指标")
    logger.info("这部分仅用于评价历史轨迹质量，不作为未来预测能力的主结论依据。")
    aux = summary["auxiliary_track_quality"]
    for k, v in aux.items():
        logger.info(f"  {k:<28} baseline={v['baseline']}  ekf={v['ekf']}")

    _save_comparison_charts(summary, steps, out_dir)
    logger.info(f"\n结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()
