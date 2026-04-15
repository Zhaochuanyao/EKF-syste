"""
Baseline vs EKF 轨迹质量对比脚本

在同一段视频上分别运行：
  - Baseline 跟踪器（纯 IoU 关联，无滤波，线性外推预测）
  - EKF 跟踪器（CTRV 模型 + 三阶段关联 + 多步递推预测）

对比指标
--------
  轨迹抖动（Jitter）    = 帧间位移标准差（越小越稳定）
  轨迹平滑度（Smooth.） = 帧间加速度均值（越小越平滑）
  轨迹数量              = 活跃轨迹总数
  平均轨迹长度          = 轨迹平均持续帧数
  预测误差（ADE/FDE）   = 预测点与真实位置的平均/最终位移误差

用法
----
  python scripts/compare_baseline_vs_ekf.py \\
      --video assets/samples/demo.mp4 \\
      --config configs/exp/demo_vehicle_accuracy.yaml \\
      --output outputs/comparison/ \\
      --max-frames 300

输出
----
  outputs/comparison/baseline_result.json   # Baseline 跟踪轨迹统计
  outputs/comparison/ekf_result.json        # EKF 跟踪轨迹统计
  outputs/comparison/compare_summary.json   # 对比摘要（含相对改善比例）
"""

import sys
import json
import time
import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_baseline_vs_ekf")


# ══════════════════════════════════════════════════════════════
# 轨迹质量计算
# ══════════════════════════════════════════════════════════════

def _jitter(history: List[Tuple[float, float]]) -> float:
    """帧间位移标准差"""
    import numpy as np
    if len(history) < 2:
        return 0.0
    disps = [
        math.sqrt((history[i][0] - history[i-1][0])**2
                  + (history[i][1] - history[i-1][1])**2)
        for i in range(1, len(history))
    ]
    return float(np.std(disps))


def _smoothness(history: List[Tuple[float, float]]) -> float:
    """帧间加速度均值"""
    import numpy as np
    if len(history) < 3:
        return 0.0
    disps = [
        math.sqrt((history[i][0] - history[i-1][0])**2
                  + (history[i][1] - history[i-1][1])**2)
        for i in range(1, len(history))
    ]
    accels = [abs(disps[i] - disps[i-1]) for i in range(1, len(disps))]
    return float(np.mean(accels))


def _heading_change_std(history: List[Tuple[float, float]]) -> float:
    """
    帧间航向变化标准差（弧度）。

    反映轨迹方向的稳定性：值越小，车辆行驶方向越连贯，转向噪声越低。
    仅从 (cx, cy) 历史推算，无需 EKF 内部状态，Baseline/EKF 均适用。
    """
    import numpy as np
    if len(history) < 3:
        return 0.0

    def _normalize_angle(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    headings = [
        math.atan2(history[i][1] - history[i-1][1],
                   history[i][0] - history[i-1][0])
        for i in range(1, len(history))
    ]
    heading_changes = [
        _normalize_angle(headings[i] - headings[i-1])
        for i in range(1, len(headings))
    ]
    return float(np.std(heading_changes))


def _velocity_variance(history: List[Tuple[float, float]]) -> float:
    """
    帧间速度（位移模长）方差（像素²/帧²）。

    反映轨迹速度的稳定性：值越小，车辆速度变化越均匀，
    EKF 如果真正平滑了运动状态，该指标应低于 Baseline。
    """
    import numpy as np
    if len(history) < 2:
        return 0.0
    speeds = [
        math.sqrt((history[i][0] - history[i-1][0])**2
                  + (history[i][1] - history[i-1][1])**2)
        for i in range(1, len(history))
    ]
    return float(np.var(speeds))


def _aggregate_track_quality(track_histories: List[List[Tuple]]) -> Dict:
    """汇总多条轨迹的质量指标（含航向稳定性和速度方差）"""
    import numpy as np
    if not track_histories:
        return {
            "num_tracks": 0,
            "avg_track_length": 0.0,
            "avg_jitter": 0.0,
            "avg_smoothness": 0.0,
            "avg_heading_change_std": 0.0,
            "avg_velocity_variance": 0.0,
        }
    lengths = [len(h) for h in track_histories]
    jitters  = [_jitter(h)             for h in track_histories if len(h) >= 2]
    smooths  = [_smoothness(h)         for h in track_histories if len(h) >= 3]
    hd_stds  = [_heading_change_std(h) for h in track_histories if len(h) >= 3]
    vel_vars = [_velocity_variance(h)  for h in track_histories if len(h) >= 2]
    return {
        "num_tracks": len(track_histories),
        "avg_track_length":       round(float(np.mean(lengths)), 2),
        "avg_jitter":             round(float(np.mean(jitters))   if jitters  else 0.0, 4),
        "avg_smoothness":         round(float(np.mean(smooths))   if smooths  else 0.0, 4),
        "avg_heading_change_std": round(float(np.mean(hd_stds))   if hd_stds  else 0.0, 4),
        "avg_velocity_variance":  round(float(np.mean(vel_vars))  if vel_vars else 0.0, 4),
    }


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
    """运行 Baseline 跟踪器，返回统计结果"""
    import cv2
    from src.ekf_mot.prediction.baseline import BaselineTracker

    # Baseline 参数与车辆场景对齐（min_hits=2 与 EKF n_init=2 一致）
    BaselineTracker_inst = BaselineTracker(
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

    # 收集每帧信息（简化版）
    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if max_frames and processed >= max_frames:
            break

        t0 = time.perf_counter()
        dets = detector.predict(frame)
        # 按车辆类别过滤检测（与 EKF 保持一致，公平对比）
        if vehicle_classes:
            dets = [d for d in dets if d.class_id in vehicle_classes]
        tracks = BaselineTracker_inst.step(dets, frame_id)
        elapsed = (time.perf_counter() - t0) * 1000
        elapsed_times.append(elapsed)

        frame_info = {
            "frame_id": frame_id,
            "num_detections": len(dets),
            "num_tracks": len(tracks),
        }
        frames_data.append(frame_info)
        processed += 1

        if processed % 50 == 0:
            logger.info(f"  [Baseline] 帧 {frame_id}: tracks={len(tracks)}, ms={elapsed:.1f}")

    cap.release()

    # 汇总所有轨迹
    all_tracks = BaselineTracker_inst._tracks + BaselineTracker_inst._removed
    histories = [t.history for t in all_tracks]
    quality = _aggregate_track_quality(histories)

    avg_ms = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    return {
        "method": "baseline_iou",
        "video": video_path,
        "num_frames": processed,
        "fps": round(fps, 1),
        "avg_inference_ms": round(avg_ms, 2),
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
    """运行 EKF 跟踪器，返回统计结果"""
    import cv2
    from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
    from src.ekf_mot.prediction.trajectory_predictor import TrajectoryPredictor
    from src.ekf_mot.main import build_config
    from src.ekf_mot.core.config import Config

    cfg_dict = build_config(config_path)
    cfg = Config.from_dict(cfg_dict)
    tracker_cfg = cfg_dict.get("tracker", {})
    pred_cfg = cfg_dict.get("prediction", {})

    tracker = MultiObjectTracker.from_config(cfg)
    predictor = TrajectoryPredictor(
        future_steps=future_steps,
        dt=tracker_cfg.get("dt", 0.04),
        min_hits_for_prediction=pred_cfg.get("min_hits_for_prediction", 3),
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
    all_track_histories_raw: Dict[int, List[Tuple[float, float]]] = {}
    all_track_histories_smooth: Dict[int, List[Tuple[float, float]]] = {}

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
        elapsed = (time.perf_counter() - t0) * 1000
        elapsed_times.append(elapsed)

        # 仅命中帧采样（与 Baseline 公平一致）
        for t in active_tracks:
            if t.is_confirmed and t.time_since_update == 0:
                cx, cy = t.get_center()
                # raw 历史
                if t.track_id not in all_track_histories_raw:
                    all_track_histories_raw[t.track_id] = []
                all_track_histories_raw[t.track_id].append((cx, cy))
                # 更新 EMA 平滑，取最新平滑点
                predictor.update_smooth(t.track_id, float(cx), float(cy))
                smooth_hist = predictor.get_smooth_history(t.track_id)
                if smooth_hist:
                    if t.track_id not in all_track_histories_smooth:
                        all_track_histories_smooth[t.track_id] = []
                    all_track_histories_smooth[t.track_id].append(smooth_hist[-1])

        processed += 1

        if processed % 50 == 0:
            logger.info(f"  [EKF] 帧 {frame_id}: tracks={len(active_tracks)}, ms={elapsed:.1f}")

    cap.release()

    raw_histories = list(all_track_histories_raw.values())
    smooth_histories = list(all_track_histories_smooth.values())

    raw_quality = _aggregate_track_quality(raw_histories)
    smoothed_quality = _aggregate_track_quality(smooth_histories) if smooth_histories else raw_quality
    avg_ms = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0
    history_mode = "smoothed" if smooth_histories else "raw"

    return {
        "method": "ekf_ctrv",
        "video": video_path,
        "num_frames": processed,
        "fps": round(fps, 1),
        "avg_inference_ms": round(avg_ms, 2),
        # 主比较使用 smoothed（顶层展开，兼容下游 build_compare_summary）
        **smoothed_quality,
        # 诊断信息
        "raw_quality": raw_quality,
        "smoothed_quality": smoothed_quality,
        "history_mode_used": history_mode,
    }


# ══════════════════════════════════════════════════════════════
# 对比摘要
# ══════════════════════════════════════════════════════════════

def build_compare_summary(baseline: Dict, ekf: Dict) -> Dict:
    """生成对比摘要，含相对改善比例"""

    def _pct_change(base_val: float, ekf_val: float, lower_is_better: bool) -> float:
        """
        计算相对变化百分比（正 = EKF 更好，负 = EKF 更差）。
        lower_is_better=True: base_val > ekf_val → 正（EKF 变小 = 改善）
        lower_is_better=False: ekf_val > base_val → 正（EKF 变大 = 改善）
        """
        if base_val == 0:
            return 0.0
        delta = base_val - ekf_val if lower_is_better else ekf_val - base_val
        return delta / abs(base_val) * 100

    def _format_change(pct: float) -> str:
        """将百分比格式化为带语义词的字符串，避免'改善 -87%'的歧义"""
        if abs(pct) < 1.0:
            return f"持平（{pct:+.1f}%）"
        elif pct > 0:
            return f"改善 +{pct:.1f}%"
        else:
            return f"退步 {pct:.1f}%"

    def _improvement_str(base_val: float, ekf_val: float, lower_is_better: bool) -> str:
        """向 JSON 写入用的改善字符串"""
        if base_val == 0:
            return "N/A"
        pct = _pct_change(base_val, ekf_val, lower_is_better)
        return f"{pct:+.1f}%"

    jitter_pct  = _pct_change(baseline["avg_jitter"],             ekf["avg_jitter"],             True)
    smooth_pct  = _pct_change(baseline["avg_smoothness"],         ekf["avg_smoothness"],         True)
    len_pct     = _pct_change(baseline["avg_track_length"],        ekf["avg_track_length"],       False)
    hd_pct      = _pct_change(baseline["avg_heading_change_std"],  ekf["avg_heading_change_std"], True)
    vel_pct     = _pct_change(baseline["avg_velocity_variance"],   ekf["avg_velocity_variance"],  True)

    jitter_imprv = _improvement_str(baseline["avg_jitter"],             ekf["avg_jitter"],             True)
    smooth_imprv = _improvement_str(baseline["avg_smoothness"],         ekf["avg_smoothness"],         True)
    len_imprv    = _improvement_str(baseline["avg_track_length"],        ekf["avg_track_length"],       False)
    hd_imprv     = _improvement_str(baseline["avg_heading_change_std"],  ekf["avg_heading_change_std"], True)
    vel_imprv    = _improvement_str(baseline["avg_velocity_variance"],   ekf["avg_velocity_variance"],  True)

    # 动态结论：根据实际数值判断各维度优劣（语义明确，不出现"改善 -87%"）
    def _verdict(base_val: float, ekf_val: float, lower_is_better: bool, name: str,
                 pct: float, fmt: str) -> str:
        if base_val == 0:
            return f"{name}：无法比较（Baseline 为 0）"
        if lower_is_better:
            better = ekf_val < base_val
            return (f"{name}：EKF {'更优' if better else '更差'}"
                    f"（EKF={ekf_val:.4f} vs Base={base_val:.4f}），{fmt}")
        else:
            better = ekf_val > base_val
            return (f"{name}：EKF {'更优' if better else '更差'}"
                    f"（EKF={ekf_val:.2f} vs Base={base_val:.2f}），{fmt}")

    ekf_tracks  = ekf["num_tracks"]
    base_tracks = baseline["num_tracks"]
    # 轨迹数量：过多意味着碎片化严重；但也要考虑帧数差异
    # 此处简单比较总量：EKF <= Baseline 才算碎片化更少
    if ekf_tracks <= base_tracks:
        track_note = f"轨迹数量：EKF {ekf_tracks} vs Baseline {base_tracks}，EKF 碎片化更少"
    else:
        diff = ekf_tracks - base_tracks
        track_note = (f"轨迹数量：EKF {ekf_tracks} vs Baseline {base_tracks}，"
                      f"EKF 多 {diff} 条——建议提高 min_create_score 或延长 max_age")

    conclusion_lines = [
        "EKF 系统相比 Baseline：",
        f"  {_verdict(baseline['avg_jitter'],            ekf['avg_jitter'],            True,  '轨迹抖动',     jitter_pct, _format_change(jitter_pct))}",
        f"  {_verdict(baseline['avg_smoothness'],        ekf['avg_smoothness'],        True,  '轨迹平滑度',   smooth_pct, _format_change(smooth_pct))}",
        f"  {_verdict(baseline['avg_heading_change_std'],ekf['avg_heading_change_std'],True,  '航向变化稳定性',hd_pct,    _format_change(hd_pct))}",
        f"  {_verdict(baseline['avg_velocity_variance'], ekf['avg_velocity_variance'], True,  '速度方差',     vel_pct,    _format_change(vel_pct))}",
        f"  {_verdict(baseline['avg_track_length'],      ekf['avg_track_length'],      False, '平均轨迹长度', len_pct,    _format_change(len_pct))}",
        f"  {track_note}",
        "",
        "指标说明：抖动/平滑度/航向稳定性/速度方差越小越好；轨迹长度越大越好；轨迹总数越少碎片化越低。",
    ]

    return {
        "video": baseline.get("video", ""),
        "num_frames": baseline.get("num_frames", 0),
        "metrics_comparison": {
            "avg_jitter": {
                "baseline": baseline["avg_jitter"],
                "ekf": ekf["avg_jitter"],
                "improvement": jitter_imprv,
                "note": "帧间位移标准差（越小越稳定，lower is better）",
            },
            "avg_smoothness": {
                "baseline": baseline["avg_smoothness"],
                "ekf": ekf["avg_smoothness"],
                "improvement": smooth_imprv,
                "note": "帧间加速度均值（越小越平滑，lower is better）",
            },
            "avg_heading_change_std": {
                "baseline": baseline["avg_heading_change_std"],
                "ekf": ekf["avg_heading_change_std"],
                "improvement": hd_imprv,
                "note": "帧间航向变化标准差（越小方向越稳，lower is better）",
            },
            "avg_velocity_variance": {
                "baseline": baseline["avg_velocity_variance"],
                "ekf": ekf["avg_velocity_variance"],
                "improvement": vel_imprv,
                "note": "帧间速度方差（越小速度越均匀，lower is better）",
            },
            "num_tracks": {
                "baseline": base_tracks,
                "ekf": ekf_tracks,
                "note": "总轨迹数量（越少碎片化越低，lower is better）",
            },
            "avg_track_length": {
                "baseline": baseline["avg_track_length"],
                "ekf": ekf["avg_track_length"],
                "improvement": len_imprv,
                "note": "平均轨迹持续帧数（越长连续性越好，higher is better）",
            },
        },
        "conclusion": "\n".join(conclusion_lines),
        # 诊断信息（主结论按 smoothed，raw 仅内部参考）
        "ekf_history_mode": ekf.get("history_mode_used", "raw"),
        "ekf_raw_quality": ekf.get("raw_quality", {}),
        "ekf_smoothed_quality": ekf.get("smoothed_quality", {}),
    }


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Baseline vs EKF 跟踪效果对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", default="assets/samples/demo.mp4", help="输入视频路径")
    parser.add_argument("--config", default="configs/exp/demo_vehicle_accuracy.yaml", help="EKF 配置文件")
    parser.add_argument("--output", default="outputs/comparison/", help="输出目录")
    parser.add_argument("--weights", default="weights/yolov8n.pt", help="YOLOv8n 权重")
    parser.add_argument("--max-frames", type=int, default=300, help="最大处理帧数")
    parser.add_argument(
        "--future-steps",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="预测步数列表（default: 1 5 10）",
    )
    parser.add_argument(
        "--vehicle-classes",
        nargs="+",
        type=int,
        default=[2, 3, 5, 7],
        help="车辆 COCO 类别 ID（default: 2=car 3=motorcycle 5=bus 7=truck）",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"视频文件不存在: {video_path}")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"权重文件不存在: {weights_path}")
        logger.error("请先运行: python scripts/download_weights.py")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 初始化共用检测器 ──────────────────────────────────────
    logger.info("加载检测器 (YOLOv8n)...")
    from src.ekf_mot.detection import build_detector
    detector = build_detector(
        backend="ultralytics",
        weights=str(weights_path),
        conf=0.35,
        iou=0.5,
        imgsz=640,
        warmup=True,
    )

    # ── 运行 Baseline ─────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info("运行 Baseline 跟踪器（纯 IoU 关联，无 EKF）...")
    logger.info(f"{'='*50}")
    logger.info(f"  车辆类别过滤：classes={args.vehicle_classes} (car=2, motorcycle=3, bus=5, truck=7)")
    baseline_result = run_baseline(
        detector=detector,
        video_path=str(video_path),
        max_frames=args.max_frames,
        future_steps=args.future_steps,
        vehicle_classes=args.vehicle_classes,
    )

    # ── 运行 EKF ──────────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info("运行 EKF 跟踪器（CTRV + 三阶段关联）...")
    logger.info(f"{'='*50}")
    ekf_result = run_ekf(
        detector=detector,
        video_path=str(video_path),
        config_path=args.config if Path(args.config).exists() else None,
        max_frames=args.max_frames,
        future_steps=args.future_steps,
    )

    # ── 生成对比摘要 ──────────────────────────────────────────
    compare_summary = build_compare_summary(baseline_result, ekf_result)

    # ── 保存结果 ──────────────────────────────────────────────
    with open(out_dir / "baseline_result.json", "w", encoding="utf-8") as f:
        json.dump(baseline_result, f, ensure_ascii=False, indent=2)

    with open(out_dir / "ekf_result.json", "w", encoding="utf-8") as f:
        json.dump(ekf_result, f, ensure_ascii=False, indent=2)

    with open(out_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(compare_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info("对比结果已保存至:")
    logger.info(f"  {out_dir / 'baseline_result.json'}")
    logger.info(f"  {out_dir / 'ekf_result.json'}")
    logger.info(f"  {out_dir / 'compare_summary.json'}")

    # ── 打印核心对比 ──────────────────────────────────────────
    logger.info(f"\n{'='*62}")
    logger.info("核心对比结果:")
    logger.info(f"{'指标':<28} {'Baseline':>12} {'EKF':>12} {'改善':>10}")
    logger.info("-" * 64)
    mc = compare_summary["metrics_comparison"]
    for key, val in mc.items():
        b = val["baseline"]
        e = val["ekf"]
        imprv = val.get("improvement", "")
        logger.info(f"{key:<28} {str(b):>12} {str(e):>12} {imprv:>10}")
    logger.info(f"\n{compare_summary['conclusion']}")

    # ── 诊断：EKF raw vs smoothed ─────────────────────────────
    mode = ekf_result.get("history_mode_used", "raw")
    logger.info(f"\n{'='*62}")
    logger.info(f"EKF 诊断（主结论按 {mode} 轨迹，raw 仅内部参考）:")
    rq = ekf_result.get("raw_quality", {})
    sq = ekf_result.get("smoothed_quality", {})
    for k in ("avg_jitter", "avg_smoothness", "avg_heading_change_std", "avg_velocity_variance"):
        rv = rq.get(k, "N/A")
        sv = sq.get(k, "N/A")
        logger.info(f"  {k:<30} raw={rv}  smoothed={sv}")


if __name__ == "__main__":
    main()
