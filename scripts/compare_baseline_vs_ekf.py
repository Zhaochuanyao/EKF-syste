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


def _aggregate_track_quality(track_histories: List[List[Tuple]]) -> Dict:
    """汇总多条轨迹的质量指标"""
    import numpy as np
    if not track_histories:
        return {
            "num_tracks": 0,
            "avg_track_length": 0.0,
            "avg_jitter": 0.0,
            "avg_smoothness": 0.0,
        }
    lengths = [len(h) for h in track_histories]
    jitters = [_jitter(h) for h in track_histories if len(h) >= 2]
    smooths = [_smoothness(h) for h in track_histories if len(h) >= 3]
    return {
        "num_tracks": len(track_histories),
        "avg_track_length": round(float(np.mean(lengths)), 2),
        "avg_jitter": round(float(np.mean(jitters)) if jitters else 0.0, 4),
        "avg_smoothness": round(float(np.mean(smooths)) if smooths else 0.0, 4),
    }


# ══════════════════════════════════════════════════════════════
# Baseline 运行
# ══════════════════════════════════════════════════════════════

def run_baseline(
    detector,
    video_path: str,
    max_frames: Optional[int],
    future_steps: List[int],
) -> Dict:
    """运行 Baseline 跟踪器，返回统计结果"""
    import cv2
    from src.ekf_mot.prediction.baseline import BaselineTracker

    BaselineTracker_inst = BaselineTracker(
        iou_threshold=0.3,
        max_age=5,
        min_hits=3,
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

    tracker = MultiObjectTracker.from_config(cfg)
    predictor = TrajectoryPredictor(
        future_steps=future_steps,
        dt=tracker_cfg.get("dt", 0.04),
        min_hits_for_prediction=3,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dt = 1.0 / fps

    frame_id = 0
    processed = 0
    elapsed_times = []
    all_track_histories: Dict[int, List[Tuple[float, float]]] = {}

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

        # 收集轨迹历史（EKF 滤波后的中心点）
        for t in active_tracks:
            if t.is_confirmed:
                cx, cy = t.get_center()
                if t.track_id not in all_track_histories:
                    all_track_histories[t.track_id] = []
                all_track_histories[t.track_id].append((cx, cy))

        processed += 1

        if processed % 50 == 0:
            logger.info(f"  [EKF] 帧 {frame_id}: tracks={len(active_tracks)}, ms={elapsed:.1f}")

    cap.release()

    histories = list(all_track_histories.values())
    quality = _aggregate_track_quality(histories)
    avg_ms = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    return {
        "method": "ekf_ctrv",
        "video": video_path,
        "num_frames": processed,
        "fps": round(fps, 1),
        "avg_inference_ms": round(avg_ms, 2),
        **quality,
    }


# ══════════════════════════════════════════════════════════════
# 对比摘要
# ══════════════════════════════════════════════════════════════

def build_compare_summary(baseline: Dict, ekf: Dict) -> Dict:
    """生成对比摘要，含相对改善比例"""

    def _improvement(base_val: float, ekf_val: float, lower_is_better: bool) -> str:
        """计算改善百分比，lower_is_better=True 时 ekf_val 小意味着改善"""
        if base_val == 0:
            return "N/A"
        delta = base_val - ekf_val if lower_is_better else ekf_val - base_val
        pct = delta / abs(base_val) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    jitter_imprv = _improvement(baseline["avg_jitter"], ekf["avg_jitter"], lower_is_better=True)
    smooth_imprv = _improvement(baseline["avg_smoothness"], ekf["avg_smoothness"], lower_is_better=True)
    len_imprv = _improvement(baseline["avg_track_length"], ekf["avg_track_length"], lower_is_better=False)

    # 动态结论：根据实际数值判断各维度优劣
    def _verdict(base_val: float, ekf_val: float, lower_is_better: bool, name: str) -> str:
        if base_val == 0:
            return f"{name} 无法比较（Baseline 为 0）"
        if lower_is_better:
            return f"{name}：EKF {'更优' if ekf_val < base_val else '更差'}（{ekf_val:.3f} vs {base_val:.3f}）"
        else:
            return f"{name}：EKF {'更优' if ekf_val > base_val else '更差'}（{ekf_val:.2f} vs {base_val:.2f}）"

    ekf_tracks = ekf["num_tracks"]
    base_tracks = baseline["num_tracks"]
    track_count_note = (
        f"轨迹数量：EKF {ekf_tracks} vs Baseline {base_tracks}"
        + ("（EKF 碎片化更少）" if ekf_tracks <= base_tracks else "（EKF 碎片化更多，建议调整参数）")
    )

    conclusion_lines = [
        "EKF 系统相比 Baseline：",
        f"  {_verdict(baseline['avg_jitter'], ekf['avg_jitter'], True, '轨迹抖动')}，改善 {jitter_imprv}",
        f"  {_verdict(baseline['avg_smoothness'], ekf['avg_smoothness'], True, '轨迹平滑度')}，改善 {smooth_imprv}",
        f"  {_verdict(baseline['avg_track_length'], ekf['avg_track_length'], False, '平均轨迹长度')}，变化 {len_imprv}",
        f"  {track_count_note}",
    ]

    return {
        "video": baseline.get("video", ""),
        "num_frames": baseline.get("num_frames", 0),
        "metrics_comparison": {
            "avg_jitter": {
                "baseline": baseline["avg_jitter"],
                "ekf": ekf["avg_jitter"],
                "improvement": jitter_imprv,
                "note": "帧间位移标准差（越小越稳定）",
            },
            "avg_smoothness": {
                "baseline": baseline["avg_smoothness"],
                "ekf": ekf["avg_smoothness"],
                "improvement": smooth_imprv,
                "note": "帧间加速度均值（越小越平滑）",
            },
            "num_tracks": {
                "baseline": base_tracks,
                "ekf": ekf_tracks,
                "note": "总轨迹数量（越少碎片化越低）",
            },
            "avg_track_length": {
                "baseline": baseline["avg_track_length"],
                "ekf": ekf["avg_track_length"],
                "improvement": len_imprv,
                "note": "平均轨迹持续帧数（越长连续性越好）",
            },
        },
        "conclusion": "\n".join(conclusion_lines),
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
    baseline_result = run_baseline(
        detector=detector,
        video_path=str(video_path),
        max_frames=args.max_frames,
        future_steps=args.future_steps,
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
    logger.info(f"\n{'='*50}")
    logger.info("核心对比结果:")
    logger.info(f"{'指标':<20} {'Baseline':>12} {'EKF':>12} {'改善':>10}")
    logger.info("-" * 56)
    mc = compare_summary["metrics_comparison"]
    for key, val in mc.items():
        b = val["baseline"]
        e = val["ekf"]
        imprv = val.get("improvement", "")
        logger.info(f"{key:<20} {str(b):>12} {str(e):>12} {imprv:>10}")
    logger.info(f"\n{compare_summary['conclusion']}")


if __name__ == "__main__":
    main()
