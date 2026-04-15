"""
统一评估入口 - 检测 / 跟踪 / 预测三模式（车辆优先）

支持两种工作流：
  A. 离线评估（先 run_tracking，再评估）
     python scripts/evaluate.py --pred outputs/demo/tracks.json [--gt data/gt.json]

  B. 联机评估（直接从视频跑 EKF 再输出统计，无需 GT）
     python scripts/evaluate.py --video assets/samples/demo.mp4 [--config ...]

典型用法：

  # 车辆联机运行统计（无 GT）
  python scripts/evaluate.py \\
      --vehicle \\
      --video assets/samples/demo.mp4 \\
      --output outputs/metrics/vehicle_stats.json

  # 从预计算轨迹 JSON 做车辆跟踪评估（有 GT）
  python scripts/evaluate.py \\
      --vehicle \\
      --mode tracking \\
      --pred outputs/demo/tracks_*.json \\
      --gt   data/processed/demo_gt.json \\
      --output outputs/metrics/tracking_eval.json

  # 全量评估（检测 + 跟踪 + 预测）
  python scripts/evaluate.py \\
      --mode all \\
      --pred outputs/demo/tracks_*.json \\
      --gt   data/processed/demo_gt.json

输出 JSON 示例（tracking 模式）:

    {
      "mode": "tracking",
      "MOTA": 0.7234,
      "MOTP": 0.6891,
      "TP": 1200, "FP": 180, "FN": 300, "ID_Switch": 12,
      "num_tracks": 8, "avg_track_length": 150.2
    }

说明：
  --pred 可以是单个 JSON 文件，也可以使用通配符（glob 展开）。
  GT JSON 格式与 scripts/generate_demo_gt.py 或 scripts/convert_annotations.py 产生的格式一致。
  若不提供 --gt，则只输出检测/跟踪基础统计（无精度指标）。
"""

import sys
import json
import argparse
import glob
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
logger = logging.getLogger("evaluate")

# ── 车辆场景常量 ──────────────────────────────────────────────
_VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car=2, motorcycle=3, bus=5, truck=7
_VEHICLE_CONFIG = "configs/exp/demo_vehicle_accuracy.yaml"


# ══════════════════════════════════════════════════════════════
# 联机跟踪（inline tracking）
# ══════════════════════════════════════════════════════════════

def run_tracking_inline(
    video_path: str,
    config_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
) -> Tuple[List[Dict], float]:
    """
    直接在内存中跑 EKF 跟踪，返回 (pred_frames, fps)。
    pred_frames 格式与离线 JSON 兼容，可直接传入 evaluate_* 函数。
    """
    import cv2
    from src.ekf_mot.detection import build_detector as _build_det
    from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
    from src.ekf_mot.core.config import get_default_config, Config

    if config_path and Path(config_path).exists():
        from src.ekf_mot.core.config import load_config
        cfg_dict = load_config(config_path)
        logger.info(f"联机跟踪配置: {config_path}")
    else:
        if config_path:
            logger.warning(f"配置文件不存在 ({config_path})，使用内置默认配置")
        cfg_dict = get_default_config()

    cfg = Config.from_dict(cfg_dict)
    det_cfg = cfg_dict.get("detector", {})

    detector = _build_det(
        backend=det_cfg.get("backend", "ultralytics"),
        weights=det_cfg.get("weights", "weights/yolov8n.pt"),
        conf=det_cfg.get("conf", 0.35),
        iou=det_cfg.get("iou", 0.5),
        imgsz=det_cfg.get("imgsz", 640),
        classes=det_cfg.get("classes", None),
        warmup=True,
    )

    tracker = MultiObjectTracker.from_config(cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        f"联机跟踪: {video_path} | FPS={fps:.1f} | 总帧数={total_in_video}"
    )

    pred_frames: List[Dict] = []
    frame_id = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if (frame_id - 1) % frame_skip != 0:
            continue
        if max_frames and processed >= max_frames:
            break

        dets = detector.predict(frame)
        active_tracks = tracker.step(dets, frame_id)

        pred_frames.append({
            "frame_id": frame_id,
            "num_detections": len(dets),
            "tracks": [
                {
                    "track_id": t.track_id,
                    "bbox": [float(v) for v in t.get_bbox()],
                    "score": float(t.score),
                    "class_id": t.class_id,
                    "class_name": t.class_name,
                    "state_name": t.state.name,
                }
                for t in active_tracks
            ],
        })
        processed += 1

        if processed % 50 == 0:
            logger.info(
                f"  [联机] 帧 {frame_id}: tracks={len(active_tracks)}, dets={len(dets)}"
            )

    cap.release()
    logger.info(f"联机跟踪完成: {processed} 帧")
    return pred_frames, fps


# ══════════════════════════════════════════════════════════════
# 无 GT 轨迹质量统计（与 compare_baseline_vs_ekf 同一套指标）
# ══════════════════════════════════════════════════════════════

def _compute_track_stats_no_gt(
    pred_frames: List[Dict],
    vehicle_classes: Optional[List[int]] = None,
) -> Dict:
    """
    从 pred_frames 中计算轨迹质量指标（无需 GT）。
    与 compare_baseline_vs_ekf.py 使用同一套指标，方便横向对比。
    """
    import numpy as np
    from collections import defaultdict

    track_histories: Dict[int, List[Tuple[float, float]]] = {}
    track_class: Dict[int, str] = {}
    class_counts: Dict[str, int] = defaultdict(int)

    for frame in pred_frames:
        for t in frame.get("tracks", []):
            if t.get("state_name", "") != "Confirmed":
                continue
            if vehicle_classes and t.get("class_id") not in vehicle_classes:
                continue
            tid = t["track_id"]
            bbox = t.get("bbox", [])
            if len(bbox) != 4:
                continue
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            if tid not in track_histories:
                track_histories[tid] = []
                track_class[tid] = t.get("class_name", "unknown")
            track_histories[tid].append((cx, cy))

    if not track_histories:
        return {
            "num_confirmed_tracks": 0,
            "avg_track_length": 0.0,
            "avg_jitter": 0.0,
            "avg_smoothness": 0.0,
            "class_breakdown": {},
            "note": "无 Confirmed 轨迹",
        }

    def _jitter(h: List[Tuple[float, float]]) -> float:
        if len(h) < 2:
            return 0.0
        disps = [
            math.sqrt((h[i][0] - h[i-1][0])**2 + (h[i][1] - h[i-1][1])**2)
            for i in range(1, len(h))
        ]
        return float(np.std(disps))

    def _smoothness(h: List[Tuple[float, float]]) -> float:
        if len(h) < 3:
            return 0.0
        disps = [
            math.sqrt((h[i][0] - h[i-1][0])**2 + (h[i][1] - h[i-1][1])**2)
            for i in range(1, len(h))
        ]
        accels = [abs(disps[i] - disps[i-1]) for i in range(1, len(disps))]
        return float(np.mean(accels))

    lengths = [len(h) for h in track_histories.values()]
    jitters = [_jitter(h) for h in track_histories.values() if len(h) >= 2]
    smooths = [_smoothness(h) for h in track_histories.values() if len(h) >= 3]

    for tid, cls in track_class.items():
        class_counts[cls] += 1

    return {
        "num_confirmed_tracks": len(track_histories),
        "avg_track_length": round(float(np.mean(lengths)), 2),
        "avg_jitter": round(float(np.mean(jitters)) if jitters else 0.0, 4),
        "avg_smoothness": round(float(np.mean(smooths)) if smooths else 0.0, 4),
        "class_breakdown": dict(class_counts),
        "note": (
            "无 GT，基于 Confirmed 轨迹中心点计算质量指标"
            "（与 compare_baseline_vs_ekf.py 同一套指标）"
        ),
    }


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

def load_pred_json(pred_path: str) -> List[Dict]:
    """加载跟踪器输出 JSON（逐帧 tracks 列表）"""
    paths = glob.glob(pred_path)
    if not paths:
        paths = [pred_path]

    all_frames = []
    for p in sorted(paths):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            all_frames.extend(data)
        elif isinstance(data, dict) and "frames" in data:
            all_frames.extend(data["frames"])
        else:
            all_frames.append(data)
    return all_frames


def load_gt_json(gt_path: str) -> Dict[int, List[Dict]]:
    """加载 GT JSON，返回 {frame_id: [annotation, ...]}"""
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    gt_by_frame: Dict[int, List[Dict]] = {}
    for frame in data.get("frames", []):
        fid = int(frame["frame_id"])
        gt_by_frame[fid] = frame.get("annotations", [])
    return gt_by_frame


def filter_pred_by_class(
    pred_frames: List[Dict],
    vehicle_classes: List[int],
) -> List[Dict]:
    """只保留 pred_frames 中 class_id 属于 vehicle_classes 的 tracks。"""
    filtered = []
    for frame in pred_frames:
        new_tracks = [
            t for t in frame.get("tracks", [])
            if t.get("class_id") in vehicle_classes
        ]
        filtered.append({**frame, "tracks": new_tracks})
    return filtered


def extract_pred_bboxes(frame: Dict) -> List[Tuple[int, List[float]]]:
    """提取帧中 Confirmed 轨迹的 (track_id, bbox)"""
    result = []
    for t in frame.get("tracks", []):
        if t.get("state_name", "Confirmed") != "Confirmed":
            continue
        bbox = t.get("bbox", [])
        if len(bbox) == 4:
            result.append((int(t["track_id"]), [float(v) for v in bbox]))
    return result


def extract_gt_bboxes(annotations: List[Dict]) -> List[Tuple[int, List[float]]]:
    """提取 GT 标注中的 (gt_id, bbox)"""
    result = []
    for ann in annotations:
        gt_id = int(ann.get("id", 0))
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            result.append((gt_id, [float(v) for v in bbox]))
    return result


# ══════════════════════════════════════════════════════════════
# 模式 1：检测评估
# ══════════════════════════════════════════════════════════════

def evaluate_detection(
    pred_frames: List[Dict],
    gt_by_frame: Optional[Dict[int, List[Dict]]],
    iou_threshold: float = 0.5,
) -> Dict:
    """对照 GT 计算检测 Precision / Recall / F1 / AP50"""
    if gt_by_frame is None:
        total = sum(len(f.get("tracks", [])) for f in pred_frames)
        return {
            "mode": "detection",
            "note": "未提供 GT，仅统计检测数量",
            "num_frames": len(pred_frames),
            "total_detections": total,
        }

    from src.ekf_mot.detection.evaluator import DetectionEvaluator
    from src.ekf_mot.core.types import Detection
    import numpy as np

    evaluator = DetectionEvaluator(
        iou_threshold=iou_threshold,
        match_class=False,
        per_class=True,
    )

    for frame in pred_frames:
        fid = int(frame.get("frame_id", 0))
        preds = []
        for t in frame.get("tracks", []):
            bbox = t.get("bbox", [])
            if len(bbox) == 4:
                preds.append(
                    Detection(
                        bbox=np.array(bbox, dtype=np.float64),
                        score=float(t.get("score", 1.0)),
                        class_id=int(t.get("class_id", 0)),
                        class_name=str(t.get("class_name", "")),
                        frame_id=fid,
                    )
                )
        gts = []
        for ann in gt_by_frame.get(fid, []):
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                gts.append(
                    Detection(
                        bbox=np.array(bbox, dtype=np.float64),
                        score=1.0,
                        class_id=int(ann.get("class_id", 0)),
                        class_name=str(ann.get("class_name", "")),
                        frame_id=fid,
                    )
                )
        evaluator.update(preds, gts)

    report = evaluator.compute()
    report["mode"] = "detection"
    return report


# ══════════════════════════════════════════════════════════════
# 模式 2：跟踪评估
# ══════════════════════════════════════════════════════════════

def evaluate_tracking(
    pred_frames: List[Dict],
    gt_by_frame: Optional[Dict[int, List[Dict]]],
    iou_threshold: float = 0.5,
    vehicle_classes: Optional[List[int]] = None,
) -> Dict:
    """计算 MOTA / MOTP / ID Switch；若无 GT 则输出轨迹质量统计"""
    if gt_by_frame is None:
        # 无 GT：输出轨迹质量统计（jitter / smoothness / track length）
        stats = _compute_track_stats_no_gt(pred_frames, vehicle_classes=vehicle_classes)
        return {
            "mode": "tracking_stats_no_gt",
            "num_frames": len(pred_frames),
            **stats,
        }

    from src.ekf_mot.metrics.tracking_metrics import TrackingEvaluator

    evaluator = TrackingEvaluator(iou_threshold=iou_threshold)

    for frame in pred_frames:
        fid = int(frame.get("frame_id", 0))
        pred_bboxes = extract_pred_bboxes(frame)
        gt_anns = gt_by_frame.get(fid, [])
        gt_bboxes = extract_gt_bboxes(gt_anns)
        evaluator.update(pred_bboxes, gt_bboxes)

    evaluator.print_summary()
    report = evaluator.compute()
    report["mode"] = "tracking"
    report["iou_threshold"] = iou_threshold
    return report


# ══════════════════════════════════════════════════════════════
# 模式 3：预测评估
# ══════════════════════════════════════════════════════════════

def evaluate_prediction(
    pred_frames: List[Dict],
    gt_by_frame: Optional[Dict[int, List[Dict]]],
) -> Dict:
    """计算预测 ADE / FDE / RMSE；若无 GT 则统计有预测的轨迹数"""
    if gt_by_frame is None:
        total_pred = sum(
            sum(1 for t in f.get("tracks", []) if t.get("prediction_valid", False))
            for f in pred_frames
        )
        return {
            "mode": "prediction",
            "note": "未提供 GT，仅统计有预测的轨迹数",
            "num_frames": len(pred_frames),
            "num_predicted_tracks": total_pred,
        }

    from src.ekf_mot.metrics.prediction_metrics import PredictionMetrics
    from src.ekf_mot.metrics.tracking_metrics import _compute_iou

    metrics = PredictionMetrics()

    for frame in pred_frames:
        fid = int(frame.get("frame_id", 0))
        for t in frame.get("tracks", []):
            if t.get("state_name", "") != "Confirmed":
                continue
            future_pts = t.get("predicted_future_points", {})
            if not future_pts:
                continue

            track_bbox = t.get("bbox", [])
            pred_dict: Dict[int, Tuple[float, float]] = {}
            gt_dict: Dict[int, Tuple[float, float]] = {}

            for step_str, pt in future_pts.items():
                step = int(step_str)
                future_fid = fid + step
                pred_dict[step] = (float(pt[0]), float(pt[1]))

                future_anns = gt_by_frame.get(future_fid, [])
                if not future_anns or not track_bbox:
                    continue

                best_iou = 0.0
                best_center = None
                for ann in future_anns:
                    bbox = ann.get("bbox", [])
                    if len(bbox) != 4:
                        continue
                    iou = _compute_iou(track_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        best_center = (cx, cy)

                if best_center is not None and best_iou >= 0.3:
                    gt_dict[step] = best_center

            if pred_dict and gt_dict:
                metrics.update(pred_dict, gt_dict)

    report = metrics.compute()
    report["mode"] = "prediction"
    return report


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="统一评估入口（检测/跟踪/预测，车辆优先）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 输入源（--pred 或 --video，二选一）──────────────────
    src_group = parser.add_argument_group("输入源（二选一）")
    src_group.add_argument(
        "--pred",
        default=None,
        help="跟踪器输出 JSON 路径（支持 glob 通配符）",
    )
    src_group.add_argument(
        "--video",
        default=None,
        help="直接从视频联机跑 EKF 跟踪（无需预计算 pred JSON）",
    )
    src_group.add_argument(
        "--config",
        default=_VEHICLE_CONFIG,
        help=f"联机跟踪配置文件（--video 模式使用，默认: 车辆配置）",
    )

    # ── GT ───────────────────────────────────────────────────
    parser.add_argument("--gt", default=None, help="GT 标注 JSON 路径（可选）")
    parser.add_argument(
        "--mode",
        choices=["detection", "tracking", "prediction", "all"],
        default="tracking",
        help="评估模式（默认: tracking）",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--frame-skip", type=int, default=1, help="跳帧（联机模式）")
    parser.add_argument("--max-frames", type=int, default=None, help="最大帧数（联机模式）")
    parser.add_argument("--output", default="outputs/metrics/eval.json")

    # ── 车辆模式 ──────────────────────────────────────────────
    vgroup = parser.add_argument_group("车辆场景选项")
    vgroup.add_argument(
        "--vehicle",
        action="store_true",
        help="车辆模式：pred 中只统计类别 2/3/5/7（轿车/摩托/巴士/卡车）",
    )
    vgroup.add_argument(
        "--vehicle-classes",
        nargs="+",
        type=int,
        default=_VEHICLE_CLASSES,
        metavar="CID",
        help=f"车辆类别 ID（默认: {_VEHICLE_CLASSES}）",
    )

    args = parser.parse_args()

    if not args.pred and not args.video:
        parser.error("请提供 --pred（预计算 JSON）或 --video（联机跟踪）之一")

    # ── 车辆模式 ──────────────────────────────────────────────
    vehicle_classes: Optional[List[int]] = None
    if args.vehicle:
        vehicle_classes = args.vehicle_classes
        logger.info("═" * 50)
        logger.info("车辆评估模式")
        logger.info(f"  过滤类别: {vehicle_classes} (car/motorcycle/bus/truck)")
        logger.info("═" * 50)

    # ── 获取 pred_frames ──────────────────────────────────────
    pred_frames: List[Dict] = []

    if args.video:
        logger.info(f"联机跟踪: {args.video}")
        pred_frames, _fps = run_tracking_inline(
            video_path=args.video,
            config_path=args.config,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
    else:
        logger.info(f"加载预测结果: {args.pred}")
        pred_frames = load_pred_json(args.pred)
        logger.info(f"  共 {len(pred_frames)} 帧")

    # ── 车辆类别过滤（pred 侧）───────────────────────────────
    if vehicle_classes:
        pred_frames = filter_pred_by_class(pred_frames, vehicle_classes)
        logger.info(f"已按车辆类别过滤 pred_frames: {vehicle_classes}")

    # ── GT ───────────────────────────────────────────────────
    gt_by_frame: Optional[Dict[int, List[Dict]]] = None
    if args.gt and Path(args.gt).exists():
        logger.info(f"加载 GT: {args.gt}")
        gt_by_frame = load_gt_json(args.gt)
        logger.info(f"  GT 帧数: {len(gt_by_frame)}")
    else:
        if args.gt:
            logger.warning(f"GT 文件不存在 ({args.gt})，跳过精度指标计算")
        else:
            logger.info("未提供 --gt，输出轨迹质量统计（无精度指标）")

    # ── 执行评估 ──────────────────────────────────────────────
    report: Dict = {
        "pred": args.pred or f"inline:{args.video}",
        "gt": args.gt or "none",
        "vehicle_classes": vehicle_classes,
    }

    if args.mode in ("detection", "all"):
        logger.info("── 检测评估 ──")
        det_report = evaluate_detection(pred_frames, gt_by_frame, args.iou_threshold)
        if args.mode == "all":
            report["detection"] = det_report
        else:
            report.update(det_report)

    if args.mode in ("tracking", "all"):
        logger.info("── 跟踪评估 ──")
        trk_report = evaluate_tracking(
            pred_frames, gt_by_frame, args.iou_threshold,
            vehicle_classes=vehicle_classes,
        )
        if args.mode == "all":
            report["tracking"] = trk_report
        else:
            report.update(trk_report)

    if args.mode in ("prediction", "all"):
        logger.info("── 预测评估 ──")
        pred_report = evaluate_prediction(pred_frames, gt_by_frame)
        if args.mode == "all":
            report["prediction"] = pred_report
        else:
            report.update(pred_report)

    # ── 保存报告 ──────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告已保存: {out_path}")

    # ── 打印关键指标 ──────────────────────────────────────────
    trk = report.get("tracking", report if args.mode == "tracking" else {})
    if trk.get("mode") == "tracking_stats_no_gt":
        logger.info(
            f"[轨迹统计] 确认轨迹={trk.get('num_confirmed_tracks', 0)} | "
            f"平均长度={trk.get('avg_track_length', 0)} | "
            f"抖动={trk.get('avg_jitter', 0):.4f} | "
            f"平滑度={trk.get('avg_smoothness', 0):.4f}"
        )
        if trk.get("class_breakdown"):
            logger.info(f"  类别分布: {trk['class_breakdown']}")
    elif "MOTA" in trk:
        logger.info(
            f"[跟踪] MOTA={trk.get('MOTA', 0):.4f}  MOTP={trk.get('MOTP', 0):.4f}  "
            f"ID_Switch={trk.get('ID_Switch', 0)}"
        )

    det = report.get("detection", report if args.mode == "detection" else {})
    g = det.get("global", {})
    if g:
        logger.info(
            f"[检测] P={g.get('precision', 0):.4f}  R={g.get('recall', 0):.4f}  "
            f"F1={g.get('f1', 0):.4f}  AP50={g.get('ap50', 0):.4f}"
        )

    pred_r = report.get("prediction", report if args.mode == "prediction" else {})
    if pred_r.get("ADE") is not None:
        logger.info(
            f"[预测] ADE={pred_r.get('ADE', 0):.4f}  FDE={pred_r.get('FDE', 0):.4f}  "
            f"RMSE={pred_r.get('RMSE', 0):.4f}"
        )

    return report


if __name__ == "__main__":
    main()
