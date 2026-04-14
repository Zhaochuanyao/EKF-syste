"""
统一评估入口 - 检测 / 跟踪 / 预测三模式

用法：

  # 模式 1：检测评估（跟踪器输出 JSON vs GT JSON，统计 TP/FP/FN）
  python scripts/evaluate.py \\
      --mode detection \\
      --pred outputs/demo/tracks_*.json \\
      --gt   data/processed/demo_gt.json \\
      --output outputs/metrics/detection_eval.json

  # 模式 2：跟踪评估（MOTA / MOTP / ID Switch）
  python scripts/evaluate.py \\
      --mode tracking \\
      --pred outputs/demo/tracks_*.json \\
      --gt   data/processed/demo_gt.json \\
      --output outputs/metrics/tracking_eval.json

  # 模式 3：预测评估（ADE / FDE / RMSE，使用跟踪器 JSON 中的预测点）
  python scripts/evaluate.py \\
      --mode prediction \\
      --pred outputs/demo/tracks_*.json \\
      --gt   data/processed/demo_gt.json \\
      --output outputs/metrics/prediction_eval.json

  # 模式 4：全量评估（以上三项同时输出）
  python scripts/evaluate.py \\
      --mode all \\
      --pred outputs/demo/tracks_*.json \\
      --gt   data/processed/demo_gt.json \\
      --output outputs/metrics/full_eval.json

输出 JSON 示例（tracking 模式）::

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


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

def load_pred_json(pred_path: str) -> List[Dict]:
    """加载跟踪器输出 JSON（逐帧 tracks 列表）"""
    # 支持通配符
    paths = glob.glob(pred_path)
    if not paths:
        paths = [pred_path]

    all_frames = []
    for p in sorted(paths):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        # 支持两种格式：直接列表 或 {frames: [...]}
        if isinstance(data, list):
            all_frames.extend(data)
        elif isinstance(data, dict) and "frames" in data:
            all_frames.extend(data["frames"])
        else:
            all_frames.append(data)
    return all_frames


def load_gt_json(gt_path: str) -> Dict[int, List[Dict]]:
    """
    加载 GT JSON（内部统一格式），返回 {frame_id: [annotation, ...]}
    """
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    gt_by_frame: Dict[int, List[Dict]] = {}
    for frame in data.get("frames", []):
        fid = int(frame["frame_id"])
        gt_by_frame[fid] = frame.get("annotations", [])
    return gt_by_frame


def extract_pred_bboxes(frame: Dict) -> List[Tuple[int, List[float]]]:
    """
    从帧数据中提取预测轨迹框（只取 Confirmed 状态）。
    返回 [(track_id, [x1,y1,x2,y2]), ...]
    """
    result = []
    for t in frame.get("tracks", []):
        if t.get("state_name", "Confirmed") != "Confirmed":
            continue
        bbox = t.get("bbox", [])
        if len(bbox) == 4:
            result.append((int(t["track_id"]), [float(v) for v in bbox]))
    return result


def extract_gt_bboxes(annotations: List[Dict]) -> List[Tuple[int, List[float]]]:
    """
    从 GT 标注中提取边界框。
    返回 [(gt_id, [x1,y1,x2,y2]), ...]
    """
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
        # 预测：用所有 track 的 bbox（不限 Confirmed）
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
) -> Dict:
    """计算 MOTA / MOTP / ID Switch"""
    if gt_by_frame is None:
        track_ids = set()
        for f in pred_frames:
            for t in f.get("tracks", []):
                track_ids.add(t.get("track_id"))
        return {
            "mode": "tracking",
            "note": "未提供 GT，仅统计轨迹数量",
            "num_frames": len(pred_frames),
            "num_tracks": len(track_ids),
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
    """
    计算预测 ADE / FDE / RMSE。

    策略：对每帧中的 Confirmed 轨迹，读取其 predicted_future_points（step→位置），
    然后在对应的未来帧 GT 中查找 IoU 最高的匹配框中心，作为真值。
    """
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

    # 建立 frame_id → pred_frame 映射，方便查询未来帧 GT
    gt_sorted = sorted(gt_by_frame.keys())

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

                # 在未来帧 GT 中找 IoU 最高的匹配框中心
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
        description="统一评估入口（检测/跟踪/预测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pred", required=True, help="跟踪器输出 JSON 路径（支持 glob 通配符）")
    parser.add_argument("--gt", default=None, help="GT 标注 JSON 路径（可选）")
    parser.add_argument(
        "--mode",
        choices=["detection", "tracking", "prediction", "all"],
        default="all",
        help="评估模式（默认: all）",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/metrics/eval.json")
    args = parser.parse_args()

    # 加载数据
    logger.info(f"加载预测结果: {args.pred}")
    pred_frames = load_pred_json(args.pred)
    logger.info(f"  共 {len(pred_frames)} 帧")

    gt_by_frame = None
    if args.gt and Path(args.gt).exists():
        logger.info(f"加载 GT: {args.gt}")
        gt_by_frame = load_gt_json(args.gt)
        logger.info(f"  GT 帧数: {len(gt_by_frame)}")
    else:
        logger.warning("未提供 GT 或文件不存在，跳过精度指标计算")

    # 执行评估
    report: Dict = {"pred": args.pred, "gt": args.gt or "none"}

    if args.mode in ("detection", "all"):
        logger.info("── 检测评估 ──")
        det_report = evaluate_detection(pred_frames, gt_by_frame, args.iou_threshold)
        if args.mode == "all":
            report["detection"] = det_report
        else:
            report.update(det_report)

    if args.mode in ("tracking", "all"):
        logger.info("── 跟踪评估 ──")
        trk_report = evaluate_tracking(pred_frames, gt_by_frame, args.iou_threshold)
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

    # 保存报告
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告已保存: {out_path}")

    # 打印关键指标
    if "tracking" in report:
        r = report["tracking"]
        logger.info(
            f"[跟踪] MOTA={r.get('MOTA', 0):.4f}  MOTP={r.get('MOTP', 0):.4f}  "
            f"ID_Switch={r.get('ID_Switch', 0)}"
        )
    if "detection" in report:
        g = report["detection"].get("global", {})
        if g:
            logger.info(
                f"[检测] P={g.get('precision', 0):.4f}  R={g.get('recall', 0):.4f}  "
                f"F1={g.get('f1', 0):.4f}  AP50={g.get('ap50', 0):.4f}"
            )
    if "prediction" in report:
        r = report["prediction"]
        logger.info(
            f"[预测] ADE={r.get('ADE', 0):.4f}  FDE={r.get('FDE', 0):.4f}  "
            f"RMSE={r.get('RMSE', 0):.4f}"
        )

    return report


if __name__ == "__main__":
    main()
