"""
检测器验证脚本 - 对视频或标注数据集运行 YOLOv8n 并输出检测评估报告

默认使用车辆稳定模式配置（demo_vehicle_accuracy），仅检测
COCO 类别 2/3/5/7（轿车/摩托车/巴士/卡车）。

用法：

  # 车辆模式（默认，推荐）
  python scripts/validate_detector.py \\
      --vehicle \\
      --video assets/samples/demo.mp4 \\
      --output outputs/detection/

  # 通用模式（全类别，不过滤）
  python scripts/validate_detector.py \\
      --video assets/samples/demo.mp4 \\
      --output outputs/detection/

  # 指定类别 ID
  python scripts/validate_detector.py \\
      --classes 2 3 5 7 \\
      --video assets/samples/demo.mp4

  # GT 精度评估（车辆）
  python scripts/validate_detector.py \\
      --vehicle \\
      --gt-json data/processed/ua_detrac/MVI_20011.json \\
      --video assets/samples/demo.mp4 \\
      --output outputs/detection/

输出文件：
  outputs/detection/detection_metrics.json   # 结构化评估报告
  outputs/detection/detection_results.json   # 逐帧检测结果（可选）
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_detector")

# ── 车辆场景常量 ──────────────────────────────────────────────
_VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car=2, motorcycle=3, bus=5, truck=7
_VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_VEHICLE_CONFIG = "configs/exp/demo_vehicle_accuracy.yaml"


# ══════════════════════════════════════════════════════════════
# GT 数据加载（从内部 JSON 格式）
# ══════════════════════════════════════════════════════════════

def load_gt_from_json(json_path: str, vehicle_classes: Optional[List[int]] = None):
    """
    从内部统一 JSON 格式加载 ground truth。
    vehicle_classes: 若指定，则只保留该类别的标注（None=全类别）

    Returns:
        {frame_id: [Detection, ...]}
    """
    from src.ekf_mot.core.types import Detection
    import numpy as np

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    gt_by_frame = {}
    for frame in data.get("frames", []):
        fid = frame["frame_id"]
        gt_by_frame[fid] = []
        for ann in frame.get("annotations", []):
            cid = ann.get("class_id", 0)
            if vehicle_classes and cid not in vehicle_classes:
                continue
            bbox = np.array(ann["bbox"], dtype=np.float64)
            gt_by_frame[fid].append(
                Detection(
                    bbox=bbox,
                    score=1.0,
                    class_id=cid,
                    class_name=ann.get("class_name", ""),
                    frame_id=fid,
                )
            )
    return gt_by_frame


# ══════════════════════════════════════════════════════════════
# 检测器初始化
# ══════════════════════════════════════════════════════════════

def build_detector(
    config_path: Optional[str] = None,
    weights: str = "weights/yolov8n.pt",
    classes_override: Optional[List[int]] = None,
):
    """
    构建检测器实例。
    classes_override: 若提供，覆盖配置文件中的 classes 设置。
    """
    try:
        if config_path and Path(config_path).exists():
            from src.ekf_mot.core.config import load_config
            cfg = load_config(config_path)
            det_cfg = cfg.get("detector", {})
            logger.info(f"使用配置文件: {config_path}")
        else:
            det_cfg = {
                "backend": "ultralytics",
                "weights": weights,
                "conf": 0.35,
                "iou": 0.5,
                "imgsz": 640,
                "classes": None,
            }

        # classes_override 优先于配置文件中的 classes
        effective_classes = classes_override if classes_override is not None else det_cfg.get("classes", None)

        backend = det_cfg.get("backend", "ultralytics")
        if backend == "ultralytics":
            from src.ekf_mot.detection.yolo_ultralytics import UltralyticsDetector
            detector = UltralyticsDetector(
                weights=det_cfg.get("weights", weights),
                conf=det_cfg.get("conf", 0.35),
                iou=det_cfg.get("iou", 0.5),
                imgsz=det_cfg.get("imgsz", 640),
                classes=effective_classes,
                device="cpu",
            )
        else:
            from src.ekf_mot.detection.yolo_onnx import OnnxDetector
            detector = OnnxDetector(
                weights=det_cfg.get("weights", weights),
                conf=det_cfg.get("conf", 0.35),
                imgsz=det_cfg.get("imgsz", 640),
            )

        classes_info = f"classes={effective_classes}" if effective_classes else "classes=ALL"
        logger.info(f"检测器已加载: backend={backend}, {classes_info}")
        return detector

    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        raise


# ══════════════════════════════════════════════════════════════
# 视频模式：只统计检测结果，无 GT 评估
# ══════════════════════════════════════════════════════════════

def run_detection_only(
    detector,
    video_path: str,
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
    vehicle_mode: bool = False,
) -> dict:
    """
    对视频运行检测，返回统计信息（无 GT，无精度指标）。
    vehicle_mode: 若 True，额外输出各车辆类别的检测数分布。
    """
    import cv2
    from collections import defaultdict

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    total_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"视频: {video_path} | FPS={total_fps:.1f} | 总帧数={total_frames_in_video}")

    frame_results = []
    total_dets = 0
    elapsed_times = []
    frame_id = 0
    processed = 0
    class_det_counts: dict = defaultdict(int)  # class_name → total count

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if (frame_id - 1) % frame_skip != 0:
            continue
        if max_frames and processed >= max_frames:
            break

        t0 = time.perf_counter()
        dets = detector.predict(frame)
        elapsed = (time.perf_counter() - t0) * 1000  # ms

        elapsed_times.append(elapsed)
        total_dets += len(dets)
        processed += 1

        for d in dets:
            class_det_counts[d.class_name] += 1

        frame_results.append({
            "frame_id": frame_id,
            "num_detections": len(dets),
            "elapsed_ms": round(elapsed, 2),
        })

        if processed % 50 == 0:
            logger.info(f"  已处理 {processed} 帧, 当前帧检测数={len(dets)}, 耗时={elapsed:.1f}ms")

    cap.release()

    avg_ms = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    report = {
        "mode": "detection_only_vehicle" if vehicle_mode else "detection_only",
        "video": video_path,
        "total_frames_processed": processed,
        "total_detections": total_dets,
        "avg_detections_per_frame": round(total_dets / processed, 2) if processed else 0,
        "avg_inference_ms": round(avg_ms, 2),
        "avg_fps": round(avg_fps, 2),
        "class_breakdown": dict(class_det_counts),
        "frame_results": frame_results,
    }
    if vehicle_mode:
        report["vehicle_classes_filter"] = _VEHICLE_CLASSES

    return report


# ══════════════════════════════════════════════════════════════
# 评估模式：对照 GT JSON 计算精度指标
# ══════════════════════════════════════════════════════════════

def run_evaluation(
    detector,
    gt_json_path: str,
    video_path: Optional[str],
    output_dir: Path,
    iou_threshold: float = 0.5,
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
    vehicle_classes: Optional[List[int]] = None,
) -> dict:
    """
    对照 GT JSON 运行检测评估。
    vehicle_classes: 若指定，GT 中只保留这些类别的标注（公平对比）。
    """
    import cv2
    from src.ekf_mot.detection.evaluator import DetectionEvaluator

    logger.info(f"加载 GT: {gt_json_path}")
    gt_by_frame = load_gt_from_json(gt_json_path, vehicle_classes=vehicle_classes)
    logger.info(f"  共 {len(gt_by_frame)} 帧 GT 数据")
    if vehicle_classes:
        logger.info(f"  GT 类别过滤: {vehicle_classes} ({list(_VEHICLE_CLASS_NAMES.values())})")

    evaluator = DetectionEvaluator(
        iou_threshold=iou_threshold,
        match_class=False,
        per_class=True,
    )

    if video_path is None:
        logger.warning("未提供视频路径，无法运行检测器，输出 GT 统计（仅供参考）")
        total_gt = sum(len(v) for v in gt_by_frame.values())
        return {
            "mode": "gt_stats_only",
            "gt_json": gt_json_path,
            "total_frames": len(gt_by_frame),
            "total_gt_annotations": total_gt,
            "vehicle_classes": vehicle_classes,
            "note": "未提供视频，无法计算检测器精度指标",
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    logger.info(f"视频: {video_path}")

    frame_id = 0
    processed = 0
    elapsed_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if (frame_id - 1) % frame_skip != 0:
            continue
        if max_frames and processed >= max_frames:
            break

        t0 = time.perf_counter()
        preds = detector.predict(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        elapsed_times.append(elapsed)

        for p in preds:
            p.frame_id = frame_id

        gts = gt_by_frame.get(frame_id, [])
        for g in gts:
            g.frame_id = frame_id

        evaluator.update(preds, gts)
        processed += 1

        if processed % 50 == 0:
            logger.info(f"  帧 {frame_id}: preds={len(preds)}, gts={len(gts)}, ms={elapsed:.1f}")

    cap.release()

    evaluator.print_summary()

    report = evaluator.compute()
    avg_ms = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    report["avg_inference_ms"] = round(avg_ms, 2)
    report["avg_fps"] = round(1000.0 / avg_ms if avg_ms > 0 else 0, 2)
    report["video"] = video_path
    report["gt_json"] = gt_json_path
    report["vehicle_classes"] = vehicle_classes
    report["mode"] = "evaluation_vehicle" if vehicle_classes else "evaluation"

    return report


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="检测器验证脚本（车辆优先 / 精度评估 / 纯检测统计）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", help="输入视频路径（.mp4 / .avi 等）")
    parser.add_argument("--gt-json", help="GT 标注 JSON 路径（内部统一格式）")
    parser.add_argument(
        "--config",
        default=None,
        help="实验配置文件路径（YAML）；--vehicle 时自动使用车辆配置",
    )
    parser.add_argument("--weights", default="weights/yolov8n.pt", help="检测器权重路径")
    parser.add_argument("--output", default="outputs/detection/", help="输出目录")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU 匹配阈值")
    parser.add_argument("--frame-skip", type=int, default=1, help="跳帧间隔")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")
    parser.add_argument("--save-frame-results", action="store_true", help="保存逐帧结果 JSON")

    # ── 车辆模式 ──
    vgroup = parser.add_argument_group("车辆场景选项")
    vgroup.add_argument(
        "--vehicle",
        action="store_true",
        help=(
            "车辆模式：自动设置 classes=[2,3,5,7]，"
            f"并使用车辆配置 {_VEHICLE_CONFIG}（若存在）"
        ),
    )
    vgroup.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        metavar="CID",
        help="手动指定检测类别 ID（车辆: --classes 2 3 5 7；默认: None=全类别）",
    )

    args = parser.parse_args()

    if not args.video and not args.gt_json:
        parser.error("至少需要提供 --video 或 --gt-json 之一")

    # ── 车辆模式处理 ──────────────────────────────────────────
    vehicle_mode = args.vehicle
    classes_override: Optional[List[int]] = None

    if vehicle_mode:
        # --vehicle: 强制使用车辆类别 + 车辆配置（若用户没手动指定 config）
        classes_override = _VEHICLE_CLASSES
        if args.config is None:
            args.config = _VEHICLE_CONFIG
        logger.info("═" * 50)
        logger.info("车辆检测验证模式")
        logger.info(f"  类别: car(2) / motorcycle(3) / bus(5) / truck(7)")
        logger.info(f"  配置: {args.config}")
        logger.info("═" * 50)
    elif args.classes:
        classes_override = args.classes
        logger.info(f"类别过滤: {classes_override}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 检查权重 ──────────────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"权重文件不存在: {weights_path}")
        logger.error("请先运行: python scripts/download_weights.py")
        sys.exit(1)

    # ── 初始化检测器 ──────────────────────────────────────────
    try:
        detector = build_detector(
            config_path=args.config,
            weights=args.weights,
            classes_override=classes_override,
        )
    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        sys.exit(1)

    # ── 运行评估 ──────────────────────────────────────────────
    try:
        if args.gt_json:
            report = run_evaluation(
                detector=detector,
                gt_json_path=args.gt_json,
                video_path=args.video,
                output_dir=out_dir,
                iou_threshold=args.iou_threshold,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
                vehicle_classes=classes_override,
            )
        else:
            report = run_detection_only(
                detector=detector,
                video_path=args.video,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
                vehicle_mode=vehicle_mode,
            )
    except Exception as e:
        logger.error(f"评估过程失败: {e}", exc_info=True)
        sys.exit(1)

    # ── 保存报告 ──────────────────────────────────────────────
    report_to_save = (
        report if args.save_frame_results
        else {k: v for k, v in report.items() if k != "frame_results"}
    )

    metrics_path = out_dir / "detection_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report_to_save, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告已保存: {metrics_path}")

    # ── 打印关键指标 ──────────────────────────────────────────
    if "global" in report:
        g = report["global"]
        logger.info(
            f"全局指标: P={g['precision']:.4f}  R={g['recall']:.4f}  "
            f"F1={g['f1']:.4f}  AP50={g['ap50']:.4f}"
        )
    elif "total_detections" in report:
        logger.info(
            f"检测统计: {report['total_frames_processed']} 帧 | "
            f"总检测 {report['total_detections']} | "
            f"帧均 {report['avg_detections_per_frame']} | "
            f"{report['avg_inference_ms']} ms | "
            f"{report['avg_fps']} FPS"
        )
        if report.get("class_breakdown"):
            logger.info(f"类别分布: {report['class_breakdown']}")


if __name__ == "__main__":
    main()
