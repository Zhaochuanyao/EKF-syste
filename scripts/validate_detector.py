"""
检测器验证脚本 - 对视频或标注数据集运行 YOLOv8n 并输出检测评估报告

用法：

  # 模式 1：对单个视频文件运行检测（无 ground truth，只输出检测统计）
  python scripts/validate_detector.py \\
      --video assets/samples/demo.mp4 \\
      --output outputs/detection/

  # 模式 2：对转换后的 JSON 标注数据集评估检测器精度
  python scripts/validate_detector.py \\
      --gt-json data/processed/ua_detrac/MVI_20011.json \\
      --output outputs/detection/ \\
      --iou-threshold 0.5

  # 模式 3：结合配置文件
  python scripts/validate_detector.py \\
      --config configs/exp/demo_cpu.yaml \\
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


# ══════════════════════════════════════════════════════════════
# GT 数据加载（从内部 JSON 格式）
# ══════════════════════════════════════════════════════════════

def load_gt_from_json(json_path: str):
    """
    从内部统一 JSON 格式加载 ground truth。

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
            bbox = np.array(ann["bbox"], dtype=np.float64)
            gt_by_frame[fid].append(
                Detection(
                    bbox=bbox,
                    score=1.0,
                    class_id=ann.get("class_id", 0),
                    class_name=ann.get("class_name", ""),
                    frame_id=fid,
                )
            )
    return gt_by_frame


# ══════════════════════════════════════════════════════════════
# 检测器初始化
# ══════════════════════════════════════════════════════════════

def build_detector(config_path: Optional[str] = None, weights: str = "weights/yolov8n.pt"):
    """
    构建检测器实例。

    如果有配置文件则从配置加载，否则用默认参数。
    """
    try:
        if config_path:
            from src.ekf_mot.core.config import load_config
            cfg = load_config(config_path)
            det_cfg = cfg.get("detector", {})
        else:
            det_cfg = {
                "backend": "ultralytics",
                "weights": weights,
                "conf": 0.35,
                "iou": 0.5,
                "imgsz": 640,
                "classes": None,
            }

        backend = det_cfg.get("backend", "ultralytics")
        if backend == "ultralytics":
            from src.ekf_mot.detection.yolo_ultralytics import UltralyticsDetector
            detector = UltralyticsDetector(
                weights=det_cfg.get("weights", weights),
                conf=det_cfg.get("conf", 0.35),
                iou=det_cfg.get("iou", 0.5),
                imgsz=det_cfg.get("imgsz", 640),
                classes=det_cfg.get("classes", None),
                device="cpu",
            )
        else:
            from src.ekf_mot.detection.yolo_onnx import OnnxDetector
            detector = OnnxDetector(
                weights=det_cfg.get("weights", weights),
                conf=det_cfg.get("conf", 0.35),
                imgsz=det_cfg.get("imgsz", 640),
            )
        logger.info(f"检测器已加载: backend={backend}, weights={det_cfg.get('weights', weights)}")
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
) -> dict:
    """
    对视频运行检测，返回统计信息（无 GT，无精度指标）。
    """
    import cv2

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

    return {
        "mode": "detection_only",
        "video": video_path,
        "total_frames_processed": processed,
        "total_detections": total_dets,
        "avg_detections_per_frame": round(total_dets / processed, 2) if processed else 0,
        "avg_inference_ms": round(avg_ms, 2),
        "avg_fps": round(avg_fps, 2),
        "frame_results": frame_results,
    }


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
) -> dict:
    """
    对照 GT JSON 运行检测评估。

    如果提供了 video_path，则从视频中读取帧运行检测器；
    否则仅计算 GT 统计（fallback 模式：不运行检测器，输出警告）。
    """
    import cv2
    from src.ekf_mot.detection.evaluator import DetectionEvaluator

    logger.info(f"加载 GT: {gt_json_path}")
    gt_by_frame = load_gt_from_json(gt_json_path)
    logger.info(f"  共 {len(gt_by_frame)} 帧 GT 数据")

    evaluator = DetectionEvaluator(
        iou_threshold=iou_threshold,
        match_class=False,
        per_class=True,
    )

    if video_path is None:
        # Fallback：没有视频，无法运行检测器，只报告 GT 统计
        logger.warning("未提供视频路径，无法运行检测器，输出 GT 统计（仅供参考）")
        total_gt = sum(len(v) for v in gt_by_frame.values())
        return {
            "mode": "gt_stats_only",
            "gt_json": gt_json_path,
            "total_frames": len(gt_by_frame),
            "total_gt_annotations": total_gt,
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

        # 检测
        t0 = time.perf_counter()
        preds = detector.predict(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        elapsed_times.append(elapsed)

        # 设置 frame_id
        for p in preds:
            p.frame_id = frame_id

        # 获取对应帧的 GT（若无则为空）
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
    report["mode"] = "evaluation"

    return report


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="检测器验证脚本（精度评估 / 纯检测统计）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", help="输入视频路径（.mp4 / .avi 等）")
    parser.add_argument("--gt-json", help="GT 标注 JSON 路径（内部统一格式）")
    parser.add_argument("--config", help="实验配置文件路径（YAML）")
    parser.add_argument("--weights", default="weights/yolov8n.pt", help="检测器权重路径")
    parser.add_argument("--output", default="outputs/detection/", help="输出目录")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU 匹配阈值")
    parser.add_argument("--frame-skip", type=int, default=1, help="跳帧间隔")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")
    parser.add_argument("--save-frame-results", action="store_true", help="保存逐帧结果 JSON")
    args = parser.parse_args()

    if not args.video and not args.gt_json:
        parser.error("至少需要提供 --video 或 --gt-json 之一")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 初始化检测器
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"权重文件不存在: {weights_path}")
        logger.error("请先运行: python scripts/download_weights.py")
        sys.exit(1)

    try:
        detector = build_detector(args.config, args.weights)
    except Exception as e:
        logger.error(f"检测器初始化失败: {e}")
        sys.exit(1)

    # 运行评估
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
            )
        else:
            report = run_detection_only(
                detector=detector,
                video_path=args.video,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
            )
    except Exception as e:
        logger.error(f"评估过程失败: {e}", exc_info=True)
        sys.exit(1)

    # 保存报告
    if not args.save_frame_results:
        report_to_save = {k: v for k, v in report.items() if k != "frame_results"}
    else:
        report_to_save = report

    metrics_path = out_dir / "detection_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report_to_save, f, ensure_ascii=False, indent=2)
    logger.info(f"评估报告已保存: {metrics_path}")

    # 打印关键指标
    if "global" in report:
        g = report["global"]
        logger.info(
            f"全局指标: P={g['precision']:.4f}  R={g['recall']:.4f}  "
            f"F1={g['f1']:.4f}  AP50={g['ap50']:.4f}"
        )
    elif "total_detections" in report:
        logger.info(
            f"统计: 处理 {report['total_frames_processed']} 帧, "
            f"共 {report['total_detections']} 个检测, "
            f"平均 {report['avg_fps']:.1f} FPS"
        )


if __name__ == "__main__":
    main()
