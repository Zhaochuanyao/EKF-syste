"""
对 demo.mp4 运行完整跟踪器，将 Confirmed 轨迹的边界框保存为伪 GT JSON。

说明：
  这不是真实标注数据，而是用跟踪器输出作为"参考基准"，
  目的是让检测评估流水线（validate_detector.py）能够完整跑通，
  展示 TP/FP/FN 统计结果。

  真实评估需要人工标注的 GT（如 UA-DETRAC / MOT17）。

用法：
  python scripts/generate_demo_gt.py
  python scripts/generate_demo_gt.py --video assets/samples/demo.mp4 --output data/processed/demo_gt.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_demo_gt")


def main():
    parser = argparse.ArgumentParser(description="生成 demo 视频的伪 GT JSON（用于检测评估演示）")
    parser.add_argument("--video", default="assets/samples/demo.mp4")
    parser.add_argument("--output", default="data/processed/demo_gt.json")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.35, help="检测置信度阈值")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"视频文件不存在: {video_path}")
        sys.exit(1)

    import cv2
    from src.ekf_mot.detection import build_detector
    from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker

    # ── 初始化检测器 ─────────────────────────────────────────────
    logger.info("加载检测器 (YOLOv8n)...")
    detector = build_detector(
        backend="ultralytics",
        weights="weights/yolov8n.pt",
        conf=args.conf,
        iou=0.5,
        imgsz=640,
        warmup=True,
    )

    # ── 初始化跟踪器 ─────────────────────────────────────────────
    tracker = MultiObjectTracker(n_init=3, max_age=15, dt=0.1)

    # ── 逐帧处理 ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"视频: {video_path.name} | FPS={fps:.1f} | 总帧数={total}")

    frames_data = []
    frame_id = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if args.max_frames and processed >= args.max_frames:
            break

        # 检测
        dets = detector.predict(frame)
        # 跟踪
        tracker.step(dets, frame_id)

        # 只保存 Confirmed 状态轨迹的 bbox 作为伪 GT
        annotations = []
        for track in tracker.manager.tracks:
            if track.is_confirmed:
                bbox = track.ekf.x  # 状态向量
                from src.ekf_mot.core.constants import IDX_CX, IDX_CY, IDX_W, IDX_H
                cx = float(bbox[IDX_CX])
                cy = float(bbox[IDX_CY])
                w  = float(bbox[IDX_W])
                h  = float(bbox[IDX_H])
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                annotations.append({
                    "id": track.track_id,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "class_id": track.class_id if hasattr(track, "class_id") else 0,
                    "class_name": track.class_name if hasattr(track, "class_name") else "object",
                })

        frames_data.append({"frame_id": frame_id, "annotations": annotations})
        processed += 1

        if processed % 50 == 0:
            logger.info(f"  已处理 {processed} 帧，当前确认轨迹数={len(annotations)}")

    cap.release()

    # ── 保存 JSON ────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_ann = sum(len(f["annotations"]) for f in frames_data)
    data = {
        "dataset": "demo_pseudo_gt",
        "sequence": video_path.stem,
        "fps": fps,
        "total_frames": len(frames_data),
        "note": "由跟踪器 Confirmed 轨迹生成的伪 GT，仅用于演示评估流水线",
        "frames": frames_data,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"伪 GT 已保存: {out_path}")
    logger.info(f"  帧数={len(frames_data)}, 标注总数={total_ann}")
    logger.info("")
    logger.info("下一步，运行检测评估（获得 TP/FP/FN）：")
    logger.info(f"  python scripts/validate_detector.py \\")
    logger.info(f"      --video {args.video} \\")
    logger.info(f"      --gt-json {args.output} \\")
    logger.info(f"      --output outputs/detection/")


if __name__ == "__main__":
    main()
