"""
主入口 - 完整跟踪流程
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from .core.config import Config, get_default_config, load_config
from .core.types import Detection, PredictionResult
from .core.constants import IDX_CX, IDX_CY
from .data.video_reader import VideoReader
from .data.frame_sampler import FrameSampler
from .detection import build_detector
from .tracking.multi_object_tracker import MultiObjectTracker
from .prediction.trajectory_predictor import TrajectoryPredictor
from .prediction.uncertainty import covariance_to_ellipse
from .visualization.draw_tracks import draw_all_tracks
from .visualization.draw_future import draw_future_trajectory
from .visualization.draw_covariance import draw_covariance_ellipse
from .visualization.video_writer import VideoWriter
from .metrics.runtime_metrics import RuntimeMetrics
from .utils.logger import setup_logger, get_logger
from .utils.file_io import save_json, save_csv, ensure_dir
from .utils.seed import set_seed
from .utils.timer import Timer

logger = get_logger("ekf_mot.main")


def build_config(config_path: Optional[str], overrides: Dict = None) -> Dict:
    """加载并合并配置"""
    base = get_default_config()
    if config_path and Path(config_path).exists():
        exp = load_config(config_path, None)
        from .core.config import _deep_merge
        cfg = _deep_merge(base, exp)
    else:
        cfg = base
    if overrides:
        from .core.config import _deep_merge
        cfg = _deep_merge(cfg, overrides)
    return cfg


def run_tracking(
    config_path: Optional[str] = None,
    video_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    show: bool = False,
) -> Dict:
    """
    完整跟踪流程主函数。

    Args:
        config_path: 配置文件路径
        video_path: 输入视频路径
        output_dir: 输出目录
        max_frames: 最大处理帧数
        frame_skip: 跳帧间隔
        show: 是否实时显示（需要 GUI）

    Returns:
        统计信息字典
    """
    # ── 加载配置 ──────────────────────────────────────────────
    cfg_dict = build_config(config_path)
    cfg = Config.from_dict(cfg_dict)

    # 命令行参数覆盖配置
    if video_path:
        cfg_dict.setdefault("data", {})["video_path"] = video_path
    if output_dir:
        cfg_dict.setdefault("output", {})["output_dir"] = output_dir
    if max_frames:
        cfg_dict.setdefault("runtime", {})["max_frames"] = max_frames
    if frame_skip > 1:
        cfg_dict.setdefault("runtime", {})["frame_skip"] = frame_skip
    cfg = Config.from_dict(cfg_dict)

    # ── 初始化日志 ────────────────────────────────────────────
    log_cfg = cfg_dict.get("logging", {})
    setup_logger(
        level=log_cfg.get("level", "INFO"),
        fmt=log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
        log_file=log_cfg.get("file"),
    )

    # ── 设置随机种子 ──────────────────────────────────────────
    set_seed(cfg_dict.get("runtime", {}).get("seed", 42))

    # ── 确定视频路径 ──────────────────────────────────────────
    data_cfg = cfg_dict.get("data", {})
    _video_path = data_cfg.get("video_path", "assets/samples/demo.mp4")
    if not Path(_video_path).exists():
        logger.error(f"视频文件不存在: {_video_path}")
        logger.info("提示: 请运行 python scripts/download_weights.py 并准备示例视频")
        return {}

    # ── 确定输出目录 ──────────────────────────────────────────
    out_cfg = cfg_dict.get("output", {})
    _output_dir = Path(out_cfg.get("output_dir", "outputs/"))
    ensure_dir(_output_dir)

    # ── 初始化检测器 ──────────────────────────────────────────
    det_cfg = cfg_dict.get("detector", {})
    logger.info(f"初始化检测器: backend={det_cfg.get('backend')} model={det_cfg.get('model')}")
    detector = build_detector(
        backend=det_cfg.get("backend", "ultralytics"),
        weights=det_cfg.get("weights", "weights/yolov8n.pt"),
        onnx_path=det_cfg.get("onnx_path", "weights/yolov8n.onnx"),
        conf=det_cfg.get("conf", 0.35),
        iou=det_cfg.get("iou", 0.5),
        imgsz=det_cfg.get("imgsz", 640),
        max_det=det_cfg.get("max_det", 100),
        classes=det_cfg.get("classes"),
        device=det_cfg.get("device", "cpu"),
        warmup=True,
    )

    # ── 初始化跟踪器 ──────────────────────────────────────────
    logger.info("初始化多目标跟踪器 (EKF-CTRV, 三阶段关联)...")
    tracker = MultiObjectTracker.from_config(cfg)

    # ── 初始化预测器 ──────────────────────────────────────────
    pred_cfg = cfg_dict.get("prediction", {})
    tracker_cfg = cfg_dict.get("tracker", {})
    predictor = TrajectoryPredictor(
        future_steps=pred_cfg.get("future_steps", [1, 5, 10]),
        dt=tracker_cfg.get("dt", 0.04),
        min_hits_for_prediction=pred_cfg.get("min_hits_for_prediction", 3),
        max_position_cov_trace=pred_cfg.get("max_position_cov_trace", 1e6),
        fixed_lag_smoothing=pred_cfg.get("fixed_lag_smoothing", False),
        smoothing_lag=pred_cfg.get("smoothing_lag", 3),
    )

    # ── 打开视频 ──────────────────────────────────────────────
    reader = VideoReader(_video_path)
    _frame_skip = cfg_dict.get("runtime", {}).get("frame_skip", 1)
    _max_frames = cfg_dict.get("runtime", {}).get("max_frames")
    sampler = FrameSampler(reader, interval=_frame_skip, max_frames=_max_frames)

    # 自动估算 dt
    if tracker_cfg.get("auto_dt", True):
        effective_fps = sampler.effective_fps
        dt = 1.0 / effective_fps if effective_fps > 0 else 0.04
        logger.info(f"自动估算 dt={dt:.4f}s (有效FPS={effective_fps:.1f})")
    else:
        dt = tracker_cfg.get("dt", 0.04)

    # ── 初始化视频写入器 ──────────────────────────────────────
    vis_cfg = cfg_dict.get("visualization", {})
    video_writer = None
    if out_cfg.get("save_video", True):
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_video = _output_dir / f"output_{_ts}.mp4"
        video_writer = VideoWriter(
            str(out_video),
            fps=out_cfg.get("video_fps", 25),
            codec=out_cfg.get("video_codec", "mp4v"),
        )

    # ── 主循环 ────────────────────────────────────────────────
    runtime_metrics = RuntimeMetrics()
    timer = Timer()
    all_frame_results = []
    csv_rows = []
    frame_count = 0

    logger.info(f"开始处理视频: {_video_path}")
    logger.info(f"输出目录: {_output_dir}")

    for frame_id, frame in sampler:
        runtime_metrics.start_frame()

        # ── 目标检测 ──────────────────────────────────────────
        with timer.measure("detection"):
            detections: List[Detection] = detector.predict(frame)

        # ── EKF 预测 + 关联 + 更新 ───────────────────────────
        with timer.measure("tracking"):
            active_tracks = tracker.step(detections, frame_id, dt=dt)

        # ── 未来轨迹预测（含质量门控和置信度）────────────────
        with timer.measure("prediction"):
            track_futures = {}
            track_confidence = {}
            for track in active_tracks:
                future_pts, conf, valid = predictor.predict_with_confidence(track, dt)
                if valid:
                    track_futures[track.track_id] = future_pts
                    track_confidence[track.track_id] = conf

        # ── 可视化 ────────────────────────────────────────────
        with timer.measure("visualization"):
            vis_frame = frame.copy()

            if vis_cfg.get("draw_tracks", True):
                draw_all_tracks(
                    vis_frame, active_tracks,
                    max_len=vis_cfg.get("track_history_len", 30),
                    thickness=vis_cfg.get("line_thickness", 2),
                    draw_bbox=vis_cfg.get("draw_bbox", True),
                    draw_id=vis_cfg.get("draw_id", True),
                    draw_score=vis_cfg.get("draw_score", True),
                    font_scale=vis_cfg.get("font_scale", 0.5),
                )

            if vis_cfg.get("draw_future", True):
                for track in active_tracks:
                    if track.track_id in track_futures:
                        cx, cy = track.get_center()
                        draw_future_trajectory(
                            vis_frame, (cx, cy),
                            track_futures[track.track_id],
                            color=tuple(vis_cfg.get("future_color", [0, 255, 255])),
                        )

            if vis_cfg.get("draw_covariance", True):
                for track in active_tracks:
                    if track.is_confirmed:
                        cx, cy = track.get_center()
                        draw_covariance_ellipse(
                            vis_frame, cx, cy,
                            track.ekf.P,
                            alpha=vis_cfg.get("covariance_alpha", 0.3),
                        )

            # 帧信息叠加（新增 confirmed/tentative/lost 统计）
            n_confirmed = sum(1 for t in active_tracks if t.is_confirmed)
            n_tentative = sum(1 for t in active_tracks if t.is_tentative)
            n_predicting = len(track_futures)
            info_text = (
                f"Frame:{frame_id} | Det:{len(detections)} | "
                f"Conf:{n_confirmed} Tent:{n_tentative} | Pred:{n_predicting}"
            )
            cv2.putText(
                vis_frame, info_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        # ── 写入视频 ──────────────────────────────────────────
        if video_writer:
            video_writer.write(vis_frame)

        if show:
            cv2.imshow("EKF Tracking", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ── 收集结果（含新增质量字段）────────────────────────
        frame_data = {
            "frame_id": frame_id,
            "timestamp": frame_id / reader.fps,
            "num_detections": len(detections),
            "tracks": [],
        }

        for track in active_tracks:
            cx, cy = track.get_center()
            bbox = track.get_bbox()
            future = track_futures.get(track.track_id, {})
            conf = track_confidence.get(track.track_id, 0.0)
            pos_uncertainty = track.ekf.get_position_uncertainty_trace()

            track_info = {
                "frame_id": frame_id,
                "track_id": track.track_id,
                "bbox": bbox.tolist(),
                "score": round(track.score, 4),
                "class_id": track.class_id,
                "class_name": track.class_name,
                "state_name": track.state.name,
                "filtered_center": [round(cx, 2), round(cy, 2)],
                # 质量字段（新增）
                "stability_score": round(track.stability_score, 4),
                "velocity_valid": track.velocity_valid,
                "heading_valid": track.heading_valid,
                "position_uncertainty": round(pos_uncertainty, 2),
                "prediction_valid": track.track_id in track_futures,
                "prediction_confidence": round(conf, 4),
                # 预测点
                "predicted_future_points": {
                    str(k): [round(v[0], 2), round(v[1], 2)]
                    for k, v in future.items()
                },
            }
            frame_data["tracks"].append(track_info)

            csv_rows.append({
                "frame_id": frame_id,
                "track_id": track.track_id,
                "x1": round(float(bbox[0]), 2),
                "y1": round(float(bbox[1]), 2),
                "x2": round(float(bbox[2]), 2),
                "y2": round(float(bbox[3]), 2),
                "score": round(track.score, 4),
                "class_id": track.class_id,
                "class_name": track.class_name,
                "state": track.state.name,
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "stability_score": round(track.stability_score, 4),
                "prediction_valid": int(track.track_id in track_futures),
                "prediction_confidence": round(conf, 4),
                "position_uncertainty": round(pos_uncertainty, 2),
            })

        all_frame_results.append(frame_data)

        elapsed_ms = runtime_metrics.end_frame()
        frame_count += 1

        if frame_count % 50 == 0:
            stats = runtime_metrics.compute()
            logger.info(
                f"已处理 {frame_count} 帧 | "
                f"FPS: {stats.get('fps', 0):.1f} | "
                f"活跃轨迹: {len(active_tracks)} "
                f"(Conf:{n_confirmed} Tent:{n_tentative}) | "
                f"有效预测: {n_predicting}"
            )

    # ── 收尾 ──────────────────────────────────────────────────
    reader.release()
    if video_writer:
        video_writer.release()
    if show:
        cv2.destroyAllWindows()

    # ── 保存结果 ──────────────────────────────────────────────
    if out_cfg.get("save_json", True):
        json_path = _output_dir / "tracks.json"
        save_json(all_frame_results, json_path)
        logger.info(f"轨迹 JSON 已保存: {json_path}")

    if out_cfg.get("save_csv", True) and csv_rows:
        csv_path = _output_dir / "tracks.csv"
        save_csv(csv_rows, csv_path)
        logger.info(f"轨迹 CSV 已保存: {csv_path}")

    # ── 打印统计 ──────────────────────────────────────────────
    stats = runtime_metrics.compute()
    timer_stats = timer.stats()

    logger.info("=" * 50)
    logger.info(f"处理完成 | 总帧数: {frame_count}")
    logger.info(f"平均 FPS: {stats.get('fps', 0):.1f}")
    logger.info(f"平均帧耗时: {stats.get('avg_frame_ms', 0):.1f} ms")
    for name, s in timer_stats.items():
        logger.info(f"  {name}: {s['mean_ms']:.1f} ms/帧")
    logger.info("=" * 50)

    return {
        "frames_processed": frame_count,
        "fps": stats.get("fps", 0),
        "output_dir": str(_output_dir),
        "timer_stats": timer_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="EKF 多目标跟踪系统")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--video", type=str, default=None, help="输入视频路径")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")
    parser.add_argument("--frame-skip", type=int, default=1, help="跳帧间隔")
    parser.add_argument("--show", action="store_true", help="实时显示（需要GUI）")
    args = parser.parse_args()

    run_tracking(
        config_path=args.config,
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        show=args.show,
    )


if __name__ == "__main__":
    main()
