"""
实时摄像头跟踪演示
使用本地摄像头进行实时目标检测与轨迹预测
"""

import sys
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.detection import build_detector
from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
from src.ekf_mot.prediction.trajectory_predictor import TrajectoryPredictor
from src.ekf_mot.visualization.draw_tracks import draw_all_tracks
from src.ekf_mot.visualization.draw_future import draw_future_trajectory
from src.ekf_mot.visualization.draw_covariance import draw_covariance_ellipse
from src.ekf_mot.core.config import get_default_config, Config
from src.ekf_mot.utils.logger import setup_logger, get_logger
from src.ekf_mot.tracking.track import Track

logger = setup_logger(level="INFO")


def run_webcam_tracking(
    camera_id: int = 0,
    config_path: str = None,
    show_fps: bool = True,
    save_video: bool = False,
    output_path: str = None,
):
    """
    实时摄像头跟踪主函数

    Args:
        camera_id: 摄像头ID（0=默认摄像头，1=外接摄像头）
        config_path: 配置文件路径（None=使用默认配置）
        show_fps: 是否显示FPS
        save_video: 是否保存输出视频
        output_path: 输出视频路径
    """
    logger.info("=" * 60)
    logger.info("EKF 实时摄像头跟踪系统")
    logger.info("=" * 60)

    # ── 加载配置 ──────────────────────────────────────────────
    if config_path and Path(config_path).exists():
        from src.ekf_mot.core.config import load_config
        cfg_dict = load_config(config_path)
    else:
        cfg_dict = get_default_config()
        # 摄像头优化配置
        cfg_dict['detector']['imgsz'] = 640
        cfg_dict['detector']['conf'] = 0.4
        cfg_dict['tracker']['n_init'] = 2
        cfg_dict['tracker']['max_age'] = 15
        cfg_dict['visualization']['track_history_len'] = 20

    cfg = Config.from_dict(cfg_dict)

    # ── 初始化检测器 ──────────────────────────────────────────
    det_cfg = cfg_dict.get('detector', {})
    logger.info(f"初始化检测器: {det_cfg.get('model', 'yolov8n')}")
    detector = build_detector(
        backend=det_cfg.get('backend', 'ultralytics'),
        weights=det_cfg.get('weights', 'weights/yolov8n.pt'),
        conf=det_cfg.get('conf', 0.4),
        iou=det_cfg.get('iou', 0.5),
        imgsz=det_cfg.get('imgsz', 640),
        max_det=det_cfg.get('max_det', 50),
        classes=det_cfg.get('classes'),  # None=所有类别
        device='cpu',
        warmup=True,
    )

    # ── 初始化跟踪器 ──────────────────────────────────────────
    logger.info("初始化 EKF 跟踪器...")
    Track.reset_id_counter()
    tracker = MultiObjectTracker.from_config(cfg)

    # ── 初始化预测器 ──────────────────────────────────────────
    pred_cfg = cfg_dict.get('prediction', {})
    predictor = TrajectoryPredictor(
        future_steps=pred_cfg.get('future_steps', [1, 5, 10]),
        dt=cfg_dict.get('tracker', {}).get('dt', 0.04),
    )

    # ── 打开摄像头 ────────────────────────────────────────────
    logger.info(f"打开摄像头 {camera_id}...")
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        logger.error(f"无法打开摄像头 {camera_id}")
        logger.info("提示：")
        logger.info("  - 检查摄像头是否连接")
        logger.info("  - 尝试其他 ID：python scripts/run_webcam.py --camera 1")
        logger.info("  - Windows 可能需要授权摄像头访问权限")
        return

    # 获取摄像头参数
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"摄像头参数: {width}x{height} @ {fps:.1f}fps")

    # ── 初始化视频写入器（可选）──────────────────────────────
    video_writer = None
    if save_video:
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = f"outputs/webcam_{_ts}.mp4"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"输出视频: {output_path}")

    # ── 主循环 ────────────────────────────────────────────────
    logger.info("开始跟踪（按 'q' 退出，'r' 重置，'s' 截图）...")
    frame_id = 0
    import time
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    vis_cfg = cfg_dict.get('visualization', {})

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("无法读取摄像头帧")
                break

            # ── 目标检测 ──────────────────────────────────────
            detections = detector.predict(frame)

            # ── 跟踪 ──────────────────────────────────────────
            active_tracks = tracker.step(detections, frame_id, dt=1.0/fps)

            # ── 未来轨迹预测 ──────────────────────────────────
            track_futures = {}
            for track in active_tracks:
                if track.is_confirmed:
                    track_futures[track.track_id] = predictor.predict_track(track, dt=1.0/fps)

            # ── 可视化 ────────────────────────────────────────
            vis_frame = frame.copy()

            # 绘制轨迹
            if vis_cfg.get('draw_tracks', True):
                draw_all_tracks(
                    vis_frame, active_tracks,
                    max_len=vis_cfg.get('track_history_len', 20),
                    thickness=vis_cfg.get('line_thickness', 2),
                    draw_bbox=vis_cfg.get('draw_bbox', True),
                    draw_id=vis_cfg.get('draw_id', True),
                    draw_score=vis_cfg.get('draw_score', True),
                    font_scale=vis_cfg.get('font_scale', 0.5),
                )

            # 绘制未来轨迹
            if vis_cfg.get('draw_future', True):
                for track in active_tracks:
                    if track.is_confirmed and track.track_id in track_futures:
                        cx, cy = track.get_center()
                        draw_future_trajectory(
                            vis_frame, (cx, cy),
                            track_futures[track.track_id],
                            color=tuple(vis_cfg.get('future_color', [0, 255, 255])),
                        )

            # 绘制协方差椭圆
            if vis_cfg.get('draw_covariance', False):
                for track in active_tracks:
                    if track.is_confirmed:
                        cx, cy = track.get_center()
                        draw_covariance_ellipse(
                            vis_frame, cx, cy, track.ekf.P,
                            alpha=vis_cfg.get('covariance_alpha', 0.3),
                        )

            # ── 信息叠加 ──────────────────────────────────────
            # FPS 计算
            fps_counter += 1
            if time.time() - fps_start > 1.0:
                current_fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()

            # 顶部信息栏
            info_lines = [
                f"Frame: {frame_id} | Det: {len(detections)} | Tracks: {len(active_tracks)}",
            ]
            if show_fps:
                info_lines[0] += f" | FPS: {current_fps:.1f}"

            y_offset = 25
            for line in info_lines:
                cv2.putText(
                    vis_frame, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                y_offset += 25

            # 底部提示
            help_text = "Press: 'q'=Quit | 'r'=Reset | 's'=Screenshot"
            cv2.putText(
                vis_frame, help_text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

            # ── 显示 ──────────────────────────────────────────
            cv2.imshow('EKF Real-time Tracking', vis_frame)

            # ── 保存视频 ──────────────────────────────────────
            if video_writer:
                video_writer.write(vis_frame)

            # ── 键盘控制 ──────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("用户退出")
                break
            elif key == ord('r'):
                logger.info("重置跟踪器")
                Track.reset_id_counter()
                tracker.reset()
                frame_id = 0
            elif key == ord('s'):
                _ts_shot = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"outputs/screenshot_{_ts_shot}.jpg"
                Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(screenshot_path, vis_frame)
                logger.info(f"截图已保存: {screenshot_path}")

            frame_id += 1

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        # ── 清理资源 ──────────────────────────────────────────
        cap.release()
        if video_writer:
            video_writer.release()
            logger.info(f"视频已保存: {output_path}")
        cv2.destroyAllWindows()
        logger.info(f"总处理帧数: {frame_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实时摄像头跟踪")
    parser.add_argument("--camera", type=int, default=0, help="摄像头ID（0=默认）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--no-fps", action="store_true", help="不显示FPS")
    parser.add_argument("--save", action="store_true", help="保存输出视频")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径（默认：outputs/webcam_<时间戳>.mp4）")
    args = parser.parse_args()

    run_webcam_tracking(
        camera_id=args.camera,
        config_path=args.config,
        show_fps=not args.no_fps,
        save_video=args.save,
        output_path=args.output,
    )
