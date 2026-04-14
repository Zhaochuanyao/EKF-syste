"""
服务逻辑封装
"""

import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from ..detection import build_detector
from ..tracking.multi_object_tracker import MultiObjectTracker
from ..prediction.trajectory_predictor import TrajectoryPredictor
from ..core.config import get_default_config, Config
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.serving")


def _id_to_color(track_id: int) -> tuple:
    """将轨迹ID映射为 BGR 颜色"""
    palette = [
        (34, 87, 255),   # 蓝
        (243, 150, 33),  # 橙
        (80, 175, 76),   # 绿
        (212, 188, 0),   # 青
        (176, 39, 156),  # 紫
        (0, 165, 255),   # 橙黄
        (99, 30, 233),   # 品红
        (0, 200, 150),   # 青绿
        (255, 64, 128),  # 粉
        (128, 128, 0),   # 橄榄
    ]
    return palette[track_id % len(palette)]


class TrackingService:
    """
    跟踪服务，封装检测器 + 跟踪器 + 预测器的完整流程。
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg_dict = get_default_config()
        self.cfg = Config.from_dict(cfg_dict) if config is None else config

        det_cfg = self.cfg.detector
        self.detector = build_detector(
            backend=getattr(det_cfg, "backend", "ultralytics"),
            weights=getattr(det_cfg, "weights", "weights/yolov8n.pt"),
            conf=getattr(det_cfg, "conf", 0.35),
            iou=getattr(det_cfg, "iou", 0.5),
            imgsz=getattr(det_cfg, "imgsz", 640),
            warmup=True,
        )

        self.tracker = MultiObjectTracker.from_config(self.cfg)

        pred_cfg = self.cfg.prediction
        self.predictor = TrajectoryPredictor(
            future_steps=getattr(pred_cfg, "future_steps", [1, 5, 10]),
            dt=getattr(self.cfg.tracker, "dt", 0.04),
        )
        self._frame_id = 0

    def process_frame(self, frame: np.ndarray):
        """处理单帧，返回 (results_list, num_detections, elapsed_ms)"""
        t0 = time.perf_counter()

        detections = self.detector.predict(frame)
        active_tracks = self.tracker.step(detections, self._frame_id)

        results = []
        for track in active_tracks:
            future: Dict = {}
            if track.is_confirmed:
                future = self.predictor.predict_track(track)
            cx, cy = track.get_center()
            bbox = track.get_bbox()
            results.append({
                "track_id": track.track_id,
                "bbox": {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                },
                "score": float(track.score),
                "class_id": track.class_id,
                "class_name": track.class_name,
                "state": track.state.name,
                "center": [float(cx), float(cy)],
                "future_points": {
                    str(k): [float(v[0]), float(v[1])]
                    for k, v in future.items()
                },
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._frame_id += 1
        return results, len(detections), elapsed_ms

    def process_video_to_file(
        self,
        video_path: str,
        task_id: str = "",
        show_tracks: bool = True,
        show_future: bool = True,
        frame_skip: int = 1,
        tasks: Optional[Dict] = None,
    ) -> dict:
        """
        处理视频文件，输出带可视化标注的结果视频。
        返回结果字典，包含输出文件名和统计信息。
        """
        # 尝试导入可视化模块
        try:
            from ..visualization.draw_tracks import draw_all_tracks
            from ..visualization.draw_future import draw_future_trajectory
            _has_vis = True
        except ImportError:
            _has_vis = False
            logger.warning("可视化模块不可用，使用基础 OpenCV 绘制")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tid_short = task_id[:6] if task_id else "video"
        out_filename = f"output_{ts}_{tid_short}.mp4"
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        out_path = output_dir / out_filename

        # 优先使用 avc1 (H.264)，浏览器可直接播放；mp4v 输出 FMP4 格式浏览器不支持
        fourcc_avc1 = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc_avc1, fps, (width, height))
        if not writer.isOpened():
            # avc1 不可用时回退 mp4v（视频无法在浏览器播放，需外部转换）
            logger.warning("avc1 编码器不可用，回退到 mp4v（视频可能无法在浏览器播放）")
            fourcc_mp4v = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc_mp4v, fps, (width, height))

        # 重置跟踪器状态
        self.tracker.reset()
        self._frame_id = 0

        frame_idx = 0
        processed = 0
        total_detections = 0
        unique_track_ids: set = set()
        frame_skip = max(1, frame_skip)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            detections = self.detector.predict(frame)
            active_tracks = self.tracker.step(detections, self._frame_id)
            self._frame_id += 1

            total_detections += len(detections)
            for t in active_tracks:
                unique_track_ids.add(t.track_id)

            vis = frame.copy()

            if show_tracks:
                if _has_vis:
                    draw_all_tracks(
                        vis, active_tracks,
                        max_len=30, thickness=2,
                        draw_bbox=True, draw_id=True,
                        draw_score=True, font_scale=0.5,
                    )
                else:
                    # 基础绘制 fallback
                    for track in active_tracks:
                        bbox = track.get_bbox()
                        x1, y1 = int(bbox[0]), int(bbox[1])
                        x2, y2 = int(bbox[2]), int(bbox[3])
                        color = _id_to_color(track.track_id)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        label = f"ID:{track.track_id} {track.class_name}"
                        cv2.putText(
                            vis, label, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        )

            if show_future:
                for track in active_tracks:
                    if track.is_confirmed:
                        future = self.predictor.predict_track(track)
                        if not future:
                            continue
                        if _has_vis:
                            cx, cy = track.get_center()
                            draw_future_trajectory(
                                vis, (cx, cy), future, color=(0, 255, 255)
                            )
                        else:
                            for _step, pt in future.items():
                                px, py = int(pt[0]), int(pt[1])
                                if 0 <= px < width and 0 <= py < height:
                                    cv2.circle(vis, (px, py), 4, (0, 255, 255), -1)

            # 信息叠加
            n_conf = sum(1 for t in active_tracks if t.is_confirmed)
            info = f"Frame:{frame_idx} | Tracks:{len(active_tracks)} Confirmed:{n_conf}"
            cv2.putText(
                vis, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )

            writer.write(vis)
            processed += 1
            frame_idx += 1

            # 更新进度
            if tasks is not None and task_id and total_frames > 0:
                tasks[task_id]["progress"] = min(99, int(frame_idx / total_frames * 100))

        cap.release()
        writer.release()

        duration_s = round(total_frames / fps, 1) if fps > 0 else 0

        return {
            "output_file": out_filename,
            "frames_processed": processed,
            "total_detections": total_detections,
            "unique_tracks": len(unique_track_ids),
            "fps": round(fps, 1),
            "duration_seconds": duration_s,
        }

    def decode_base64_image(self, b64: str) -> Optional[np.ndarray]:
        """解码 base64 图像（支持 data URL 格式）"""
        try:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            data = base64.b64decode(b64)
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"图像解码失败: {e}")
            return None
