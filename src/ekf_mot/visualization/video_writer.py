"""
视频写入模块
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from ..utils.logger import get_logger

logger = get_logger("ekf_mot.visualization.video_writer")


class VideoWriter:
    """
    OpenCV 视频写入器封装。

    用法:
        writer = VideoWriter("output.mp4", fps=25, size=(1280, 720))
        writer.write(frame)
        writer.release()
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 25.0,
        size: Optional[Tuple[int, int]] = None,
        codec: str = "mp4v",
    ) -> None:
        """
        Args:
            output_path: 输出视频路径
            fps: 帧率
            size: (width, height)，None 则在第一帧时自动确定
            codec: 视频编码器（mp4v / XVID / avc1）
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.size = size
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_writer(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, self.size
        )
        if not self._writer.isOpened():
            logger.warning(f"mp4v 编码器失败，尝试 XVID...")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_path = self.output_path.with_suffix(".avi")
            self._writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, self.size)
            self.output_path = out_path
        logger.info(f"视频写入器初始化: {self.output_path} | {w}x{h} @ {self.fps}fps")

    def write(self, frame: np.ndarray) -> None:
        """写入一帧"""
        if self._writer is None:
            self._init_writer(frame)
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(f"视频已保存: {self.output_path}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()
