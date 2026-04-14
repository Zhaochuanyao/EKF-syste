"""
Ultralytics YOLOv8 推理后端
"""

from pathlib import Path
from typing import List, Optional
import numpy as np

from .base import DetectorBase
from ..core.types import Detection
from ..core.constants import COCO_CLASSES
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.detection.ultralytics")


class UltralyticsDetector(DetectorBase):
    """
    基于 ultralytics 库的 YOLOv8 检测器。
    支持 CPU 推理，无需 GPU。
    """

    def load_model(self) -> None:
        """加载 YOLOv8 模型"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装 ultralytics: pip install ultralytics")

        weights_path = Path(self.weights)
        if not weights_path.exists():
            logger.warning(f"权重文件不存在: {weights_path}，尝试自动下载...")
            # ultralytics 会自动下载官方权重
            model_name = weights_path.stem  # e.g. "yolov8n"
            self._model = YOLO(f"{model_name}.pt")
            logger.info(f"已下载并加载模型: {model_name}")
        else:
            self._model = YOLO(str(weights_path))
            logger.info(f"已加载模型: {weights_path}")

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        对单帧执行检测。

        Args:
            frame: BGR 格式图像 (H, W, 3)

        Returns:
            Detection 列表
        """
        self.ensure_loaded()

        results = self._model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()      # (N, 4)
        scores = boxes.conf.cpu().numpy()    # (N,)
        class_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

        # 获取类别名称映射
        names = result.names or COCO_CLASSES

        for i in range(len(xyxy)):
            cid = int(class_ids[i])
            det = Detection(
                bbox=xyxy[i].astype(np.float64),
                score=float(scores[i]),
                class_id=cid,
                class_name=names.get(cid, str(cid)),
            )
            detections.append(det)

        return detections
