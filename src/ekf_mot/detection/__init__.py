"""
检测器工厂函数
"""

from typing import Optional, List
from .base import DetectorBase
from .yolo_ultralytics import UltralyticsDetector
from .yolo_onnx import OnnxDetector
from ..core.constants import BACKEND_ULTRALYTICS, BACKEND_ONNX
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.detection")


def build_detector(
    backend: str = BACKEND_ULTRALYTICS,
    weights: str = "weights/yolov8n.pt",
    onnx_path: str = "weights/yolov8n.onnx",
    conf: float = 0.35,
    iou: float = 0.5,
    imgsz: int = 640,
    max_det: int = 100,
    classes: Optional[List[int]] = None,
    device: str = "cpu",
    warmup: bool = True,
) -> DetectorBase:
    """
    根据配置构建检测器实例。

    Args:
        backend: 推理后端，"ultralytics" 或 "onnx"
        weights: 模型权重路径
        onnx_path: ONNX 模型路径（backend=onnx 时使用）
        ...

    Returns:
        检测器实例
    """
    if backend == BACKEND_ULTRALYTICS:
        detector = UltralyticsDetector(
            weights=weights,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            classes=classes,
            device=device,
        )
    elif backend == BACKEND_ONNX:
        detector = OnnxDetector(
            weights=onnx_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            classes=classes,
            device=device,
        )
    else:
        raise ValueError(f"不支持的检测后端: {backend}，可选: ultralytics / onnx")

    detector.load_model()
    if warmup:
        detector.warmup()

    return detector
