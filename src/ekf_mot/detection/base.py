"""
检测器基类
"""

from abc import abstractmethod
from typing import List, Optional
import numpy as np

from ..core.interfaces import BaseDetector
from ..core.types import Detection
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.detection")


class DetectorBase(BaseDetector):
    """
    检测器基类，提供通用的初始化和预热逻辑。
    子类需实现 load_model 和 predict。
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        max_det: int = 100,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
    ) -> None:
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.classes = classes
        self.device = device
        self._model = None
        self._loaded = False

    def ensure_loaded(self) -> None:
        """确保模型已加载"""
        if not self._loaded:
            self.load_model()
            self._loaded = True

    def warmup(self, imgsz: Optional[int] = None) -> None:
        """预热模型，减少首帧延迟"""
        self.ensure_loaded()
        size = imgsz or self.imgsz
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        logger.info(f"模型预热中 (imgsz={size})...")
        self.predict(dummy)
        logger.info("模型预热完成")
