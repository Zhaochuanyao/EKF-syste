"""
抽象接口定义模块 - 定义系统各模块的抽象基类
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from .types import Detection, TrackStateVector, Measurement, PredictionResult


class BaseDetector(ABC):
    """目标检测器抽象基类"""

    @abstractmethod
    def load_model(self) -> None:
        """加载模型权重"""
        ...

    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        对单帧图像执行目标检测

        Args:
            frame: BGR格式的图像帧 (H, W, 3)

        Returns:
            检测结果列表
        """
        ...

    def warmup(self, imgsz: int = 640) -> None:
        """模型预热，减少首帧延迟"""
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.predict(dummy)


class BaseFilter(ABC):
    """状态滤波器抽象基类"""

    @abstractmethod
    def predict(self, dt: Optional[float] = None) -> TrackStateVector:
        """
        执行预测步骤

        Args:
            dt: 时间步长（秒），None则使用默认值

        Returns:
            预测后的状态
        """
        ...

    @abstractmethod
    def update(self, measurement: Measurement) -> TrackStateVector:
        """
        执行更新步骤

        Args:
            measurement: 观测值

        Returns:
            更新后的状态
        """
        ...

    @abstractmethod
    def get_state(self) -> TrackStateVector:
        """获取当前状态"""
        ...


class BaseTracker(ABC):
    """多目标跟踪器抽象基类"""

    @abstractmethod
    def step(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_id: int,
    ) -> List[PredictionResult]:
        """
        处理单帧，返回当前活跃轨迹

        Args:
            frame: 当前帧图像
            detections: 当前帧检测结果
            frame_id: 帧ID

        Returns:
            活跃轨迹列表
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置跟踪器状态"""
        ...
