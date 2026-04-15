"""
类型定义模块 - 定义系统中使用的核心数据结构
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """单个目标检测结果"""
    bbox: np.ndarray        # [x1, y1, x2, y2] 格式，像素坐标
    score: float            # 置信度 [0, 1]
    class_id: int           # 类别ID
    class_name: str = ""    # 类别名称
    frame_id: int = 0       # 所属帧ID

    @property
    def cx(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2)

    @property
    def cy(self) -> float:
        return float((self.bbox[1] + self.bbox[3]) / 2)

    @property
    def w(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def h(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    def to_cxcywh(self) -> np.ndarray:
        """转换为 [cx, cy, w, h] 格式"""
        return np.array([self.cx, self.cy, self.w, self.h], dtype=np.float64)

    def to_measurement(self, anchor_mode: str = "center") -> np.ndarray:
        """
        转换为 EKF 观测向量。

        anchor_mode:
          "center"        - [cx, cy, w, h]  (几何中心，默认)
          "bottom_center" - [cx, y2, w, h]  (底部中心，减少行人垂直抖动)
          "auto"          - 与 "center" 相同（保留扩展接口）
        """
        if anchor_mode == "bottom_center":
            return np.array(
                [self.cx, float(self.bbox[3]), self.w, self.h], dtype=np.float64
            )
        return self.to_cxcywh()


@dataclass
class TrackStateVector:
    """EKF 状态向量封装"""
    x: np.ndarray           # 状态均值向量 [cx, cy, v, theta, omega, w, h]
    P: np.ndarray           # 状态协方差矩阵 (7x7)

    @property
    def cx(self) -> float:
        return float(self.x[0])

    @property
    def cy(self) -> float:
        return float(self.x[1])

    @property
    def v(self) -> float:
        return float(self.x[2])

    @property
    def theta(self) -> float:
        return float(self.x[3])

    @property
    def omega(self) -> float:
        return float(self.x[4])

    @property
    def w(self) -> float:
        return float(self.x[5])

    @property
    def h(self) -> float:
        return float(self.x[6])

    def to_bbox(self, anchor_mode: str = "center") -> np.ndarray:
        """
        转换为 [x1, y1, x2, y2] 格式。

        anchor_mode:
          "center"        - cy 是几何中心 y（默认）
          "bottom_center" - cy 是底部 y，y1 = cy-h，y2 = cy
        """
        if anchor_mode == "bottom_center":
            x1 = self.cx - self.w / 2
            y2 = self.cy          # cy stores bottom y in bottom_center mode
            y1 = y2 - self.h
            x2 = self.cx + self.w / 2
            return np.array([x1, y1, x2, y2], dtype=np.float64)
        x1 = self.cx - self.w / 2
        y1 = self.cy - self.h / 2
        x2 = self.cx + self.w / 2
        y2 = self.cy + self.h / 2
        return np.array([x1, y1, x2, y2], dtype=np.float64)


@dataclass
class Measurement:
    """EKF 观测值封装"""
    z: np.ndarray           # 观测向量 [cx, cy, w, h]
    R: Optional[np.ndarray] = None  # 观测噪声协方差（可选，用于自适应）
    score: float = 1.0      # 检测置信度（用于自适应R）
    frame_id: int = 0
    # 自适应 R 所需的目标尺寸信息（可选，不填则跳过自适应）
    bbox_w: Optional[float] = None      # 检测框宽度（像素），size_adaptive 使用
    bbox_h: Optional[float] = None      # 检测框高度（像素），size_adaptive 使用
    aspect_ratio: Optional[float] = None  # 宽高比 w/h，aspect_adaptive 使用


@dataclass
class PredictionResult:
    """单条轨迹的预测结果"""
    track_id: int
    frame_id: int
    # 当前状态
    bbox: np.ndarray            # [x1, y1, x2, y2]
    score: float
    class_id: int
    class_name: str
    state_name: str             # Tentative/Confirmed/Lost
    filtered_center: Tuple[float, float]  # EKF滤波后的中心点
    # 历史轨迹
    history: List[Tuple[float, float]] = field(default_factory=list)
    # 未来预测
    predicted_future_points: dict = field(default_factory=dict)  # {step: (cx, cy)}
    predicted_future_bboxes: dict = field(default_factory=dict)  # {step: [x1,y1,x2,y2]}
    # 不确定性
    covariance_ellipse: Optional[dict] = None  # 协方差椭圆参数
    # 质量指标（新增）
    stability_score: float = 0.0          # 轨迹稳定性 [0,1]
    velocity_valid: bool = False           # 速度估计是否有效
    heading_valid: bool = False            # 航向估计是否有效
    position_uncertainty: float = 0.0     # 位置不确定性（协方差迹）
    prediction_valid: bool = False         # 未来预测是否满足质量门限
    prediction_confidence: float = 0.0    # 预测置信度 [0,1]


@dataclass
class FrameResult:
    """单帧处理结果"""
    frame_id: int
    timestamp: float
    tracks: List[PredictionResult] = field(default_factory=list)
    num_detections: int = 0
    num_active_tracks: int = 0
    process_time_ms: float = 0.0
