"""API 数据模式定义"""
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class TrackInfo(BaseModel):
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    class_name: str
    state: str
    center: Tuple[float, float]
    future_points: Dict[int, Tuple[float, float]] = {}
    # 历史轨迹（原始 EKF 滤波后 / EMA 平滑后）
    raw_history: List[Tuple[float, float]] = []
    smoothed_history: List[Tuple[float, float]] = []
    # EKF 运动状态
    velocity: float = 0.0
    heading: float = 0.0
    omega: float = 0.0
    # 恢复保护标志
    recovered_recently: bool = False


class FramePredictRequest(BaseModel):
    """单帧预测请求（base64 编码图像）"""
    image_base64: str
    frame_id: int = 0
    config_name: str = "demo_vehicle_accuracy"   # 跟踪配置名（车辆为默认）


class FramePredictResponse(BaseModel):
    frame_id: int
    tracks: List[TrackInfo]
    num_detections: int
    process_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
