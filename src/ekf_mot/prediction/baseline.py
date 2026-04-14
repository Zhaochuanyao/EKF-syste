"""
检测中心轨迹 Baseline — 无 EKF 滤波的纯 IoU 关联跟踪器

设计目标
--------
作为 EKF 系统的对比基线，使用最简单的跟踪策略：
  1. 每帧运行检测器
  2. 用 IoU 贪心匹配将检测框关联到已有轨迹
  3. 直接使用检测中心（无滤波）作为轨迹位置
  4. 无预测步骤，"未来位置"通过线性外推估算

通过对比 Baseline 和 EKF 系统的轨迹抖动（Jitter）与平滑度（Smoothness），
可以量化 EKF 滤波带来的改善效果。

API
---
    tracker = BaselineTracker(iou_threshold=0.3, max_age=5, min_hits=1)
    for frame_id, frame in video:
        dets = detector.predict(frame)
        tracks = tracker.step(dets, frame_id)
        # tracks: List[BaselineTrack]

    summary = tracker.get_summary()
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _iou(box_a: List[float], box_b: List[float]) -> float:
    """计算两个 [x1,y1,x2,y2] 框的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _center(bbox: List[float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


# ══════════════════════════════════════════════════════════════
# BaselineTrack — 单条轨迹（无 EKF）
# ══════════════════════════════════════════════════════════════

class BaselineTrack:
    """
    无 EKF 的纯检测跟踪轨迹。

    轨迹中心 = 当前检测框的几何中心，无任何滤波。
    """

    _id_counter: int = 0

    def __init__(self, detection_bbox: List[float], frame_id: int,
                 class_id: int = 0, class_name: str = "object") -> None:
        BaselineTrack._id_counter += 1
        self.track_id: int = BaselineTrack._id_counter

        self.bbox: List[float] = list(detection_bbox)
        self.class_id = class_id
        self.class_name = class_name
        self.frame_id_created = frame_id
        self.frame_id_last = frame_id
        self.hits: int = 1
        self.age: int = 1
        self.time_since_update: int = 0

        cx, cy = _center(detection_bbox)
        # 历史中心点（原始检测，无滤波）
        self.history: List[Tuple[float, float]] = [(cx, cy)]
        self._confirmed: bool = False

    @property
    def is_confirmed(self) -> bool:
        return self._confirmed

    def get_center(self) -> Tuple[float, float]:
        return _center(self.bbox)

    def predict_linear(self, steps: List[int]) -> Dict[int, Tuple[float, float]]:
        """
        线性外推：用最近两帧中心点估算未来位置。
        若历史不足两点，返回当前位置的复制。
        """
        if len(self.history) < 2:
            cx, cy = self.get_center()
            return {s: (cx, cy) for s in steps}
        dx = self.history[-1][0] - self.history[-2][0]
        dy = self.history[-1][1] - self.history[-2][1]
        cx, cy = self.get_center()
        return {s: (cx + dx * s, cy + dy * s) for s in steps}

    def update(self, detection_bbox: List[float], frame_id: int) -> None:
        self.bbox = list(detection_bbox)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.frame_id_last = frame_id
        cx, cy = _center(detection_bbox)
        self.history.append((cx, cy))

    def mark_missed(self) -> None:
        self.age += 1
        self.time_since_update += 1

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0


# ══════════════════════════════════════════════════════════════
# BaselineTracker — 无 EKF 多目标跟踪器
# ══════════════════════════════════════════════════════════════

class BaselineTracker:
    """
    基于纯 IoU 关联的 baseline 多目标跟踪器。

    参数
    ----
    iou_threshold : 匹配 IoU 阈值（默认 0.3，略低于 EKF 关联阈值）
    max_age       : 目标丢失最多允许的帧数
    min_hits      : 确认轨迹所需最少命中次数
    future_steps  : 线性外推预测步数列表
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 3,
        future_steps: Optional[List[int]] = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.future_steps = future_steps or [1, 5, 10]

        self._tracks: List[BaselineTrack] = []
        self._removed: List[BaselineTrack] = []

    def step(self, detections, frame_id: int) -> List[BaselineTrack]:
        """
        处理一帧检测结果。

        Args:
            detections: List[Detection]（与主系统 Detection 类型兼容）
            frame_id:   当前帧 ID

        Returns:
            当前活跃轨迹列表（包含 Tentative 和 Confirmed）
        """
        from src.ekf_mot.core.types import Detection

        # 转换 Detection → bbox list
        det_bboxes: List[List[float]] = []
        det_meta: List[Dict] = []
        for d in detections:
            bbox = d.bbox.tolist() if hasattr(d.bbox, "tolist") else list(d.bbox)
            det_bboxes.append(bbox)
            det_meta.append({
                "class_id": int(d.class_id),
                "class_name": str(d.class_name),
                "score": float(d.score),
            })

        # ── IoU 矩阵 ──────────────────────────────────────────
        n_tracks = len(self._tracks)
        n_dets = len(det_bboxes)

        matched_tracks: set = set()
        matched_dets: set = set()

        if n_tracks > 0 and n_dets > 0:
            iou_mat = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for ti, track in enumerate(self._tracks):
                for di, dbbox in enumerate(det_bboxes):
                    iou_mat[ti, di] = _iou(track.bbox, dbbox)

            # 贪心匹配（按 IoU 降序）
            pairs = sorted(
                [(ti, di, float(iou_mat[ti, di]))
                 for ti in range(n_tracks)
                 for di in range(n_dets)],
                key=lambda x: -x[2],
            )
            for ti, di, iou in pairs:
                if iou < self.iou_threshold:
                    break
                if ti in matched_tracks or di in matched_dets:
                    continue
                matched_tracks.add(ti)
                matched_dets.add(di)
                self._tracks[ti].update(det_bboxes[di], frame_id)

        # 未匹配轨迹 → mark_missed
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.mark_missed()

        # 未匹配检测 → 创建新轨迹
        for di in range(n_dets):
            if di not in matched_dets:
                meta = det_meta[di]
                t = BaselineTrack(
                    detection_bbox=det_bboxes[di],
                    frame_id=frame_id,
                    class_id=meta["class_id"],
                    class_name=meta["class_name"],
                )
                self._tracks.append(t)

        # 移除超时轨迹；更新 Confirmed 状态
        active = []
        for track in self._tracks:
            if track.time_since_update > self.max_age:
                self._removed.append(track)
            else:
                if track.hits >= self.min_hits:
                    track._confirmed = True
                active.append(track)
        self._tracks = active

        return active

    def get_summary(self) -> Dict:
        """返回跟踪统计摘要"""
        all_tracks = self._tracks + self._removed
        if not all_tracks:
            return {
                "num_tracks": 0,
                "avg_track_length": 0.0,
                "avg_jitter": 0.0,
                "avg_smoothness": 0.0,
            }

        lengths = [len(t.history) for t in all_tracks]
        jitters = [_compute_jitter(t.history) for t in all_tracks if len(t.history) >= 2]
        smoothness = [_compute_smoothness(t.history) for t in all_tracks if len(t.history) >= 3]

        return {
            "num_tracks": len(all_tracks),
            "avg_track_length": round(float(np.mean(lengths)), 2),
            "avg_jitter": round(float(np.mean(jitters)) if jitters else 0.0, 4),
            "avg_smoothness": round(float(np.mean(smoothness)) if smoothness else 0.0, 4),
        }

    def reset(self) -> None:
        self._tracks.clear()
        self._removed.clear()
        BaselineTrack.reset_id_counter()


# ══════════════════════════════════════════════════════════════
# 轨迹质量指标
# ══════════════════════════════════════════════════════════════

def _compute_jitter(history: List[Tuple[float, float]]) -> float:
    """
    轨迹抖动 = 帧间位移的标准差（越小越平稳）。

    物理含义：速度的标准差，即速度波动幅度。
    """
    if len(history) < 2:
        return 0.0
    disps = [
        math.sqrt((history[i][0] - history[i - 1][0]) ** 2
                  + (history[i][1] - history[i - 1][1]) ** 2)
        for i in range(1, len(history))
    ]
    return float(np.std(disps))


def _compute_smoothness(history: List[Tuple[float, float]]) -> float:
    """
    轨迹平滑度 = 帧间加速度的均值（越小越平滑）。

    物理含义：速度变化幅度，反映轨迹曲折程度。
    """
    if len(history) < 3:
        return 0.0
    disps = [
        math.sqrt((history[i][0] - history[i - 1][0]) ** 2
                  + (history[i][1] - history[i - 1][1]) ** 2)
        for i in range(1, len(history))
    ]
    accels = [abs(disps[i] - disps[i - 1]) for i in range(1, len(disps))]
    return float(np.mean(accels))


def compute_track_quality(history: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    计算单条轨迹的质量指标。

    Returns:
        {jitter, smoothness, length, avg_speed}
    """
    n = len(history)
    if n < 2:
        return {"jitter": 0.0, "smoothness": 0.0, "length": n, "avg_speed": 0.0}

    disps = [
        math.sqrt((history[i][0] - history[i - 1][0]) ** 2
                  + (history[i][1] - history[i - 1][1]) ** 2)
        for i in range(1, n)
    ]
    return {
        "jitter": round(_compute_jitter(history), 4),
        "smoothness": round(_compute_smoothness(history), 4),
        "length": n,
        "avg_speed": round(float(np.mean(disps)), 4),
    }
