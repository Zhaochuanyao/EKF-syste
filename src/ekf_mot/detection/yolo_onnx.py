"""
ONNX Runtime CPU 推理后端
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .base import DetectorBase
from ..core.types import Detection
from ..core.constants import COCO_CLASSES
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.detection.onnx")

# YOLOv8 ONNX 输出格式: (1, 84, 8400) -> 84 = 4(box) + 80(classes)
_YOLOV8_NUM_CLASSES = 80


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """YOLOv8 ONNX 输出的 cx,cy,w,h -> x1,y1,x2,y2"""
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """简单 NMS 实现"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


class OnnxDetector(DetectorBase):
    """
    基于 ONNX Runtime 的 YOLOv8 CPU 推理检测器。
    不依赖 GPU，适合普通笔记本运行。
    """

    def __init__(self, *args, onnx_providers: Optional[List[str]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.onnx_providers = onnx_providers or ["CPUExecutionProvider"]
        self._input_name: Optional[str] = None
        self._input_shape: Optional[Tuple] = None

    def load_model(self) -> None:
        """加载 ONNX 模型"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")

        onnx_path = Path(self.weights)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX 模型文件不存在: {onnx_path}\n"
                f"请先运行: python scripts/export_onnx.py"
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        self._model = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=self.onnx_providers,
        )
        self._input_name = self._model.get_inputs()[0].name
        self._input_shape = self._model.get_inputs()[0].shape
        logger.info(f"已加载 ONNX 模型: {onnx_path} | 输入: {self._input_shape}")

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        预处理图像：letterbox + 归一化 + 转置为 NCHW。

        Returns:
            (blob, scale, (pad_w, pad_h))
        """
        import cv2
        h, w = frame.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        pad_w = (self.imgsz - new_w) // 2
        pad_h = (self.imgsz - new_h) // 2
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR -> RGB, HWC -> CHW, 归一化
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]  # (1, 3, H, W)
        return blob, scale, (pad_w, pad_h)

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """执行 ONNX 推理"""
        self.ensure_loaded()

        orig_h, orig_w = frame.shape[:2]
        blob, scale, (pad_w, pad_h) = self._preprocess(frame)

        # 推理
        outputs = self._model.run(None, {self._input_name: blob})
        # YOLOv8 输出: (1, 84, 8400)
        pred = outputs[0][0].T  # (8400, 84)

        # 解析 box + class scores
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]  # (8400, 80)

        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)

        # 置信度过滤
        mask = scores >= self.conf
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 类别过滤
        if self.classes is not None:
            cls_mask = np.isin(class_ids, self.classes)
            boxes_xywh = boxes_xywh[cls_mask]
            scores = scores[cls_mask]
            class_ids = class_ids[cls_mask]

        if len(scores) == 0:
            return []

        # xywh -> xyxy（letterbox 坐标系）
        boxes_xyxy = _xywh2xyxy(boxes_xywh)

        # NMS
        keep = _nms(boxes_xyxy, scores, self.iou)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # 还原到原始图像坐标
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / scale
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / scale
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / scale
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / scale

        # 裁剪到图像边界
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, orig_w)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, orig_h)

        detections: List[Detection] = []
        for i in range(min(len(keep), self.max_det)):
            cid = int(class_ids[i])
            det = Detection(
                bbox=boxes_xyxy[i].astype(np.float64),
                score=float(scores[i]),
                class_id=cid,
                class_name=COCO_CLASSES.get(cid, str(cid)),
            )
            detections.append(det)

        return detections
