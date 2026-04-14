"""
后处理模块 - 检测框过滤、坐标映射、类别筛选

支持过滤策略:
  - 置信度过滤
  - 类别过滤
  - 面积过滤（最小面积）
  - 短边过滤（最小短边像素）
  - 宽高比过滤（正常比例范围）
  - 边界过滤（去除靠近图像边界的检测）
  - 数量上限（保留 top-K）
"""

from typing import List, Optional, Tuple
import numpy as np

from ..core.types import Detection
from ..core.constants import COCO_CLASSES


def filter_by_conf(detections: List[Detection], conf: float) -> List[Detection]:
    """按置信度过滤"""
    return [d for d in detections if d.score >= conf]


def filter_by_class(
    detections: List[Detection],
    classes: Optional[List[int]],
) -> List[Detection]:
    """按类别过滤，classes=None 则不过滤"""
    if classes is None:
        return detections
    return [d for d in detections if d.class_id in classes]


def filter_by_size(
    detections: List[Detection],
    min_area: float = 100.0,
) -> List[Detection]:
    """过滤过小的检测框（面积过滤）"""
    return [d for d in detections if d.w * d.h >= min_area]


def filter_by_short_side(
    detections: List[Detection],
    min_short_side: float = 10.0,
) -> List[Detection]:
    """
    过滤短边过小的检测框。

    短边过小往往是噪声或极端遮挡，不适合作为跟踪输入。

    Args:
        detections: 检测列表
        min_short_side: 最小短边像素长度
    """
    return [d for d in detections if min(d.w, d.h) >= min_short_side]


def filter_by_aspect_ratio(
    detections: List[Detection],
    min_aspect: float = 0.1,
    max_aspect: float = 10.0,
) -> List[Detection]:
    """
    过滤宽高比不合理的检测框。

    极端宽高比通常是误检，过滤后可减少误匹配。

    Args:
        detections: 检测列表
        min_aspect: 最小 w/h
        max_aspect: 最大 w/h
    """
    result = []
    for d in detections:
        if d.h <= 0:
            continue
        ar = d.w / d.h
        if min_aspect <= ar <= max_aspect:
            result.append(d)
    return result


def filter_by_border(
    detections: List[Detection],
    img_w: float,
    img_h: float,
    border_margin: float = 5.0,
) -> List[Detection]:
    """
    过滤完全紧贴图像边界的检测框。

    此类框通常是截断目标，位置不准确，
    可能对关联造成干扰。

    Args:
        detections: 检测列表
        img_w: 图像宽度（像素）
        img_h: 图像高度（像素）
        border_margin: 到边界的最小距离（像素）
    """
    result = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        # 检测框如果几乎完全贴合边界则过滤
        if (x1 <= border_margin and x2 <= border_margin) or \
           (x1 >= img_w - border_margin and x2 >= img_w - border_margin) or \
           (y1 <= border_margin and y2 <= border_margin) or \
           (y1 >= img_h - border_margin and y2 >= img_h - border_margin):
            continue
        result.append(d)
    return result


def limit_detections(
    detections: List[Detection],
    max_det: int,
) -> List[Detection]:
    """按置信度降序保留前 max_det 个"""
    if len(detections) <= max_det:
        return detections
    return sorted(detections, key=lambda d: d.score, reverse=True)[:max_det]


def postprocess(
    detections: List[Detection],
    conf: float = 0.35,
    classes: Optional[List[int]] = None,
    max_det: int = 100,
    min_area: float = 100.0,
    min_short_side: float = 0.0,
    min_aspect: float = 0.0,
    max_aspect: float = 0.0,
    img_w: float = 0.0,
    img_h: float = 0.0,
    border_margin: float = 0.0,
) -> List[Detection]:
    """
    完整后处理流程。

    Args:
        detections: 原始检测列表
        conf: 置信度阈值
        classes: 允许的类别 ID 列表（None=全部）
        max_det: 最大保留数量
        min_area: 最小面积（像素²）
        min_short_side: 最小短边（像素，0=不过滤）
        min_aspect: 最小宽高比（0=不过滤）
        max_aspect: 最大宽高比（0=不过滤）
        img_w: 图像宽度（0=不过滤边界）
        img_h: 图像高度（0=不过滤边界）
        border_margin: 边界过滤距离（0=不过滤）
    """
    dets = filter_by_conf(detections, conf)
    dets = filter_by_class(dets, classes)
    dets = filter_by_size(dets, min_area)

    if min_short_side > 0:
        dets = filter_by_short_side(dets, min_short_side)

    if min_aspect > 0 or max_aspect > 0:
        _min = min_aspect if min_aspect > 0 else 0.01
        _max = max_aspect if max_aspect > 0 else 1000.0
        dets = filter_by_aspect_ratio(dets, _min, _max)

    if img_w > 0 and img_h > 0 and border_margin > 0:
        dets = filter_by_border(dets, img_w, img_h, border_margin)

    dets = limit_detections(dets, max_det)
    return dets
