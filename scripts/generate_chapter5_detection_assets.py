"""
论文第 5 章目标检测实验资产生成脚本

基于 UA-DETRAC 数据集生成：
  - 图5-3：目标检测可视化结果图
  - 表5-5：目标检测定量结果表（Precision、Recall、F1、AP50）
  - 表5-6：检测运行效率表
  - 图5-4：检测失败案例图

用法：
  python scripts/generate_chapter5_detection_assets.py \\
      --config configs/data/uadetrac_subset_cpu.yaml \\
      --output outputs/chapter5/detection \\
      --model weights/yolov8n.pt \\
      --detector-backend auto \\
      --conf-thres 0.25 \\
      --iou-thres 0.50 \\
      --vehicle-only

  或直接指定 UA-DETRAC 路径：
  python scripts/generate_chapter5_detection_assets.py \\
      --uadetrac-root data/UA-DETRAC \\
      --split train \\
      --output outputs/chapter5/detection \\
      --model weights/yolov8n.pt
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.detection.yolo_ultralytics import UltralyticsDetector
from src.ekf_mot.metrics.detection_metrics import DetectionMetrics, match_detections
from src.ekf_mot.core.types import Detection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chapter5_detection")


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class FrameDetectionResult:
    """单帧检测结果"""
    sequence: str
    frame_id: int
    image_path: Path
    detections: List[Detection]
    ground_truths: List[Detection]
    inference_time_ms: float
    tp: int
    fp: int
    fn: int


@dataclass
class FailureCase:
    """失败案例"""
    case_type: str  # "missed", "false_positive", "occlusion", "small_object"
    sequence: str
    frame_id: int
    image_path: Path
    detections: List[Detection]
    ground_truths: List[Detection]
    description: str


# ══════════════════════════════════════════════════════════════
# UA-DETRAC 数据加载
# ══════════════════════════════════════════════════════════════

def load_yaml(path: Path) -> dict:
    """加载 YAML 配置文件"""
    try:
        import yaml
    except ImportError:
        logger.error("缺少 pyyaml，请运行: pip install pyyaml")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_uadetrac_xml(xml_path: Path) -> Dict[int, List[Dict]]:
    """
    解析 UA-DETRAC XML 标注文件。

    Returns:
        {frame_id: [{"bbox": [x1,y1,x2,y2], "class_name": "car", ...}, ...]}
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("xml.etree.ElementTree 是 Python 标准库")

    logger.info(f"  解析标注: {xml_path.name}")

    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        raise ValueError(f"XML 解析失败 {xml_path}: {e}") from e

    root = tree.getroot()
    frame_dict: Dict[int, List[Dict]] = defaultdict(list)

    for frame_el in root.findall("frame"):
        frame_num_str = frame_el.get("num", "0")
        try:
            frame_id = int(frame_num_str)
        except ValueError:
            continue

        target_list_el = frame_el.find("target_list")
        if target_list_el is None:
            continue

        for target_el in target_list_el.findall("target"):
            target_id_str = target_el.get("id", "0")
            try:
                target_id = int(target_id_str)
            except ValueError:
                target_id = 0

            box_el = target_el.find("box")
            if box_el is None:
                continue

            try:
                left = float(box_el.get("left", 0))
                top = float(box_el.get("top", 0))
                w = float(box_el.get("width", 0))
                h = float(box_el.get("height", 0))
            except (TypeError, ValueError):
                continue

            x1, y1, x2, y2 = left, top, left + w, top + h

            # 解析类别和遮挡信息
            attr_el = target_el.find("attribute")
            vehicle_type = "car"
            occlusion = 0
            truncation = 0

            if attr_el is not None:
                vehicle_type = attr_el.get("vehicle_type", "car").lower()
                try:
                    occlusion = int(attr_el.get("occlusion", 0))
                    truncation = int(attr_el.get("truncation", 0))
                except (TypeError, ValueError):
                    pass

            frame_dict[frame_id].append({
                "id": target_id,
                "bbox": [x1, y1, x2, y2],
                "class_name": vehicle_type,
                "occlusion": occlusion,
                "truncation": truncation,
            })

    return dict(frame_dict)


def load_uadetrac_sequences(
    uadetrac_root: Path,
    split: str,
    sequence_list: Optional[List[str]] = None,
    max_frames_per_seq: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    加载 UA-DETRAC 序列信息。

    Returns:
        [{"name": "MVI_20011", "image_dir": Path, "anno_path": Path, "max_frames": int}, ...]
    """
    image_base = uadetrac_root / "DETRAC-Images" / "DETRAC-Images"

    if split == "train":
        anno_base = uadetrac_root / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"
    elif split == "test":
        anno_base = uadetrac_root / "DETRAC-Test-Annotations-XML" / "DETRAC-Test-Annotations-XML"
    else:
        anno_base = uadetrac_root / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"

    if not image_base.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_base}")
    if not anno_base.exists():
        raise FileNotFoundError(f"标注目录不存在: {anno_base}")

    sequences = []

    if sequence_list:
        seq_names = sequence_list
    else:
        # 扫描所有序列
        seq_names = sorted([d.name for d in image_base.iterdir() if d.is_dir() and d.name.startswith("MVI_")])

    for seq_name in seq_names:
        seq_dir = image_base / seq_name
        if not seq_dir.exists():
            logger.warning(f"  序列目录不存在，跳过: {seq_name}")
            continue

        # 查找标注文件
        anno_path = anno_base / f"{seq_name}_v3.xml"
        if not anno_path.exists():
            anno_path = anno_base / f"{seq_name}.xml"

        if not anno_path.exists():
            logger.warning(f"  标注文件不存在，跳过: {seq_name}")
            continue

        sequences.append({
            "name": seq_name,
            "image_dir": seq_dir,
            "anno_path": anno_path,
            "max_frames": max_frames_per_seq,
        })

    logger.info(f"加载了 {len(sequences)} 个 UA-DETRAC 序列")
    return sequences



# ══════════════════════════════════════════════════════════════
# 检测器加载
# ══════════════════════════════════════════════════════════════

def load_detector(
    model_path: str,
    backend: str = "auto",
    conf_thres: float = 0.25,
    device: str = "cpu",
) -> UltralyticsDetector:
    """加载检测器"""
    logger.info(f"加载检测器: {model_path}")
    logger.info(f"  backend={backend}, conf={conf_thres}, device={device}")

    detector = UltralyticsDetector(
        weights=model_path,
        conf=conf_thres,
        device=device,
    )
    detector.ensure_loaded()
    detector.warmup()

    return detector


# ══════════════════════════════════════════════════════════════
# 检测执行
# ══════════════════════════════════════════════════════════════

def run_detection_on_uadetrac(
    sequences: List[Dict[str, Any]],
    detector: UltralyticsDetector,
    iou_thres: float = 0.5,
    max_frames: int = 500,
    frame_stride: int = 1,
    vehicle_only: bool = True,
) -> List[FrameDetectionResult]:
    """
    在 UA-DETRAC 序列上运行检测。

    Returns:
        所有帧的检测结果列表
    """
    logger.info(f"开始检测，共 {len(sequences)} 个序列")
    logger.info(f"  max_frames={max_frames}, frame_stride={frame_stride}, vehicle_only={vehicle_only}")

    all_results: List[FrameDetectionResult] = []
    vehicle_classes = {"car", "bus", "van", "others"}

    for seq_info in sequences:
        seq_name = seq_info["name"]
        image_dir = seq_info["image_dir"]
        anno_path = seq_info["anno_path"]
        seq_max_frames = seq_info.get("max_frames") or max_frames

        logger.info(f"\n处理序列: {seq_name}")

        # 加载标注
        try:
            gt_dict = parse_uadetrac_xml(anno_path)
        except Exception as e:
            logger.error(f"  标注解析失败: {e}")
            continue

        # 获取图像文件列表
        image_files = sorted(image_dir.glob("img*.jpg"))
        if not image_files:
            logger.warning(f"  未找到图像文件")
            continue

        # 限制帧数
        image_files = image_files[:seq_max_frames:frame_stride]
        logger.info(f"  处理 {len(image_files)} 帧")

        # 逐帧检测
        for img_idx, img_path in enumerate(image_files):
            frame_id = img_idx * frame_stride

            # 读取图像
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"  无法读取图像: {img_path.name}")
                continue

            # 执行检测（计时）
            start_time = time.perf_counter()
            detections = detector.predict(frame)
            inference_time_ms = (time.perf_counter() - start_time) * 1000

            # 过滤车辆类别
            if vehicle_only:
                detections = [d for d in detections if d.class_name.lower() in vehicle_classes]

            # 获取 GT
            gt_list = gt_dict.get(frame_id, [])
            if vehicle_only:
                gt_list = [g for g in gt_list if g["class_name"].lower() in vehicle_classes]

            # 转换 GT 为 Detection 对象
            ground_truths = []
            for gt in gt_list:
                gt_det = Detection(
                    bbox=np.array(gt["bbox"], dtype=np.float64),
                    score=1.0,
                    class_id=0,
                    class_name=gt["class_name"],
                )
                gt_det.frame_id = frame_id
                gt_det.occlusion = gt.get("occlusion", 0)
                gt_det.truncation = gt.get("truncation", 0)
                ground_truths.append(gt_det)

            # 设置 frame_id
            for det in detections:
                det.frame_id = frame_id

            # 计算 TP/FP/FN
            tp, fp, fn, _ = match_detections(detections, ground_truths, iou_threshold=iou_thres)

            result = FrameDetectionResult(
                sequence=seq_name,
                frame_id=frame_id,
                image_path=img_path,
                detections=detections,
                ground_truths=ground_truths,
                inference_time_ms=inference_time_ms,
                tp=tp,
                fp=fp,
                fn=fn,
            )
            all_results.append(result)

            if (img_idx + 1) % 50 == 0:
                logger.info(f"    已处理 {img_idx + 1}/{len(image_files)} 帧")

    logger.info(f"\n检测完成，共处理 {len(all_results)} 帧")
    return all_results


# ══════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════

def compute_detection_metrics(results: List[FrameDetectionResult], iou_thres: float = 0.5) -> Dict[str, Any]:
    """计算检测指标（表5-5）"""
    logger.info("\n计算检测指标...")

    metrics = DetectionMetrics(iou_threshold=iou_thres)

    for result in results:
        metrics.update(result.detections, result.ground_truths)

    metric_dict = metrics.compute()

    logger.info(f"  Precision: {metric_dict['precision']:.4f}")
    logger.info(f"  Recall:    {metric_dict['recall']:.4f}")
    logger.info(f"  F1:        {metric_dict['f1']:.4f}")
    logger.info(f"  AP50:      {metric_dict['ap50']:.4f}")
    logger.info(f"  TP={metric_dict['tp']}, FP={metric_dict['fp']}, FN={metric_dict['fn']}")

    return metric_dict


def compute_efficiency_metrics(
    results: List[FrameDetectionResult],
    model_name: str,
    device: str,
) -> Dict[str, Any]:
    """计算效率指标（表5-6）"""
    logger.info("\n计算效率指标...")

    if not results:
        return {}

    # 统计序列数
    sequences = set(r.sequence for r in results)
    sequence_count = len(sequences)

    # 统计帧数和检测目标数
    total_frames = len(results)
    total_detections = sum(len(r.detections) for r in results)
    avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0

    # 统计推理时间
    inference_times = [r.inference_time_ms for r in results]
    avg_inference_ms = np.mean(inference_times)
    avg_fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0

    efficiency = {
        "dataset": "UA-DETRAC",
        "sequence_count": sequence_count,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "avg_detections_per_frame": round(avg_detections_per_frame, 2),
        "avg_inference_ms": round(avg_inference_ms, 2),
        "avg_fps": round(avg_fps, 2),
        "device": device,
        "model_name": model_name,
    }

    logger.info(f"  测试序列数: {sequence_count}")
    logger.info(f"  测试帧数: {total_frames}")
    logger.info(f"  检测目标总数: {total_detections}")
    logger.info(f"  平均每帧目标数: {avg_detections_per_frame:.2f}")
    logger.info(f"  平均推理时间: {avg_inference_ms:.2f} ms")
    logger.info(f"  平均 FPS: {avg_fps:.2f}")

    return efficiency



# ══════════════════════════════════════════════════════════════
# 可视化帧选择
# ══════════════════════════════════════════════════════════════

def select_visualization_frames(results: List[FrameDetectionResult], num_frames: int = 4) -> List[FrameDetectionResult]:
    """
    选择用于可视化的代表性帧（图5-3）。

    选择标准：
    1. 检测目标数量较多
    2. 检测准确率较高（TP 较多）
    3. 来自不同序列
    """
    logger.info(f"\n选择可视化帧（目标 {num_frames} 帧）...")

    # 过滤：至少有一些检测结果
    candidates = [r for r in results if len(r.detections) >= 3]

    if len(candidates) < num_frames:
        logger.warning(f"  候选帧不足，使用所有可用帧")
        candidates = results

    # 按检测数量和准确率排序
    def score_frame(r: FrameDetectionResult) -> float:
        num_dets = len(r.detections)
        accuracy = r.tp / (r.tp + r.fp + 1e-9)
        return num_dets * 0.5 + accuracy * 10

    candidates.sort(key=score_frame, reverse=True)

    # 选择来自不同序列的帧
    selected = []
    used_sequences = set()

    for candidate in candidates:
        if len(selected) >= num_frames:
            break
        if candidate.sequence not in used_sequences:
            selected.append(candidate)
            used_sequences.add(candidate.sequence)

    # 如果还不够，从剩余候选中选择
    if len(selected) < num_frames:
        for candidate in candidates:
            if len(selected) >= num_frames:
                break
            if candidate not in selected:
                selected.append(candidate)

    logger.info(f"  选择了 {len(selected)} 帧")
    for r in selected:
        logger.info(f"    {r.sequence} 帧{r.frame_id}: {len(r.detections)} 个检测, TP={r.tp}")

    return selected[:num_frames]


def select_failure_cases(
    results: List[FrameDetectionResult],
    iou_thres: float = 0.5,
    small_object_ratio: float = 0.002,
) -> Dict[str, Optional[FailureCase]]:
    """
    选择失败案例（图5-4）。

    返回：
    {
        "missed": FailureCase or None,
        "false_positive": FailureCase or None,
        "occlusion": FailureCase or None,
        "small_object": FailureCase or None,
    }
    """
    logger.info("\n选择失败案例...")

    cases: Dict[str, Optional[FailureCase]] = {
        "missed": None,
        "false_positive": None,
        "occlusion": None,
        "small_object": None,
    }

    for result in results:
        frame = cv2.imread(str(result.image_path))
        if frame is None:
            continue
        img_area = frame.shape[0] * frame.shape[1]

        # 1. 漏检案例：GT 存在但未被检测
        if cases["missed"] is None and result.fn > 0:
            # 找到未匹配的 GT
            _, _, _, matched_pairs = match_detections(
                result.detections, result.ground_truths, iou_threshold=iou_thres
            )
            matched_gt_indices = {gt_idx for _, gt_idx in matched_pairs}
            missed_gts = [gt for i, gt in enumerate(result.ground_truths) if i not in matched_gt_indices]

            if missed_gts:
                cases["missed"] = FailureCase(
                    case_type="missed",
                    sequence=result.sequence,
                    frame_id=result.frame_id,
                    image_path=result.image_path,
                    detections=result.detections,
                    ground_truths=missed_gts,
                    description=f"漏检 {len(missed_gts)} 个目标",
                )

        # 2. 误检案例：检测框不匹配任何 GT
        if cases["false_positive"] is None and result.fp > 0:
            _, _, _, matched_pairs = match_detections(
                result.detections, result.ground_truths, iou_threshold=iou_thres
            )
            matched_pred_indices = {pred_idx for pred_idx, _ in matched_pairs}
            false_preds = [det for i, det in enumerate(result.detections) if i not in matched_pred_indices]

            if false_preds:
                cases["false_positive"] = FailureCase(
                    case_type="false_positive",
                    sequence=result.sequence,
                    frame_id=result.frame_id,
                    image_path=result.image_path,
                    detections=false_preds,
                    ground_truths=result.ground_truths,
                    description=f"误检 {len(false_preds)} 个目标",
                )

        # 3. 遮挡案例：使用 GT 的 occlusion 字段
        if cases["occlusion"] is None:
            occluded_gts = [gt for gt in result.ground_truths if getattr(gt, "occlusion", 0) >= 2]
            if occluded_gts:
                cases["occlusion"] = FailureCase(
                    case_type="occlusion",
                    sequence=result.sequence,
                    frame_id=result.frame_id,
                    image_path=result.image_path,
                    detections=result.detections,
                    ground_truths=occluded_gts,
                    description=f"遮挡场景（{len(occluded_gts)} 个遮挡目标）",
                )

        # 4. 小目标案例
        if cases["small_object"] is None:
            small_gts = []
            for gt in result.ground_truths:
                bbox = gt.bbox
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area = w * h
                if area / img_area < small_object_ratio:
                    small_gts.append(gt)

            if small_gts:
                cases["small_object"] = FailureCase(
                    case_type="small_object",
                    sequence=result.sequence,
                    frame_id=result.frame_id,
                    image_path=result.image_path,
                    detections=result.detections,
                    ground_truths=small_gts,
                    description=f"远距离小目标（{len(small_gts)} 个）",
                )

        # 如果所有案例都找到了，提前退出
        if all(v is not None for v in cases.values()):
            break

    # 输出结果
    for case_type, case in cases.items():
        if case:
            logger.info(f"  {case_type}: {case.sequence} 帧{case.frame_id} - {case.description}")
        else:
            logger.info(f"  {case_type}: 未找到")

    return cases


# ══════════════════════════════════════════════════════════════
# 可视化绘图
# ══════════════════════════════════════════════════════════════

def setup_chinese_font() -> FontProperties:
    """查找系统中文字体，返回 FontProperties"""
    from matplotlib import font_manager
    preferred = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS", "WenQuanYi Micro Hei"]
    for name in preferred:
        matches = font_manager.findfont(FontProperties(family=name), fallback_to_default=False)
        if matches and "DejaVu" not in matches:
            return FontProperties(family=name)
    return FontProperties()


def draw_detection_visualization(
    selected_frames: List[FrameDetectionResult],
    output_path: Path,
    selected_frames_dir: Path,
) -> None:
    """绘制图5-3：目标检测可视化结果图"""
    logger.info("\n绘制图5-3：目标检测可视化结果图...")

    selected_frames_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    chinese_font = setup_chinese_font()

    for idx, result in enumerate(selected_frames[:4]):
        if idx >= 4:
            break

        ax = axes[idx]

        # 读取图像
        frame = cv2.imread(str(result.image_path))
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 保存原始帧
        frame_save_path = selected_frames_dir / f"{result.sequence}_frame{result.frame_id}.jpg"
        cv2.imwrite(str(frame_save_path), frame)

        # 绘制检测框
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)

            # 标注类别和置信度
            label = f"{det.class_name} {det.score:.2f}"
            ax.text(
                x1, y1 - 5, label,
                color='green', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

        ax.imshow(frame_rgb)
        ax.axis('off')

        # 子图标题
        title = f"{result.sequence} 帧{result.frame_id}\n检测目标: {len(result.detections)}"
        ax.set_title(title, fontsize=12, fontproperties=chinese_font)

    # 整体标题
    fig.suptitle("图5-3 目标检测可视化结果图", fontsize=16, fontproperties=chinese_font, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  已保存: {output_path}")



def draw_failure_cases(
    cases: Dict[str, Optional[FailureCase]],
    output_path: Path,
) -> None:
    """绘制图5-4：检测失败案例图"""
    logger.info("\n绘制图5-4：检测失败案例图...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    chinese_font = setup_chinese_font()

    case_order = ["missed", "false_positive", "occlusion", "small_object"]
    case_titles = {
        "missed": "漏检案例",
        "false_positive": "误检案例",
        "occlusion": "遮挡案例",
        "small_object": "远距离小目标案例",
    }

    for idx, case_type in enumerate(case_order):
        ax = axes[idx]
        case = cases.get(case_type)

        if case is None:
            # 显示"未找到案例"
            ax.text(
                0.5, 0.5, f"未在当前子集中检出该类案例",
                ha='center', va='center', fontsize=14,
                fontproperties=chinese_font
            )
            ax.set_title(case_titles[case_type], fontsize=12, fontproperties=chinese_font)
            ax.axis('off')
            continue

        # 读取图像
        frame = cv2.imread(str(case.image_path))
        if frame is None:
            ax.text(0.5, 0.5, "图像加载失败", ha='center', va='center')
            ax.axis('off')
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 绘制检测框（蓝色）
        for det in case.detections:
            x1, y1, x2, y2 = det.bbox
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='blue', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)

        # 绘制 GT 框（根据案例类型选择颜色）
        if case_type == "missed":
            # 漏检：GT 用红色标出
            for gt in case.ground_truths:
                x1, y1, x2, y2 = gt.bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, "漏检", color='red', fontsize=10,
                        fontproperties=chinese_font,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

        elif case_type == "false_positive":
            # 误检：检测框用橙色标出
            for det in case.detections:
                x1, y1, x2, y2 = det.bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=3, edgecolor='orange', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, "误检", color='orange', fontsize=10,
                        fontproperties=chinese_font,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))

        elif case_type == "occlusion":
            # 遮挡：GT 用黄色标出
            for gt in case.ground_truths:
                x1, y1, x2, y2 = gt.bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=3, edgecolor='yellow', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, "遮挡", color='yellow', fontsize=10,
                        fontproperties=chinese_font,
                        bbox=dict(facecolor='black', alpha=0.8, edgecolor='yellow'))

        elif case_type == "small_object":
            # 小目标：GT 用紫色标出
            for gt in case.ground_truths:
                x1, y1, x2, y2 = gt.bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=3, edgecolor='purple', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, "小目标", color='purple', fontsize=10,
                        fontproperties=chinese_font,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple'))

        ax.imshow(frame_rgb)
        ax.axis('off')

        # 子图标题和说明
        title = f"{case_titles[case_type]}\n{case.sequence} 帧{case.frame_id}"
        ax.set_title(title, fontsize=12, fontproperties=chinese_font)

        # 底部说明
        ax.text(
            0.5, -0.05, case.description,
            ha='center', va='top', fontsize=10,
            fontproperties=chinese_font,
            transform=ax.transAxes
        )

    # 整体标题
    fig.suptitle("图5-4 检测失败案例图", fontsize=16, fontproperties=chinese_font, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  已保存: {output_path}")


# ══════════════════════════════════════════════════════════════
# 表格保存
# ══════════════════════════════════════════════════════════════

def save_detection_tables(
    metrics: Dict[str, Any],
    efficiency: Dict[str, Any],
    output_dir: Path,
) -> None:
    """保存表5-5和表5-6"""
    logger.info("\n保存表格...")

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 表5-5：检测定量结果
    table_5_5_data = {
        "指标": ["Precision", "Recall", "F1", "AP50"],
        "数值": [
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{metrics['ap50']:.3f}",
        ],
        "含义": [
            "检测为车辆的结果中有多少是真的",
            "真实车辆中有多少被检测出来",
            "Precision 与 Recall 的综合指标",
            "IoU=0.5 条件下的平均精度",
        ],
    }

    # 保存 CSV
    import csv
    csv_path = tables_dir / "table_5_5_detection_metrics.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "数值", "含义"])
        for i in range(len(table_5_5_data["指标"])):
            writer.writerow([
                table_5_5_data["指标"][i],
                table_5_5_data["数值"][i],
                table_5_5_data["含义"][i],
            ])
    logger.info(f"  已保存: {csv_path}")

    # 保存 Markdown
    md_path = tables_dir / "table_5_5_detection_metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 表5-5 目标检测定量结果表\n\n")
        f.write("| 指标 | 数值 | 含义 |\n")
        f.write("|------|------|------|\n")
        for i in range(len(table_5_5_data["指标"])):
            f.write(f"| {table_5_5_data['指标'][i]} | {table_5_5_data['数值'][i]} | {table_5_5_data['含义'][i]} |\n")
    logger.info(f"  已保存: {md_path}")

    # 表5-6：检测运行效率
    table_5_6_data = {
        "指标": [
            "测试数据集",
            "测试序列数",
            "测试帧数",
            "检测目标总数",
            "平均每帧目标数",
            "平均推理时间/ms",
            "平均 FPS",
            "运行设备",
            "检测模型",
        ],
        "数值": [
            efficiency.get("dataset", "UA-DETRAC"),
            str(efficiency.get("sequence_count", 0)),
            str(efficiency.get("total_frames", 0)),
            str(efficiency.get("total_detections", 0)),
            f"{efficiency.get('avg_detections_per_frame', 0):.2f}",
            f"{efficiency.get('avg_inference_ms', 0):.2f}",
            f"{efficiency.get('avg_fps', 0):.2f}",
            efficiency.get("device", "CPU"),
            efficiency.get("model_name", "YOLOv8n"),
        ],
        "含义": [
            "使用的数据集",
            "参与测试的序列数量",
            "实际参与检测的帧数",
            "全部帧检测到的车辆目标数量",
            "检测目标总数 / 测试帧数",
            "单帧模型推理平均耗时",
            "1000 / 平均推理时间",
            "CPU 或 GPU",
            "使用的检测模型",
        ],
    }

    # 保存 CSV
    csv_path = tables_dir / "table_5_6_detection_efficiency.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "数值", "含义"])
        for i in range(len(table_5_6_data["指标"])):
            writer.writerow([
                table_5_6_data["指标"][i],
                table_5_6_data["数值"][i],
                table_5_6_data["含义"][i],
            ])
    logger.info(f"  已保存: {csv_path}")

    # 保存 Markdown
    md_path = tables_dir / "table_5_6_detection_efficiency.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 表5-6 检测运行效率表\n\n")
        f.write("| 指标 | 数值 | 含义 |\n")
        f.write("|------|------|------|\n")
        for i in range(len(table_5_6_data["指标"])):
            f.write(f"| {table_5_6_data['指标'][i]} | {table_5_6_data['数值'][i]} | {table_5_6_data['含义'][i]} |\n")
    logger.info(f"  已保存: {md_path}")



def save_raw_outputs(
    results: List[FrameDetectionResult],
    metrics: Dict[str, Any],
    efficiency: Dict[str, Any],
    failure_cases: Dict[str, Optional[FailureCase]],
    output_dir: Path,
) -> None:
    """保存原始 JSON 数据"""
    logger.info("\n保存原始数据...")

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 保存检测结果
    detections_data = []
    for result in results:
        detections_data.append({
            "sequence": result.sequence,
            "frame_id": result.frame_id,
            "image_path": str(result.image_path),
            "num_detections": len(result.detections),
            "num_ground_truths": len(result.ground_truths),
            "inference_time_ms": result.inference_time_ms,
            "tp": result.tp,
            "fp": result.fp,
            "fn": result.fn,
        })

    detections_path = raw_dir / "uadetrac_detections.json"
    with open(detections_path, "w", encoding="utf-8") as f:
        json.dump(detections_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  已保存: {detections_path}")

    # 保存评估指标
    eval_path = raw_dir / "uadetrac_detection_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"  已保存: {eval_path}")

    # 保存效率指标
    efficiency_path = raw_dir / "uadetrac_efficiency.json"
    with open(efficiency_path, "w", encoding="utf-8") as f:
        json.dump(efficiency, f, ensure_ascii=False, indent=2)
    logger.info(f"  已保存: {efficiency_path}")

    # 保存失败案例
    failure_cases_data = {}
    for case_type, case in failure_cases.items():
        if case:
            failure_cases_data[case_type] = {
                "sequence": case.sequence,
                "frame_id": case.frame_id,
                "image_path": str(case.image_path),
                "description": case.description,
                "num_detections": len(case.detections),
                "num_ground_truths": len(case.ground_truths),
            }
        else:
            failure_cases_data[case_type] = None

    failure_path = raw_dir / "uadetrac_failure_cases.json"
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_cases_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  已保存: {failure_path}")


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="生成论文第 5 章目标检测实验资产",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 数据集参数
    parser.add_argument("--config", type=str, help="配置文件路径（YAML）")
    parser.add_argument("--uadetrac-root", type=str, help="UA-DETRAC 数据集根目录")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "subset"],
                        help="数据划分（默认：train）")
    parser.add_argument("--sequence", type=str, help="指定单个序列名称（可选）")

    # 输出参数
    parser.add_argument("--output", type=str, default="outputs/chapter5/detection",
                        help="输出目录（默认：outputs/chapter5/detection）")

    # 检测器参数
    parser.add_argument("--model", type=str, default="weights/yolov8n.pt",
                        help="检测模型路径（默认：weights/yolov8n.pt）")
    parser.add_argument("--detector-backend", type=str, default="auto",
                        choices=["ultralytics", "onnx", "auto"],
                        help="检测后端（默认：auto）")
    parser.add_argument("--conf-thres", type=float, default=0.25,
                        help="检测置信度阈值（默认：0.25）")
    parser.add_argument("--iou-thres", type=float, default=0.50,
                        help="IoU 匹配阈值（默认：0.50）")

    # 处理参数
    parser.add_argument("--max-frames", type=int, default=500,
                        help="最多处理帧数（默认：500）")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="抽帧间隔（默认：1）")
    parser.add_argument("--vehicle-only", action="store_true", default=True,
                        help="只统计车辆类（默认：开启）")
    parser.add_argument("--device", type=str, default="cpu",
                        help="运行设备（默认：cpu）")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("论文第 5 章目标检测实验资产生成")
    logger.info("=" * 60)

    # 确定数据集路径
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            sys.exit(1)

        cfg = load_yaml(config_path)
        uadetrac_root = ROOT / cfg.get("dataset_root", "data/UA-DETRAC/")
        sequence_list = cfg.get("sequence_list", [])
        max_frames_per_seq = cfg.get("max_frames_per_sequence", args.max_frames)
        split = cfg.get("split", args.split)

        logger.info(f"从配置文件加载: {config_path}")
        logger.info(f"  数据集根目录: {uadetrac_root}")
        logger.info(f"  序列列表: {sequence_list}")

    elif args.uadetrac_root:
        uadetrac_root = Path(args.uadetrac_root)
        sequence_list = [args.sequence] if args.sequence else None
        max_frames_per_seq = args.max_frames
        split = args.split

        logger.info(f"使用指定路径: {uadetrac_root}")

    else:
        logger.error("必须指定 --config 或 --uadetrac-root")
        sys.exit(1)

    # 检查数据集路径
    if not uadetrac_root.exists():
        logger.error(f"UA-DETRAC 数据集不存在: {uadetrac_root}")
        logger.error("请从 https://detrac-db.rit.albany.edu/ 下载数据集")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载序列
    try:
        sequences = load_uadetrac_sequences(
            uadetrac_root=uadetrac_root,
            split=split,
            sequence_list=sequence_list,
            max_frames_per_seq=max_frames_per_seq,
        )
    except Exception as e:
        logger.error(f"加载序列失败: {e}")
        sys.exit(1)

    if not sequences:
        logger.error("未找到可用序列")
        sys.exit(1)

    # 加载检测器
    try:
        detector = load_detector(
            model_path=args.model,
            backend=args.detector_backend,
            conf_thres=args.conf_thres,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"加载检测器失败: {e}")
        sys.exit(1)

    # 运行检测
    try:
        results = run_detection_on_uadetrac(
            sequences=sequences,
            detector=detector,
            iou_thres=args.iou_thres,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            vehicle_only=args.vehicle_only,
        )
    except Exception as e:
        logger.error(f"检测执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not results:
        logger.error("未生成任何检测结果")
        sys.exit(1)

    # 计算指标
    metrics = compute_detection_metrics(results, iou_thres=args.iou_thres)
    model_name = Path(args.model).stem
    efficiency = compute_efficiency_metrics(results, model_name=model_name, device=args.device)

    # 选择可视化帧
    selected_frames = select_visualization_frames(results, num_frames=4)

    # 选择失败案例
    failure_cases = select_failure_cases(results, iou_thres=args.iou_thres)

    # 生成图表
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    draw_detection_visualization(
        selected_frames=selected_frames,
        output_path=figures_dir / "fig_5_3_detection_visualization.png",
        selected_frames_dir=figures_dir / "selected_frames",
    )

    draw_failure_cases(
        cases=failure_cases,
        output_path=figures_dir / "fig_5_4_detection_failure_cases.png",
    )

    # 保存表格
    save_detection_tables(metrics, efficiency, output_dir)

    # 保存原始数据
    save_raw_outputs(results, metrics, efficiency, failure_cases, output_dir)

    # 输出摘要
    logger.info("\n" + "=" * 60)
    logger.info("生成完成")
    logger.info("=" * 60)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"\n图表:")
    logger.info(f"  - 图5-3: {figures_dir / 'fig_5_3_detection_visualization.png'}")
    logger.info(f"  - 图5-4: {figures_dir / 'fig_5_4_detection_failure_cases.png'}")
    logger.info(f"\n表格:")
    logger.info(f"  - 表5-5: {output_dir / 'tables' / 'table_5_5_detection_metrics.csv'}")
    logger.info(f"  - 表5-6: {output_dir / 'tables' / 'table_5_6_detection_efficiency.csv'}")
    logger.info(f"\n指标摘要:")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall:    {metrics['recall']:.3f}")
    logger.info(f"  F1:        {metrics['f1']:.3f}")
    logger.info(f"  AP50:      {metrics['ap50']:.3f}")
    logger.info(f"\n效率摘要:")
    logger.info(f"  测试帧数: {efficiency['total_frames']}")
    logger.info(f"  检测目标总数: {efficiency['total_detections']}")
    logger.info(f"  平均推理时间: {efficiency['avg_inference_ms']:.2f} ms")
    logger.info(f"  平均 FPS: {efficiency['avg_fps']:.2f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

