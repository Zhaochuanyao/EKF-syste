"""
标注格式转换工具
支持：
  UA-DETRAC XML  → 内部统一 JSON 格式
  MOT17 txt      → 内部统一 JSON 格式

内部统一 JSON 格式：
{
  "dataset": "ua_detrac" | "mot17",
  "sequence": "<序列名>",
  "fps": 25,
  "frames": [
    {
      "frame_id": 0,
      "annotations": [
        {
          "id": 1,
          "bbox": [x1, y1, x2, y2],    // 像素坐标，左上+右下
          "class_id": 0,
          "class_name": "car"
        }
      ]
    }
  ]
}

用法示例：
  # UA-DETRAC 单序列 XML → JSON
  python scripts/convert_annotations.py \\
      --format ua_detrac \\
      --input data/UA-DETRAC/DETRAC-Train-Annotations-XML/MVI_20011.xml \\
      --output data/processed/ua_detrac/

  # UA-DETRAC 整个标注目录批量转换
  python scripts/convert_annotations.py \\
      --format ua_detrac \\
      --input data/UA-DETRAC/DETRAC-Train-Annotations-XML/ \\
      --output data/processed/ua_detrac/

  # MOT17 单序列目录 → JSON
  python scripts/convert_annotations.py \\
      --format mot17 \\
      --input data/MOT17/train/MOT17-02-FRCNN \\
      --output data/processed/mot17/

  # MOT17 整个 train/ 目录批量转换
  python scripts/convert_annotations.py \\
      --format mot17 \\
      --input data/MOT17/train/ \\
      --output data/processed/mot17/
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_annotations")

# ── UA-DETRAC 类别映射 ────────────────────────────────────────
UA_DETRAC_CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "van": 2,
    "others": 3,
}

# ── MOT17 类别映射（类别 1=行人，其余忽略）──────────────────────
MOT17_CLASS_MAP = {
    1: (0, "pedestrian"),
    2: (1, "person_on_vehicle"),
    7: (2, "static"),
    8: (3, "distractor"),
    12: (4, "reflection"),
}
MOT17_DEFAULT_CLASS = (0, "pedestrian")


# ══════════════════════════════════════════════════════════════
# UA-DETRAC 转换
# ══════════════════════════════════════════════════════════════

def _parse_ua_detrac_xml(xml_path: Path) -> Dict[str, Any]:
    """
    解析单个 UA-DETRAC XML 文件，返回内部统一格式字典。

    UA-DETRAC XML 结构：
      <sequence name="MVI_20011">
        <frame density="7" num="0">
          <target_list>
            <target id="1">
              <box left="592" top="378" width="88" height="63" />
              <attribute vehicle_type="car" ... />
            </target>
          </target_list>
        </frame>
      </sequence>
    """
    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        raise ImportError("xml.etree.ElementTree 是 Python 标准库，不应缺失")

    logger.info(f"解析 UA-DETRAC XML: {xml_path.name}")

    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        raise ValueError(f"XML 解析失败 {xml_path}: {e}") from e

    root = tree.getroot()
    sequence_name = root.get("name", xml_path.stem)

    frames_data = []
    frame_elements = root.findall("frame")

    if not frame_elements:
        logger.warning(f"  {xml_path.name}: 未找到 <frame> 元素，输出空序列")

    for frame_el in frame_elements:
        frame_num_str = frame_el.get("num", "0")
        try:
            frame_id = int(frame_num_str)
        except ValueError:
            logger.warning(f"  无效 frame num='{frame_num_str}'，跳过")
            continue

        annotations = []
        target_list_el = frame_el.find("target_list")
        if target_list_el is None:
            frames_data.append({"frame_id": frame_id, "annotations": []})
            continue

        for target_el in target_list_el.findall("target"):
            target_id_str = target_el.get("id", "0")
            try:
                target_id = int(target_id_str)
            except ValueError:
                target_id = 0

            # 解析边界框
            box_el = target_el.find("box")
            if box_el is None:
                continue
            try:
                left = float(box_el.get("left", 0))
                top = float(box_el.get("top", 0))
                w = float(box_el.get("width", 0))
                h = float(box_el.get("height", 0))
            except (TypeError, ValueError) as e:
                logger.warning(f"  目标 {target_id} 边界框解析失败: {e}，跳过")
                continue

            x1, y1, x2, y2 = left, top, left + w, top + h

            # 解析类别
            attr_el = target_el.find("attribute")
            vehicle_type = "car"  # 默认
            if attr_el is not None:
                vehicle_type = attr_el.get("vehicle_type", "car").lower()

            class_id = UA_DETRAC_CLASS_MAP.get(vehicle_type, 3)
            class_name = vehicle_type

            annotations.append({
                "id": target_id,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "class_id": class_id,
                "class_name": class_name,
            })

        frames_data.append({"frame_id": frame_id, "annotations": annotations})

    # 按 frame_id 排序
    frames_data.sort(key=lambda f: f["frame_id"])

    return {
        "dataset": "ua_detrac",
        "sequence": sequence_name,
        "fps": 25,
        "total_frames": len(frames_data),
        "frames": frames_data,
    }


def convert_ua_detrac(input_path: str, output_dir: str) -> List[Path]:
    """
    转换 UA-DETRAC 标注：
      - 若 input_path 是 .xml 文件：转换单个序列
      - 若 input_path 是目录：批量转换目录下所有 .xml

    Returns:
        生成的 JSON 文件路径列表
    """
    input_p = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files: List[Path] = []
    if input_p.is_file():
        if input_p.suffix.lower() != ".xml":
            raise ValueError(f"输入文件不是 XML：{input_p}")
        xml_files = [input_p]
    elif input_p.is_dir():
        xml_files = sorted(input_p.glob("*.xml"))
        if not xml_files:
            logger.warning(f"目录中未找到 .xml 文件：{input_p}")
            return []
    else:
        raise FileNotFoundError(f"输入路径不存在：{input_p}")

    generated = []
    errors = []
    for xml_file in xml_files:
        try:
            data = _parse_ua_detrac_xml(xml_file)
            out_file = out_dir / f"{xml_file.stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            total_ann = sum(len(fr["annotations"]) for fr in data["frames"])
            logger.info(
                f"  ✓ {xml_file.stem}: {data['total_frames']} 帧, "
                f"{total_ann} 标注 → {out_file.name}"
            )
            generated.append(out_file)
        except Exception as e:
            logger.error(f"  ✗ {xml_file.name} 转换失败: {e}")
            errors.append(xml_file.name)

    logger.info(
        f"UA-DETRAC 转换完成：{len(generated)} 成功, {len(errors)} 失败"
        + (f"  失败: {errors}" if errors else "")
    )
    return generated


# ══════════════════════════════════════════════════════════════
# MOT17 转换
# ══════════════════════════════════════════════════════════════

def _parse_mot17_sequence(seq_dir: Path) -> Dict[str, Any]:
    """
    解析单个 MOT17 序列目录。

    MOT17 目录结构：
      MOT17-02-FRCNN/
        img1/             ← 图像帧（.jpg）
        gt/gt.txt         ← ground truth 标注
        seqinfo.ini       ← 序列元信息

    gt.txt 格式（逗号分隔）：
      frame_id, track_id, x, y, w, h, conf, class, visibility
      - conf=0: 忽略区域（ignore region）
      - class=1: 行人
    """
    gt_file = seq_dir / "gt" / "gt.txt"
    seqinfo_file = seq_dir / "seqinfo.ini"

    if not gt_file.exists():
        raise FileNotFoundError(f"gt.txt 不存在：{gt_file}")

    # 读取序列 FPS
    fps = 25  # 默认值
    if seqinfo_file.exists():
        for line in seqinfo_file.read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("framerate"):
                try:
                    fps = int(line.split("=")[-1].strip())
                except ValueError:
                    pass

    logger.info(f"解析 MOT17 序列: {seq_dir.name} (fps={fps})")

    # 读取 gt.txt，按 frame_id 聚合
    frame_dict: Dict[int, List[Dict]] = {}
    skipped = 0

    for line_no, raw_line in enumerate(
        gt_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            logger.warning(f"  第 {line_no} 行列数不足，跳过: {line[:60]}")
            continue

        try:
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = int(parts[6]) if len(parts) > 6 else 1
            cls = int(parts[7]) if len(parts) > 7 else 1
        except (ValueError, IndexError) as e:
            logger.warning(f"  第 {line_no} 行解析失败: {e}，跳过")
            continue

        # conf=0 表示忽略区域（遮挡/标注不全），跳过
        if conf == 0:
            skipped += 1
            continue

        class_id, class_name = MOT17_CLASS_MAP.get(cls, MOT17_DEFAULT_CLASS)
        x1, y1, x2, y2 = x, y, x + w, y + h

        if frame_id not in frame_dict:
            frame_dict[frame_id] = []
        frame_dict[frame_id].append({
            "id": track_id,
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "class_id": class_id,
            "class_name": class_name,
        })

    frames_data = [
        {"frame_id": fid, "annotations": anns}
        for fid, anns in sorted(frame_dict.items())
    ]

    total_ann = sum(len(f["annotations"]) for f in frames_data)
    logger.info(
        f"  {seq_dir.name}: {len(frames_data)} 帧, "
        f"{total_ann} 标注 (跳过 ignore={skipped})"
    )

    return {
        "dataset": "mot17",
        "sequence": seq_dir.name,
        "fps": fps,
        "total_frames": len(frames_data),
        "frames": frames_data,
    }


def convert_mot17(input_path: str, output_dir: str) -> List[Path]:
    """
    转换 MOT17 标注：
      - 若 input_path 是单个序列目录（含 gt/gt.txt）：转换单序列
      - 若 input_path 是 train/ 或 test/ 目录：批量转换所有子序列

    Returns:
        生成的 JSON 文件路径列表
    """
    input_p = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_p.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_p}")

    # 判断是单序列还是集合目录
    if (input_p / "gt" / "gt.txt").exists():
        seq_dirs = [input_p]
    else:
        # 取直接子目录中包含 gt/gt.txt 的
        seq_dirs = sorted(
            d for d in input_p.iterdir()
            if d.is_dir() and (d / "gt" / "gt.txt").exists()
        )
        if not seq_dirs:
            logger.warning(f"目录下未找到有效 MOT17 序列（无 gt/gt.txt）：{input_p}")
            return []

    generated = []
    errors = []
    for seq_dir in seq_dirs:
        try:
            data = _parse_mot17_sequence(seq_dir)
            out_file = out_dir / f"{seq_dir.name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"  ✓ {seq_dir.name} → {out_file.name}")
            generated.append(out_file)
        except Exception as e:
            logger.error(f"  ✗ {seq_dir.name} 转换失败: {e}")
            errors.append(seq_dir.name)

    logger.info(
        f"MOT17 转换完成：{len(generated)} 成功, {len(errors)} 失败"
        + (f"  失败: {errors}" if errors else "")
    )
    return generated


# ══════════════════════════════════════════════════════════════
# 格式验证工具
# ══════════════════════════════════════════════════════════════

def validate_json(json_path: Path, max_show: int = 3) -> bool:
    """验证生成的 JSON 文件格式是否符合内部规范"""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"JSON 读取失败 {json_path}: {e}")
        return False

    required_keys = {"dataset", "sequence", "fps", "frames"}
    missing = required_keys - set(data.keys())
    if missing:
        logger.error(f"缺少必要字段：{missing}")
        return False

    frames = data["frames"]
    shown = 0
    for frame in frames:
        if "frame_id" not in frame or "annotations" not in frame:
            logger.error(f"帧缺少 frame_id 或 annotations 字段")
            return False
        for ann in frame["annotations"]:
            if "bbox" not in ann or len(ann["bbox"]) != 4:
                logger.error(f"标注缺少有效 bbox 字段: {ann}")
                return False
        if shown < max_show and frame["annotations"]:
            logger.info(f"  帧 {frame['frame_id']}: {frame['annotations'][0]}")
            shown += 1

    logger.info(f"✓ 格式验证通过: {json_path.name}")
    return True


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="标注格式转换工具（UA-DETRAC / MOT17 → 内部 JSON）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--format",
        choices=["ua_detrac", "mot17"],
        required=True,
        help="输入数据集格式",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入路径（.xml 文件 或 目录）",
    )
    parser.add_argument(
        "--output",
        default="data/processed/",
        help="输出目录（默认：data/processed/）",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="转换完成后对生成的 JSON 文件进行格式验证",
    )
    args = parser.parse_args()

    logger.info(f"=== 标注转换: format={args.format}, input={args.input} ===")

    try:
        if args.format == "ua_detrac":
            generated = convert_ua_detrac(args.input, args.output)
        else:
            generated = convert_mot17(args.input, args.output)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"转换失败: {e}")
        sys.exit(1)

    if args.validate and generated:
        logger.info("=== 格式验证 ===")
        all_ok = all(validate_json(p) for p in generated)
        if not all_ok:
            logger.error("部分文件格式验证失败")
            sys.exit(1)

    logger.info(f"=== 完成，共生成 {len(generated)} 个 JSON 文件 ===")


if __name__ == "__main__":
    main()
