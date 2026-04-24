"""
UA-DETRAC 小子集数据检查脚本

功能
----
1. 检查 dataset_root 是否存在
2. 检查图像目录 / 标注目录是否存在
3. 检查 cpu_small 子集中每个序列是否存在
4. 统计可用序列的帧数与标注框数
5. 输出本地数据集摘要报告

用法
----
  python scripts/check_uadetrac_subset.py
  python scripts/check_uadetrac_subset.py --subset-config configs/data/uadetrac_subset_cpu.yaml
  python scripts/check_uadetrac_subset.py --mode full
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_uadetrac_subset")


# ──────────────────────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        logger.error("缺少 pyyaml，请运行: pip install pyyaml")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ──────────────────────────────────────────────────────────────
# 图像序列帧数统计
# ──────────────────────────────────────────────────────────────

def _count_image_frames(seq_dir: Path) -> int:
    """统计序列图像目录中 .jpg 帧数，目录不存在返回 -1"""
    if not seq_dir.is_dir():
        return -1
    return len(list(seq_dir.glob("*.jpg")))


def _count_anno_boxes(xml_path: Path) -> int:
    """统计 XML 标注文件中的框数，失败返回 -1"""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return len(root.findall(".//box"))
    except Exception:
        return -1


# ──────────────────────────────────────────────────────────────
# 单序列检查
# ──────────────────────────────────────────────────────────────

def check_sequence(
    seq_name: str,
    image_dir: Path,
    anno_dir: Path,
    max_frames: Optional[int],
) -> Dict:
    """检查单个序列，返回状态字典"""
    result = {
        "seq": seq_name,
        "image_found": False,
        "anno_found": False,
        "image_frames": 0,
        "effective_frames": 0,
        "anno_boxes": 0,
        "status": "missing",
        "notes": [],
    }

    # 查找图像序列目录（image_dir/seq_name/img*.jpg）
    seq_dir = image_dir / seq_name
    n_frames = _count_image_frames(seq_dir)
    if n_frames >= 0:
        result["image_found"] = True
        result["image_frames"] = n_frames
        eff = n_frames
        if max_frames and eff > max_frames:
            eff = max_frames
        result["effective_frames"] = eff
    else:
        result["notes"].append(f"图像目录未找到（已在 {image_dir} 中搜索）")

    # 查找标注 XML
    anno_path = anno_dir / f"{seq_name}_v3.xml"
    if not anno_path.exists():
        anno_path = anno_dir / f"{seq_name}.xml"
    if anno_path.exists():
        result["anno_found"] = True
        boxes = _count_anno_boxes(anno_path)
        result["anno_boxes"] = boxes if boxes >= 0 else 0
    else:
        result["notes"].append(f"标注 XML 未找到（已在 {anno_dir} 中搜索）")

    # 综合状态
    if result["image_found"] and result["anno_found"]:
        result["status"] = "ok"
    elif result["image_found"] and not result["anno_found"]:
        result["status"] = "no_anno"
    elif not result["image_found"] and result["anno_found"]:
        result["status"] = "no_video"
    else:
        result["status"] = "missing"

    return result


# ──────────────────────────────────────────────────────────────
# 全量序列扫描（full 模式）
# ──────────────────────────────────────────────────────────────

def scan_all_sequences(image_dir: Path) -> List[str]:
    """扫描 image_dir 下所有图像序列目录（MVI_* 格式）"""
    seqs = []
    for p in sorted(image_dir.iterdir()):
        if p.is_dir() and p.name.startswith("MVI_"):
            seqs.append(p.name)
    return seqs


# ──────────────────────────────────────────────────────────────
# 主检查逻辑
# ──────────────────────────────────────────────────────────────

def run_check(subset_config_path: Path, mode: str) -> bool:
    """
    执行完整检查，返回 True 表示子集可用，False 表示存在严重缺失。
    """
    logger.info("=" * 60)
    logger.info("UA-DETRAC 数据集检查")
    logger.info("=" * 60)

    if not subset_config_path.exists():
        logger.error(f"子集配置文件不存在: {subset_config_path}")
        return False

    cfg = load_yaml(subset_config_path)
    dataset_root = ROOT / cfg.get("dataset_root", "data/UA-DETRAC/")
    image_dir = ROOT / cfg.get("image_dir", "data/UA-DETRAC/DETRAC-Images/DETRAC-Images/")
    anno_dir  = ROOT / cfg.get("anno_dir",  "data/UA-DETRAC/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/")
    max_frames = cfg.get("max_frames_per_sequence", 1200)
    subset_seqs: List[str] = cfg.get("sequence_list", [])
    max_total = cfg.get("max_total_sequences", 8)
    allow_missing_gt = cfg.get("allow_missing_gt", False)

    # ── 目录存在性检查 ──────────────────────────────────────
    logger.info(f"\n[1/4] 根目录检查")
    logger.info(f"  dataset_root : {dataset_root}")
    logger.info(f"  image_dir    : {image_dir}")
    logger.info(f"  anno_dir     : {anno_dir}")

    root_ok  = dataset_root.exists()
    image_ok = image_dir.exists()
    anno_ok  = anno_dir.exists()

    logger.info(f"  dataset_root 存在: {'是' if root_ok else '否 ⚠'}")
    logger.info(f"  image_dir    存在: {'是' if image_ok else '否 ⚠'}")
    logger.info(f"  anno_dir     存在: {'是' if anno_ok else '否 ⚠'}")

    if not root_ok:
        logger.error(
            "\n数据集根目录不存在！\n"
            "请从 https://detrac-db.rit.albany.edu/ 下载 UA-DETRAC 数据集，\n"
            f"并解压到: {dataset_root}\n"
            "目录结构应为:\n"
            "  data/UA-DETRAC/\n"
            "    DETRAC-Images/DETRAC-Images/  (图像序列目录)\n"
            "    DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/  (标注目录)"
        )
        return False

    # ── 序列列表确定 ────────────────────────────────────────
    logger.info(f"\n[2/4] 序列列表确定（mode={mode}）")
    if mode == "full":
        if image_ok:
            all_seqs = scan_all_sequences(image_dir)
            logger.info(f"  full 模式：扫描到 {len(all_seqs)} 个序列目录")
            check_seqs = all_seqs
        else:
            logger.error("  image_dir 不存在，无法扫描序列")
            return False
    else:
        # cpu_small 模式
        check_seqs = subset_seqs[:max_total]
        logger.info(f"  cpu_small 模式：配置了 {len(subset_seqs)} 个序列，取前 {max_total} 个")
        logger.info(f"  序列列表: {check_seqs}")

    # ── 逐序列检查 ──────────────────────────────────────────
    logger.info(f"\n[3/4] 逐序列检查（共 {len(check_seqs)} 个）")
    results = []
    for seq in check_seqs:
        r = check_sequence(seq, image_dir, anno_dir, max_frames)
        results.append(r)
        status_icon = {"ok": "✓", "no_anno": "△", "no_video": "△", "missing": "✗"}.get(r["status"], "?")
        notes_str = "; ".join(r["notes"]) if r["notes"] else ""
        logger.info(
            f"  {status_icon} {seq:<15} "
            f"图像:{r['image_frames']:>5}帧→{r['effective_frames']:>4}帧  "
            f"标注:{r['anno_boxes']:>6}框  "
            f"状态:{r['status']}"
            + (f"  [{notes_str}]" if notes_str else "")
        )

    # ── 汇总 ────────────────────────────────────────────────
    logger.info(f"\n[4/4] 汇总")
    ok_seqs = [r for r in results if r["status"] == "ok"]
    no_anno_seqs = [r for r in results if r["status"] == "no_anno"]
    missing_seqs = [r for r in results if r["status"] in ("missing", "no_video")]

    total_effective_frames = sum(r["effective_frames"] for r in ok_seqs)
    total_anno_boxes       = sum(r["anno_boxes"]       for r in ok_seqs)

    logger.info(f"  完整可用序列  : {len(ok_seqs)} / {len(check_seqs)}")
    logger.info(f"  仅有视频无标注: {len(no_anno_seqs)}")
    logger.info(f"  完全缺失序列  : {len(missing_seqs)}")
    logger.info(f"  有效帧数合计  : {total_effective_frames:,}")
    logger.info(f"  标注框数合计  : {total_anno_boxes:,}")

    # 数据充足性判断
    min_ok = 4  # 至少需要 4 个完整序列才能做消融实验
    is_sufficient = len(ok_seqs) >= min_ok

    if is_sufficient:
        logger.info(f"\n  数据集检查通过 — 可用序列 {len(ok_seqs)} 个，满足实验要求")
    else:
        logger.warning(
            f"\n  可用序列不足（{len(ok_seqs)} < {min_ok}）\n"
            "  实验可能无法运行。请下载更多 UA-DETRAC 序列数据。"
        )

    if missing_seqs:
        logger.warning(
            f"\n  以下序列在本地缺失（跳过）：\n"
            + "\n".join(f"    - {r['seq']}" for r in missing_seqs)
        )

    if not allow_missing_gt and no_anno_seqs:
        logger.warning(
            f"\n  以下序列缺少标注（allow_missing_gt=false，将被排除）：\n"
            + "\n".join(f"    - {r['seq']}" for r in no_anno_seqs)
        )

    logger.info("\n" + "=" * 60)
    logger.info("如需下载 UA-DETRAC 数据集，请访问:")
    logger.info("  https://detrac-db.rit.albany.edu/")
    logger.info("下载后解压到 data/UA-DETRAC/ 目录")
    logger.info("=" * 60)

    return is_sufficient


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="检查 UA-DETRAC 小子集数据是否就绪",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subset-config",
        default="configs/data/uadetrac_subset_cpu.yaml",
        help="子集配置 YAML 路径",
    )
    parser.add_argument(
        "--mode",
        default="cpu_small",
        choices=["cpu_small", "full"],
        help="检查模式：cpu_small 只检查配置列表；full 扫描全部序列",
    )
    args = parser.parse_args()

    config_path = ROOT / args.subset_config
    ok = run_check(config_path, args.mode)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
