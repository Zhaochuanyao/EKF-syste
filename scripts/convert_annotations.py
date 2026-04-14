"""
标注格式转换工具（预留入口）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def convert_ua_detrac(xml_dir: str, output_dir: str):
    """将 UA-DETRAC XML 标注转换为内部格式（TODO）"""
    print(f"[TODO] UA-DETRAC 转换: {xml_dir} -> {output_dir}")
    print("此功能待实现，请参考 src/ekf_mot/data/converters.py")


def convert_mot17(mot_dir: str, output_dir: str):
    """将 MOT17 标注转换为内部格式（TODO）"""
    print(f"[TODO] MOT17 转换: {mot_dir} -> {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["ua_detrac", "mot17"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.format == "ua_detrac":
        convert_ua_detrac(args.input, args.output)
    elif args.format == "mot17":
        convert_mot17(args.input, args.output)
