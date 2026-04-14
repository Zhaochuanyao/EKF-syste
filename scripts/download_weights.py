"""
自动下载 YOLOv8n 权重
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def download_weights(weights_dir: str = "weights") -> None:
    weights_path = Path(weights_dir)
    weights_path.mkdir(parents=True, exist_ok=True)

    target = weights_path / "yolov8n.pt"
    if target.exists():
        print(f"[OK] 权重已存在: {target}")
        return

    print(f"正在下载 YOLOv8n 权重到 {target} ...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # ultralytics 自动下载到缓存
        # 复制到项目 weights 目录
        import shutil
        cache_path = Path.home() / ".cache" / "ultralytics" / "weights" / "yolov8n.pt"
        # ultralytics 下载后保存在当前目录
        local = Path("yolov8n.pt")
        if local.exists():
            shutil.move(str(local), str(target))
        else:
            # 尝试从 ultralytics 缓存复制
            import glob
            candidates = glob.glob(str(Path.home() / "**" / "yolov8n.pt"), recursive=True)
            if candidates:
                shutil.copy(candidates[0], str(target))
        print(f"[OK] 权重已保存: {target}")
    except ImportError:
        print("[ERROR] 请先安装 ultralytics: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        print("请手动下载: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        print(f"并保存到: {target}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="weights", help="权重保存目录")
    args = parser.parse_args()
    download_weights(args.dir)
