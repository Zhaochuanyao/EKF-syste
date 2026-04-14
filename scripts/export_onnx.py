"""
导出 ONNX 模型
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def export_onnx(weights: str = "weights/yolov8n.pt", output: str = "weights/yolov8n.onnx"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 请安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[ERROR] 权重文件不存在: {weights_path}")
        print("请先运行: python scripts/download_weights.py")
        sys.exit(1)

    print(f"正在导出 ONNX 模型: {weights} -> {output}")
    model = YOLO(str(weights_path))
    model.export(format="onnx", imgsz=640, simplify=True, opset=12)

    # ultralytics 默认导出到同目录
    default_out = weights_path.with_suffix(".onnx")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if default_out.exists() and default_out != out_path:
        import shutil
        shutil.move(str(default_out), str(out_path))

    print(f"[OK] ONNX 模型已保存: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/yolov8n.pt")
    parser.add_argument("--output", default="weights/yolov8n.onnx")
    args = parser.parse_args()
    export_onnx(args.weights, args.output)
