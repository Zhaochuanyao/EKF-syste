"""
运行演示 - 快速演示跟踪效果
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.main import run_tracking


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EKF 跟踪演示")
    parser.add_argument("--config", default="configs/exp/demo_vehicle_accuracy.yaml")
    parser.add_argument("--video", default=None, help="视频路径（覆盖配置）")
    parser.add_argument("--output", default="outputs/demo/")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("EKF 多目标跟踪系统 - 演示模式")
    print("=" * 60)

    stats = run_tracking(
        config_path=args.config,
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
        show=args.show,
    )

    if stats:
        print(f"\n演示完成！")
        print(f"处理帧数: {stats.get('frames_processed', 0)}")
        print(f"平均 FPS: {stats.get('fps', 0):.1f}")
        print(f"输出目录: {stats.get('output_dir', '')}")


if __name__ == "__main__":
    main()
