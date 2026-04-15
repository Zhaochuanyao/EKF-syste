"""
运行演示 - 车辆多目标跟踪快速演示

默认使用 demo_vehicle_accuracy 配置（轿车/卡车/巴士/摩托车专项优化），
也可通过 --config 切换到其他配置。

用法：
  # 车辆场景演示（默认）
  python scripts/run_demo.py --video assets/samples/demo.mp4

  # 行人场景演示
  python scripts/run_demo.py --config configs/exp/demo_person_accuracy.yaml \\
                              --video assets/samples/people.mp4
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.main import run_tracking

_VEHICLE_CONFIG = "configs/exp/demo_vehicle_accuracy.yaml"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="EKF 车辆多目标跟踪演示（默认：车辆稳定模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default=_VEHICLE_CONFIG,
        help=f"实验配置文件（默认: 车辆稳定模式）",
    )
    parser.add_argument("--video", default=None, help="视频路径（覆盖配置）")
    parser.add_argument("--output", default="outputs/demo/")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="实时显示窗口（需要 GUI）")
    args = parser.parse_args()

    cfg_name = Path(args.config).stem if args.config else "default"
    is_vehicle = "vehicle" in cfg_name

    print("=" * 60)
    print("EKF 车辆多目标跟踪系统 — 演示模式")
    print(f"  配置: {cfg_name}")
    if is_vehicle:
        print("  目标: 轿车(2) / 摩托车(3) / 巴士(5) / 卡车(7) [COCO]")
    print("=" * 60)

    if not args.video:
        print("[提示] 未指定 --video，将使用配置中的默认视频路径（若有）")

    stats = run_tracking(
        config_path=args.config,
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
        show=args.show,
    )

    if stats:
        print(f"\n演示完成！")
        print(f"  处理帧数: {stats.get('frames_processed', 0)}")
        print(f"  平均 FPS: {stats.get('fps', 0):.1f}")
        print(f"  输出目录: {stats.get('output_dir', '')}")


if __name__ == "__main__":
    main()
