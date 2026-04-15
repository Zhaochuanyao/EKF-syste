"""
正式运行入口（默认：车辆场景）
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
        description="EKF 多目标跟踪（默认：车辆稳定模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  # 车辆场景（默认）\n"
            "  python scripts/run_tracking.py --video assets/samples/demo.mp4\n\n"
            "  # 行人场景\n"
            "  python scripts/run_tracking.py --config configs/exp/demo_person_accuracy.yaml "
            "--video assets/samples/people.mp4\n"
        ),
    )
    parser.add_argument(
        "--config",
        default=_VEHICLE_CONFIG,
        help=f"实验配置文件路径（默认: {_VEHICLE_CONFIG}）",
    )
    parser.add_argument("--video", default=None, help="输入视频路径")
    parser.add_argument("--output", default=None, help="输出目录")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[警告] 配置文件不存在: {cfg_path}，将使用内置默认配置")
    else:
        print(f"[配置] {cfg_path.name}")

    run_tracking(
        config_path=args.config,
        video_path=args.video,
        output_dir=args.output,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        show=args.show,
    )


if __name__ == "__main__":
    main()
