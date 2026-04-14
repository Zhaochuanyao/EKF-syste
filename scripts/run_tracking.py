"""
正式运行入口
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.main import run_tracking


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EKF 多目标跟踪")
    parser.add_argument("--config", required=True, help="实验配置文件路径")
    parser.add_argument("--video", default=None, help="输入视频路径")
    parser.add_argument("--output", default=None, help="输出目录")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

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
