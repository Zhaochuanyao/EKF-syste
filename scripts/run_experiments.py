"""
批量实验脚本 - 多配置 × 多视频组合实验，自动汇总结果

功能
----
  1. 遍历指定目录下的所有 YAML 配置文件
  2. 对每个配置文件，处理指定目录下的所有视频
  3. 每个（配置, 视频）组合独立输出到子目录
  4. 自动从输出 JSON 提取关键指标并汇总
  5. 生成 experiment_summary.json 和 experiment_summary.csv

用法
----
  # 最简：用默认 demo 视频测试所有场景配置
  python scripts/run_experiments.py

  # 指定配置目录和视频目录
  python scripts/run_experiments.py \\
      --config-dir configs/exp/ \\
      --video-dir  assets/samples/ \\
      --output-dir outputs/experiments/

  # 只测试特定配置
  python scripts/run_experiments.py \\
      --configs configs/exp/scenario_uniform_motion.yaml \\
               configs/exp/scenario_turning_motion.yaml \\
      --video  assets/samples/demo.mp4

  # 限制每个实验最多处理帧数（快速验证）
  python scripts/run_experiments.py --max-frames 200

输出结构
--------
  outputs/experiments/
  ├── uniform_motion_demo/          # 配置名_视频名
  │   ├── tracks.json               # 逐帧跟踪结果
  │   ├── tracks.csv
  │   └── output_*.mp4              # 可视化视频
  ├── accelerated_motion_demo/
  │   └── ...
  ├── experiment_summary.json       # 所有实验汇总
  └── experiment_summary.csv        # 方便 Excel/Python 绘图的 CSV
"""

import sys
import json
import time
import argparse
import logging
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiments")


# ══════════════════════════════════════════════════════════════
# 指标提取工具
# ══════════════════════════════════════════════════════════════

def _load_tracks_json(tracks_path: Path) -> List[Dict]:
    """加载跟踪结果 JSON"""
    if not tracks_path.exists():
        return []
    with open(tracks_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]
    return []


def _extract_track_stats(frames: List[Dict]) -> Dict[str, Any]:
    """
    从逐帧跟踪 JSON 提取轨迹统计。

    Returns:
        {num_tracks, avg_track_length, max_tracks_per_frame,
         total_detections, num_frames_with_tracks}
    """
    if not frames:
        return {
            "num_tracks": 0,
            "avg_track_length": 0.0,
            "max_tracks_per_frame": 0,
            "total_detections": 0,
            "num_frames_with_tracks": 0,
        }

    # 统计每个 track_id 出现的帧数
    track_frames: Dict[int, int] = {}
    max_per_frame = 0
    total_dets = 0

    for frame in frames:
        tracks = frame.get("tracks", [])
        confirmed = [t for t in tracks if t.get("state_name", "") == "Confirmed"]
        if len(confirmed) > max_per_frame:
            max_per_frame = len(confirmed)
        for t in confirmed:
            tid = int(t.get("track_id", 0))
            track_frames[tid] = track_frames.get(tid, 0) + 1
        total_dets += frame.get("num_detections", 0)

    num_tracks = len(track_frames)
    avg_len = (
        sum(track_frames.values()) / num_tracks if num_tracks else 0.0
    )
    frames_with_tracks = sum(
        1 for f in frames
        if any(t.get("state_name", "") == "Confirmed" for t in f.get("tracks", []))
    )

    return {
        "num_tracks": num_tracks,
        "avg_track_length": round(avg_len, 2),
        "max_tracks_per_frame": max_per_frame,
        "total_detections": total_dets,
        "num_frames_with_tracks": frames_with_tracks,
    }


def _get_experiment_name_from_config(cfg_path: Path) -> str:
    """从配置文件读取 experiment.name（如有），否则使用文件名"""
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("experiment", {}).get("name", cfg_path.stem)
    except Exception:
        return cfg_path.stem


def _get_scenario_description(cfg_path: Path) -> str:
    """从配置文件读取场景说明"""
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("experiment", {}).get("description", "")
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════
# 单次实验运行
# ══════════════════════════════════════════════════════════════

def run_single_experiment(
    config_path: Path,
    video_path: Path,
    output_dir: Path,
    max_frames: Optional[int],
) -> Dict[str, Any]:
    """
    运行单次（配置, 视频）实验。

    Returns:
        实验结果字典，包含 fps / track_stats / 状态等
    """
    exp_name = _get_experiment_name_from_config(config_path)
    run_id = f"{exp_name}_{video_path.stem}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  运行: {run_id}")
    logger.info(f"    config: {config_path}")
    logger.info(f"    video:  {video_path}")
    logger.info(f"    output: {run_dir}")

    t_start = time.perf_counter()
    status = "success"
    error_msg = ""

    try:
        from src.ekf_mot.main import run_tracking

        stats = run_tracking(
            config_path=str(config_path),
            video_path=str(video_path),
            output_dir=str(run_dir),
            max_frames=max_frames,
        )

        elapsed = time.perf_counter() - t_start
        fps = stats.get("fps", 0.0)
        frames_processed = stats.get("frames_processed", 0)

        # 读取 tracks.json 获取轨迹统计
        tracks_path = run_dir / "tracks.json"
        frames_data = _load_tracks_json(tracks_path)
        track_stats = _extract_track_stats(frames_data)

    except Exception as e:
        elapsed = time.perf_counter() - t_start
        fps = 0.0
        frames_processed = 0
        track_stats = _extract_track_stats([])
        status = "error"
        error_msg = str(e)
        logger.error(f"  实验失败: {run_id} — {e}")

    return {
        "run_id": run_id,
        "config": str(config_path),
        "video": str(video_path),
        "scenario": _get_experiment_name_from_config(config_path),
        "scenario_description": _get_scenario_description(config_path),
        "output_dir": str(run_dir),
        "status": status,
        "error": error_msg,
        "frames_processed": frames_processed,
        "fps": round(fps, 2),
        "elapsed_sec": round(elapsed, 1),
        **track_stats,
    }


# ══════════════════════════════════════════════════════════════
# CSV 导出
# ══════════════════════════════════════════════════════════════

# CSV 列顺序（方便后续绘图）
_CSV_COLUMNS = [
    "run_id",
    "scenario",
    "video",
    "status",
    "frames_processed",
    "fps",
    "elapsed_sec",
    "num_tracks",
    "avg_track_length",
    "max_tracks_per_frame",
    "total_detections",
    "num_frames_with_tracks",
    "scenario_description",
    "config",
    "output_dir",
]


def save_summary_csv(results: List[Dict], csv_path: Path) -> None:
    """保存实验摘要 CSV（可直接在 Excel 或 pandas 中绘图）"""
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ══════════════════════════════════════════════════════════════
# 可选图表导出
# ══════════════════════════════════════════════════════════════

def try_plot_summary(results: List[Dict], output_dir: Path) -> bool:
    """
    尝试生成简单图表（需要 matplotlib）。
    如果 matplotlib 不可用则静默跳过。

    生成：
      fps_bar.png          — 各实验 FPS 柱状图
      track_length_bar.png — 各实验平均轨迹长度柱状图
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # 非交互模式
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib 未安装，跳过图表生成（pip install matplotlib 可启用）")
        return False

    success_results = [r for r in results if r["status"] == "success"]
    if not success_results:
        return False

    labels = [r["run_id"] for r in success_results]
    # 缩短标签
    short_labels = [
        lb.replace("_", "\n") if len(lb) > 20 else lb
        for lb in labels
    ]

    fig_w = max(8, len(labels) * 1.5)

    # ── FPS 柱状图 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    fps_vals = [r["fps"] for r in success_results]
    bars = ax.bar(range(len(labels)), fps_vals, color="#4C72B0", edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("FPS")
    ax.set_title("各实验帧率对比 (FPS)")
    for bar, val in zip(bars, fps_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fps_path = output_dir / "fps_bar.png"
    fig.savefig(fps_path, dpi=120)
    plt.close(fig)
    logger.info(f"  FPS 图表: {fps_path}")

    # ── 平均轨迹长度柱状图 ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    len_vals = [r["avg_track_length"] for r in success_results]
    bars = ax.bar(range(len(labels)), len_vals, color="#55A868", edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("平均轨迹长度（帧）")
    ax.set_title("各实验平均轨迹长度对比")
    for bar, val in zip(bars, len_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    len_path = output_dir / "track_length_bar.png"
    fig.savefig(len_path, dpi=120)
    plt.close(fig)
    logger.info(f"  轨迹长度图表: {len_path}")

    return True


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="批量实验：多配置 × 多视频，自动汇总",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 目录模式
    parser.add_argument(
        "--config-dir",
        default="configs/exp/",
        help="实验配置目录（遍历所有 *.yaml）",
    )
    parser.add_argument(
        "--video-dir",
        default="assets/samples/",
        help="视频目录（遍历所有 *.mp4 / *.avi）",
    )
    # 精确指定模式（覆盖 --config-dir / --video-dir）
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="直接指定配置文件列表（覆盖 --config-dir）",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="直接指定单个视频（覆盖 --video-dir）",
    )
    # 过滤：只运行包含关键字的配置文件
    parser.add_argument(
        "--filter",
        default=None,
        help='过滤配置文件名（如 "scenario" 只跑场景配置）',
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/",
        help="实验输出根目录",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="每个实验最多处理帧数（None=不限制）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="跳过图表生成",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 确定配置列表 ──────────────────────────────────────────
    if args.configs:
        config_paths = [Path(p) for p in args.configs]
    else:
        config_dir = Path(args.config_dir)
        config_paths = sorted(config_dir.glob("*.yaml"))
        if args.filter:
            config_paths = [p for p in config_paths if args.filter in p.name]

    config_paths = [p for p in config_paths if p.exists()]
    if not config_paths:
        logger.error("未找到任何配置文件，请检查 --config-dir 或 --configs 参数")
        sys.exit(1)

    # ── 确定视频列表 ──────────────────────────────────────────
    if args.video:
        video_paths = [Path(args.video)]
    else:
        video_dir = Path(args.video_dir)
        video_paths = sorted(
            list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        )

    video_paths = [p for p in video_paths if p.exists()]
    if not video_paths:
        logger.error("未找到任何视频文件，请检查 --video-dir 或 --video 参数")
        sys.exit(1)

    total = len(config_paths) * len(video_paths)
    logger.info(f"共 {len(config_paths)} 个配置 × {len(video_paths)} 个视频 = {total} 次实验")
    logger.info(f"输出目录: {output_dir}")

    # ── 批量运行 ──────────────────────────────────────────────
    all_results: List[Dict] = []
    exp_count = 0

    for cfg_path in config_paths:
        for vid_path in video_paths:
            exp_count += 1
            logger.info(f"\n[{exp_count}/{total}] 开始实验")
            result = run_single_experiment(
                config_path=cfg_path,
                video_path=vid_path,
                output_dir=output_dir,
                max_frames=args.max_frames,
            )
            all_results.append(result)

            status_icon = "✓" if result["status"] == "success" else "✗"
            logger.info(
                f"  {status_icon} {result['run_id']} | "
                f"FPS={result['fps']:.1f} | "
                f"帧数={result['frames_processed']} | "
                f"轨迹数={result['num_tracks']} | "
                f"均长={result['avg_track_length']}"
            )

    # ── 保存汇总 ──────────────────────────────────────────────
    summary = {
        "total_experiments": len(all_results),
        "success_count": sum(1 for r in all_results if r["status"] == "success"),
        "error_count": sum(1 for r in all_results if r["status"] == "error"),
        "output_dir": str(output_dir),
        "results": all_results,
    }

    json_path = output_dir / "experiment_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\n实验汇总 JSON: {json_path}")

    csv_path = output_dir / "experiment_summary.csv"
    save_summary_csv(all_results, csv_path)
    logger.info(f"实验汇总 CSV:  {csv_path}")

    # ── 可选图表 ──────────────────────────────────────────────
    if not args.no_plot:
        logger.info("生成汇总图表...")
        try_plot_summary(all_results, output_dir)

    # ── 打印汇总表格 ──────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"{'实验名称':<35} {'FPS':>6} {'帧数':>6} {'轨迹':>6} {'均长':>7} {'状态':>6}")
    logger.info("-" * 70)
    for r in all_results:
        status_icon = "成功" if r["status"] == "success" else "失败"
        logger.info(
            f"{r['run_id']:<35} {r['fps']:>6.1f} {r['frames_processed']:>6} "
            f"{r['num_tracks']:>6} {r['avg_track_length']:>7.1f} {status_icon:>6}"
        )
    logger.info("=" * 70)

    success = sum(1 for r in all_results if r["status"] == "success")
    logger.info(f"\n完成: {success}/{len(all_results)} 实验成功")

    return summary


if __name__ == "__main__":
    main()
