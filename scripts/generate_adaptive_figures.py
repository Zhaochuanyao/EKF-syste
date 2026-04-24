"""
生成自适应噪声 EKF 实验三张核心图表

  fig_main_metrics.png     — 6 组主指标对比（MOTA/MOTP/IDSW/AvgLen，含误差棒）
  fig_ablation_metrics.png — 6 组 × 序列 MOTA 折线 + 逐组箱线图
  fig_trajectory_compare.png — G2(Current EKF) vs G6(Full Adaptive) 轨迹追踪质量对比

用法：
  python scripts/generate_adaptive_figures.py
  python scripts/generate_adaptive_figures.py --input outputs/adaptive_ekf/uadetrac_subset/
"""

import sys
import csv
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("gen_figures")

# ── 全局配色（colorblind-friendly, 6 groups）────────────────
COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4"]
GROUP_IDS   = ["G1", "G2", "G3", "G4", "G5", "G6"]
GROUP_SHORT = ["Baseline", "Cur.EKF", "+R-adapt", "+Q-sched", "+RQ-adapt", "Full Adpt"]

OUTPUT_DIR = Path("outputs/adaptive_ekf/uadetrac_subset")


# ═══════════════════════════════════════════════════════════════
# 0. CSV 读取工具
# ═══════════════════════════════════════════════════════════════

def _read_csv(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _flt(d: Dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d[key])
    except (KeyError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════
# 1. fig_main_metrics.png
# ═══════════════════════════════════════════════════════════════

def plot_main_metrics(main_rows: List[Dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    metrics = [
        ("MOTA_mean",  "MOTA_std",  "MOTA",             False),
        ("MOTP_mean",  "MOTP_std",  "MOTP (IoU)",        False),
        ("IDSW_mean",  "IDSW_std",  "ID Switches (↓)",   True),
        ("AvgLen_mean","AvgLen_std","Avg Track Length",   False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Six-Group Ablation: Main Tracking Metrics\n"
        "(Adaptive Noise EKF, Synthetic Sequences)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    labels = GROUP_SHORT

    for ax, (mean_key, std_key, title, lower_better) in zip(axes, metrics):
        means = [_flt(r, mean_key) for r in main_rows]
        stds  = [_flt(r, std_key)  for r in main_rows]
        n_groups = len(means)

        bars = ax.bar(range(n_groups), means, yerr=stds,
                      color=COLORS, edgecolor="white",
                      capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#555"})

        # 标注数值
        for bar, val, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + (max(means) - min(means)) * 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
            )

        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        if lower_better:
            ax.invert_yaxis()
            ax.set_title(title + " ↓ lower=better", fontsize=9, fontweight="bold")

    # 共享图例
    patches = [mpatches.Patch(color=c, label=f"{gid}: {gs}")
               for c, gid, gs in zip(COLORS, GROUP_IDS, GROUP_SHORT)]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.08), fontsize=8.0, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  保存: {out_path}")


# ═══════════════════════════════════════════════════════════════
# 2. fig_ablation_metrics.png
# ═══════════════════════════════════════════════════════════════

def plot_ablation_metrics(ablation_rows: List[Dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 按 group / sequence 整理 MOTA
    groups: Dict[str, Dict[str, float]] = {}
    for r in ablation_rows:
        gid = r["group_id"]
        seq = r["sequence"]
        mota = _flt(r, "MOTA")
        groups.setdefault(gid, {})[seq] = mota

    sequences = sorted({r["sequence"] for r in ablation_rows})
    n_seq     = len(sequences)

    fig, (ax_line, ax_box) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Per-Sequence MOTA: 5-Group Ablation",
        fontsize=12, fontweight="bold",
    )

    # ── 折线图（逐序列）────────────────────────────────────
    x = range(n_seq)
    for gid, gs, color in zip(GROUP_IDS, GROUP_SHORT, COLORS):
        motas = [groups[gid].get(s, float("nan")) for s in sequences]
        ax_line.plot(x, motas, marker="o", markersize=5,
                     label=f"{gid}: {gs}", color=color, linewidth=1.6)

    ax_line.set_xticks(range(n_seq))
    ax_line.set_xticklabels(
        [s.replace("Synthetic_", "S") for s in sequences],
        fontsize=8, rotation=30, ha="right",
    )
    ax_line.set_ylabel("MOTA", fontsize=10)
    ax_line.set_title("Per-Sequence MOTA", fontsize=10, fontweight="bold")
    ax_line.legend(fontsize=8, frameon=False)
    ax_line.axhline(0, color="#999", linestyle="--", linewidth=0.8)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_line.set_axisbelow(True)

    # ── 箱线图（按组）──────────────────────────────────────
    box_data = [
        [groups[gid].get(s, float("nan")) for s in sequences]
        for gid in GROUP_IDS
    ]
    bp = ax_box.boxplot(
        box_data,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "x", "markersize": 5, "alpha": 0.6},
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax_box.set_xticks(range(1, 7))
    ax_box.set_xticklabels(GROUP_SHORT, fontsize=8.5, rotation=15, ha="right")
    ax_box.set_ylabel("MOTA", fontsize=10)
    ax_box.set_title("MOTA Distribution per Group", fontsize=10, fontweight="bold")
    ax_box.axhline(0, color="#999", linestyle="--", linewidth=0.8)
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)
    ax_box.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax_box.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  保存: {out_path}")


# ═══════════════════════════════════════════════════════════════
# 3. fig_trajectory_compare.png
#    重跑合成序列，收集 G2/G5 轨迹对比 GT
# ═══════════════════════════════════════════════════════════════

def _wrap(a: float) -> float:
    while a > math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def _ctrv_step(state: list, dt: float, acc: float = 0.0, alpha: float = 0.0) -> list:
    cx, cy, v, theta, omega, w, h = state
    v_new     = max(0.0, v + acc * dt)
    theta_new = _wrap(theta + omega * dt)
    omega_new = omega + alpha * dt
    if abs(omega) < 1e-4:
        cx_new = cx + v * math.cos(theta) * dt
        cy_new = cy + v * math.sin(theta) * dt
    else:
        cx_new = cx + (v / omega) * (math.sin(theta_new) - math.sin(theta))
        cy_new = cy - (v / omega) * (math.cos(theta_new) - math.cos(theta))
    return [cx_new, cy_new, v_new, theta_new, omega_new, w, h]


def _run_tracker_collect(adaptive_noise_cfg, gt_frames, det_frames):
    """运行 EKF 跟踪器，返回 {gt_id: [(cx,cy),...]} 和 {track_id: [(cx,cy),...]}。"""
    from src.ekf_mot.core.types import Detection
    from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker

    tracker = MultiObjectTracker(
        n_init=2, max_age=15, dt=0.1,
        high_conf_threshold=0.5, low_conf_threshold=0.30,
        gating_threshold_confirmed=9.4877,
        iou_weight=0.4, mahal_weight=0.4, center_weight=0.2,
        cost_threshold_a=0.80, iou_threshold_b=0.35, iou_threshold_c=0.25,
        second_stage_match=True, lost_recovery_stage=True, cost_threshold_a2=0.90,
        min_create_score=0.30, anchor_mode="center",
        adaptive_noise_cfg=adaptive_noise_cfg,
        std_acc=3.0, std_yaw_rate=0.5, std_size=0.1,
        std_cx=5.0, std_cy=5.0, std_w=8.0, std_h=8.0,
        score_adaptive=True, lost_age_q_scale=1.3,
        init_std_cx=10.0, init_std_cy=10.0,
        init_std_v=8.0, init_std_theta=0.8, init_std_omega=0.3,
        init_std_w=15.0, init_std_h=15.0,
    )

    track_hist: Dict[int, List[Tuple[float, float]]] = {}

    for frame_id, (gt, dets) in enumerate(zip(gt_frames, det_frames)):
        detections = [
            Detection(
                bbox=np.array(d["bbox"], dtype=np.float64),
                score=d["score"],
                class_id=d["class_id"],
                class_name=d["class_name"],
                frame_id=frame_id,
            )
            for d in dets
        ]
        active = tracker.step(detections, frame_id, dt=0.1)
        for t in active:
            if t.is_confirmed:
                cx = float(t.ekf.x[0])
                cy = float(t.ekf.x[1])
                track_hist.setdefault(t.track_id, []).append((cx, cy))

    return track_hist


def plot_trajectory_compare(out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # 重新生成同一合成序列（seq_id=2，含明显转弯机动场景）
    sys.path.insert(0, str(ROOT))
    from scripts.run_adaptive_ablation import (
        generate_synthetic_sequence,
        _CFG_FULL,
    )

    SEQ_ID = 2
    N_FRAMES  = 300
    N_VEH     = 4

    gt_frames, det_frames = generate_synthetic_sequence(
        seq_id=SEQ_ID, num_frames=N_FRAMES, num_vehicles=N_VEH
    )

    # GT 轨迹
    gt_hist: Dict[int, List[Tuple[float, float]]] = {}
    for gt in gt_frames:
        for ann in gt:
            x1, y1, x2, y2 = ann["bbox"]
            gt_hist.setdefault(ann["id"], []).append(
                ((x1 + x2) / 2, (y1 + y2) / 2)
            )

    # G2: Current EKF（无自适应）
    logger.info("  运行 G2 (Current EKF) ...")
    hist_g2 = _run_tracker_collect(None, gt_frames, det_frames)

    # G6: Full Adaptive EKF
    logger.info("  运行 G6 (Full Adaptive EKF) ...")
    hist_g6 = _run_tracker_collect(_CFG_FULL, gt_frames, det_frames)

    # ── 绘图 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Trajectory Tracking Quality Comparison\n"
        "Synthetic Sequence (seq_id=2): GT vs G2(Current EKF) vs G6(Full Adaptive EKF)",
        fontsize=12, fontweight="bold",
    )

    cmap = matplotlib.colormaps["tab10"]

    # 子图标题与轨迹集合
    panels = [
        ("Ground Truth",      {gid: traj for gid, traj in gt_hist.items()}, None,      False),
        ("G2: Current EKF",   hist_g2,                                       "#4DBBD5", True),
        ("G6: Full Adaptive", hist_g6,                                       "#8491B4", True),
    ]

    for ax, (title, hist, track_color, show_gt_bg) in zip(axes, panels):
        # GT 背景灰线
        if show_gt_bg:
            for gt_traj in gt_hist.values():
                xs = [p[0] for p in gt_traj]
                ys = [p[1] for p in gt_traj]
                ax.plot(xs, ys, color="#cccccc", linewidth=0.8, zorder=1)

        # 主轨迹
        for i, (tid, traj) in enumerate(hist.items()):
            if len(traj) < 3:
                continue
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            color = track_color if track_color else cmap(i % 10)
            alpha = 0.85 if not show_gt_bg else 1.0
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=alpha, zorder=2)
            # 起点标记
            ax.scatter(xs[0], ys[0], color=color, s=25, zorder=3, marker="o")
            # 终点标记
            ax.scatter(xs[-1], ys[-1], color=color, s=40, zorder=3, marker="s")

        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.invert_yaxis()
        ax.set_title(f"{title}\n(tracks={len(hist)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("x (px)", fontsize=9)
        ax.set_ylabel("y (px)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

    # 图例
    legend_elems = [
        Line2D([0], [0], color="#cccccc", linewidth=1.5, label="GT (background)"),
        Line2D([0], [0], color="#4DBBD5", linewidth=2.0, label="G2: Current EKF"),
        Line2D([0], [0], color="#8491B4", linewidth=2.0, label="G6: Full Adaptive EKF"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=6, label="Track start"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None",
               markersize=7, label="Track end"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  保存: {out_path}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="生成自适应噪声 EKF 实验图表")
    parser.add_argument("--input",  default=str(OUTPUT_DIR), help="CSV 所在目录")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="图表输出目录")
    args = parser.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_csv     = in_dir / "main_metrics.csv"
    ablation_csv = in_dir / "ablation_metrics.csv"

    if not main_csv.exists() or not ablation_csv.exists():
        logger.error(
            f"CSV 不存在，请先运行:\n"
            f"  python scripts/run_adaptive_ablation.py"
        )
        sys.exit(1)

    main_rows    = _read_csv(main_csv)
    ablation_rows = _read_csv(ablation_csv)

    logger.info("生成 fig_main_metrics.png ...")
    plot_main_metrics(main_rows, out_dir / "fig_main_metrics.png")

    logger.info("生成 fig_ablation_metrics.png ...")
    plot_ablation_metrics(ablation_rows, out_dir / "fig_ablation_metrics.png")

    logger.info("生成 fig_trajectory_compare.png ...")
    plot_trajectory_compare(out_dir / "fig_trajectory_compare.png")

    logger.info("\n全部图表生成完毕:")
    for fn in ["fig_main_metrics.png", "fig_ablation_metrics.png",
               "fig_trajectory_compare.png"]:
        p = out_dir / fn
        size_kb = p.stat().st_size // 1024 if p.exists() else 0
        logger.info(f"  {fn}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
