"""
六组消融实验脚本 — 基于新息统计的分层自适应噪声调度 EKF

实验矩阵（6 组）：
  G1  BaselineTracker    — 纯 IoU 关联 + 恒速线性外推，无 EKF
  G2  Current EKF        — CTRV-EKF，无自适应噪声
  G3  EKF+R-adapt        — CTRV-EKF + 仅观测噪声自适应
  G4  EKF+Q-schedule     — CTRV-EKF + 仅过程噪声机动调度
  G5  EKF+RQ-adapt       — CTRV-EKF + R+Q 自适应，无鲁棒更新
  G6  Full Adaptive EKF  — CTRV-EKF + R+Q 自适应 + 鲁棒更新（skip/clip）

数据源（按优先级）：
  1. UA-DETRAC cpu_small 子集（8 序列，需本地数据）
  2. 内置合成序列（UA-DETRAC 不可用时自动回退）

输出（outputs/adaptive_ekf/uadetrac_subset/）：
  main_metrics.csv     — 6 组跨序列平均指标（MOTA/MOTP/IDSW/AvgLen）
  ablation_metrics.csv — 6 组 × N 序列逐序列原始指标
  significance.csv     — Full Adaptive EKF(G6) vs 其余 5 组 Wilcoxon 检验
  diagnostics.csv      — 自适应组（G3/G4/G5/G6）逐序列诊断统计

用法：
  python scripts/run_adaptive_ablation.py
  python scripts/run_adaptive_ablation.py --data uadetrac
  python scripts/run_adaptive_ablation.py --data synthetic --sequences 8
"""

import sys
import csv
import math
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("adaptive_ablation")

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs/adaptive_ekf/uadetrac_subset")
UADETRAC_TRAIN_DIR = Path("data/UA-DETRAC/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML")
UADETRAC_TEST_DIR = Path("data/UA-DETRAC/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML")
UADETRAC_SEQS = [
    "MVI_20011", "MVI_20032", "MVI_20051", "MVI_20061",
    "MVI_39761", "MVI_39771", "MVI_40732", "MVI_40751",
]
MAX_FRAMES_PER_SEQ = 1200

GROUP_IDS   = ["G1", "G2", "G3", "G4", "G5", "G6"]
GROUP_NAMES = [
    "BaselineTracker",
    "Current EKF",
    "EKF+R-adapt",
    "EKF+Q-schedule",
    "EKF+RQ-adapt",
    "Full Adaptive EKF",
]

_ADAPTIVE_BASE = dict(
    nis_threshold=9.4877,
    drop_threshold=20.0,
    lambda_r=0.3,
    lambda_q=0.3,
    beta=0.85,
    q_max_scale=4.0,
    delta_max=400.0,
    low_score=0.35,
    use_robust_update=True,
    robust_clip_delta=25.0,
    recover_alpha_r=0.65,
    maneuver_cap=3.0,
    maneuver_w_nis=1.0,
    maneuver_w_omega=0.8,
    maneuver_w_theta=0.5,
)

# G3: only R adaptation (q_adapt_on=False via only_r_adapt=True)
_CFG_R_ONLY = {**_ADAPTIVE_BASE, "enabled": True, "only_r_adapt": True}
# G4: only Q scheduling (r_adapt_on=False via only_q_schedule=True)
_CFG_Q_ONLY = {**_ADAPTIVE_BASE, "enabled": True, "only_q_schedule": True}
# G5: R+Q adapt, robust update OFF (use_robust_update=False → robust_on=False)
_CFG_RQ     = {**_ADAPTIVE_BASE, "enabled": True, "use_robust_update": False}
# G6: full adaptive (R+Q + robust skip/clip)
_CFG_FULL   = {**_ADAPTIVE_BASE, "enabled": True}

# None  → G2 (disabled)
ADAPTIVE_CFGS = [None, None, _CFG_R_ONLY, _CFG_Q_ONLY, _CFG_RQ, _CFG_FULL]

# ═══════════════════════════════════════════════════════════════
# 1. 合成序列生成（UA-DETRAC 不可用时的备选）
# ═══════════════════════════════════════════════════════════════

def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
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


def generate_synthetic_sequence(
    seq_id: int,
    num_frames: int = 300,
    num_vehicles: int = 5,
    dt: float = 0.1,
    miss_prob: float = 0.08,
    fp_rate: float = 0.04,
    pos_noise_std: float = 4.0,
    size_noise_std: float = 2.5,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """生成合成 GT + 带噪检测对。"""
    rng = np.random.default_rng(42 + seq_id * 7)

    vehicles = []
    for i in range(num_vehicles):
        cx    = rng.uniform(100, 1100)
        cy    = rng.uniform(100, 600)
        v     = rng.uniform(15, 80)
        theta = rng.uniform(-math.pi, math.pi)
        omega = rng.uniform(-0.08, 0.08)
        w     = rng.uniform(40, 90)
        h     = rng.uniform(25, 60)
        vehicles.append({
            "id": i + 1,
            "state": [cx, cy, v, theta, omega, w, h],
            "occlude": 0,
        })

    gt_frames:  List[List[Dict]] = []
    det_frames: List[List[Dict]] = []

    for _ in range(num_frames):
        gt_ann:  List[Dict] = []
        det_ann: List[Dict] = []

        for veh in vehicles:
            acc   = rng.choice([-12.0, 0.0, 0.0, 0.0, 12.0]) if rng.random() < 0.04 else 0.0
            alpha = rng.uniform(-0.25, 0.25)                  if rng.random() < 0.05 else 0.0
            veh["state"] = _ctrv_step(veh["state"], dt, acc=acc, alpha=alpha)

            cx, cy, v, theta, omega, w, h = veh["state"]
            cx = float(np.clip(cx, w / 2, 1280 - w / 2))
            cy = float(np.clip(cy, h / 2, 720  - h / 2))
            veh["state"][0] = cx
            veh["state"][1] = cy

            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            gt_ann.append({"id": veh["id"], "bbox": [x1, y1, x2, y2]})

            if veh["occlude"] > 0:
                veh["occlude"] -= 1
                continue
            if rng.random() < miss_prob:
                veh["occlude"] = int(rng.integers(1, 4))
                continue

            dx = rng.normal(0, pos_noise_std)
            dy = rng.normal(0, pos_noise_std)
            dw = rng.normal(0, size_noise_std)
            dh = rng.normal(0, size_noise_std)
            score = float(np.clip(rng.normal(0.78, 0.10), 0.40, 0.99))
            det_ann.append({
                "bbox": [x1 + dx - dw / 2, y1 + dy - dh / 2,
                         x2 + dx + dw / 2, y2 + dy + dh / 2],
                "score": score,
                "class_id": 2,
                "class_name": "car",
            })

        # 假阳性
        n_fp = int(rng.poisson(fp_rate * num_vehicles))
        for _ in range(n_fp):
            cx_f = rng.uniform(60, 1220)
            cy_f = rng.uniform(50, 670)
            wf   = rng.uniform(35, 85)
            hf   = rng.uniform(22, 55)
            det_ann.append({
                "bbox": [cx_f - wf / 2, cy_f - hf / 2,
                         cx_f + wf / 2, cy_f + hf / 2],
                "score": float(rng.uniform(0.30, 0.52)),
                "class_id": 2,
                "class_name": "car",
            })

        gt_frames.append(gt_ann)
        det_frames.append(det_ann)

    return gt_frames, det_frames


# ═══════════════════════════════════════════════════════════════
# 2. UA-DETRAC 数据加载（可选）
# ═══════════════════════════════════════════════════════════════

def _load_uadetrac_gt(xml_path: Path, max_frames: int) -> Optional[List[List[Dict]]]:
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(xml_path)).getroot()
    except Exception as e:
        logger.warning(f"XML 解析失败: {xml_path} — {e}")
        return None

    frames_map: Dict[int, List[Dict]] = {}
    for frame_elem in root.findall(".//frame"):
        fnum = int(frame_elem.get("num", 0))
        if fnum > max_frames:
            continue
        annots = []
        for tgt in frame_elem.findall(".//target"):
            tid = int(tgt.get("id", 0))
            box = tgt.find("box")
            if box is None:
                continue
            left   = float(box.get("left",   0))
            top    = float(box.get("top",    0))
            width  = float(box.get("width",  0))
            height = float(box.get("height", 0))
            annots.append({"id": tid,
                           "bbox": [left, top, left + width, top + height]})
        frames_map[fnum] = annots

    if not frames_map:
        return None
    max_f = max(frames_map)
    return [frames_map.get(i, []) for i in range(1, max_f + 1)]


def _gt_to_dets(
    gt_frame: List[Dict],
    rng: np.random.Generator,
    miss_prob: float = 0.10,
    pos_noise_std: float = 5.0,
    size_noise_std: float = 3.0,
    fp_rate: float = 0.03,
) -> List[Dict]:
    dets = []
    for ann in gt_frame:
        if rng.random() < miss_prob:
            continue
        x1, y1, x2, y2 = ann["bbox"]
        dx = rng.normal(0, pos_noise_std);  dy = rng.normal(0, pos_noise_std)
        dw = rng.normal(0, size_noise_std); dh = rng.normal(0, size_noise_std)
        score = float(np.clip(rng.normal(0.75, 0.12), 0.35, 0.99))
        dets.append({
            "bbox": [x1 + dx - dw / 2, y1 + dy - dh / 2,
                     x2 + dx + dw / 2, y2 + dy + dh / 2],
            "score": score, "class_id": 2, "class_name": "car",
        })
    n_fp = int(rng.poisson(fp_rate * max(len(gt_frame), 1)))
    for _ in range(n_fp):
        cx_f = rng.uniform(50, 910);  cy_f = rng.uniform(40, 500)
        wf   = rng.uniform(30, 100);  hf   = rng.uniform(20, 80)
        dets.append({
            "bbox": [cx_f - wf / 2, cy_f - hf / 2,
                     cx_f + wf / 2, cy_f + hf / 2],
            "score": float(rng.uniform(0.30, 0.50)),
            "class_id": 2, "class_name": "car",
        })
    return dets


# ═══════════════════════════════════════════════════════════════
# 3. G1：简单基线跟踪器（IoU + 恒速外推）
# ═══════════════════════════════════════════════════════════════

def _iou_box(a: List[float], b: List[float]) -> float:
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class _BTrack:
    _counter = 0

    def __init__(self, bbox: List[float], score: float) -> None:
        _BTrack._counter += 1
        self.track_id    = _BTrack._counter
        self.bbox        = bbox[:]
        self.prev_bbox   = bbox[:]
        self.score       = score
        self.hits        = 1
        self.age         = 1
        self.time_since_update = 0

    def predict(self) -> None:
        dx = self.bbox[0] - self.prev_bbox[0]
        dy = self.bbox[1] - self.prev_bbox[1]
        self.prev_bbox = self.bbox[:]
        self.bbox = [self.bbox[0] + dx, self.bbox[1] + dy,
                     self.bbox[2] + dx, self.bbox[3] + dy]
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: List[float], score: float) -> None:
        self.prev_bbox = self.bbox[:]
        self.bbox      = bbox[:]
        self.score     = score
        self.hits     += 1
        self.time_since_update = 0


class BaselineTracker:
    def __init__(self, n_init: int = 2, max_age: int = 5,
                 iou_thr: float = 0.25, min_score: float = 0.35) -> None:
        self.n_init     = n_init
        self.max_age    = max_age
        self.iou_thr    = iou_thr
        self.min_score  = min_score
        self._tracks: List[_BTrack] = []
        _BTrack._counter = 0

    def reset(self) -> None:
        self._tracks.clear()
        _BTrack._counter = 0

    def step(self, dets_raw: List[Dict]) -> List[Tuple[int, List[float]]]:
        for t in self._tracks:
            t.predict()

        dets = [d for d in dets_raw if d["score"] >= self.min_score]

        matched_t: set = set()
        matched_d: set = set()
        if self._tracks and dets:
            pairs = sorted(
                [(i, j, _iou_box(self._tracks[i].bbox, dets[j]["bbox"]))
                 for i in range(len(self._tracks))
                 for j in range(len(dets))],
                key=lambda x: -x[2],
            )
            for i, j, iou_val in pairs:
                if iou_val < self.iou_thr:
                    break
                if i in matched_t or j in matched_d:
                    continue
                self._tracks[i].update(dets[j]["bbox"], dets[j]["score"])
                matched_t.add(i);  matched_d.add(j)

        self._tracks = [
            t for i, t in enumerate(self._tracks)
            if i in matched_t or t.time_since_update <= self.max_age
        ]

        for j, d in enumerate(dets):
            if j not in matched_d:
                self._tracks.append(_BTrack(d["bbox"], d["score"]))

        return [(t.track_id, t.bbox[:])
                for t in self._tracks if t.hits >= self.n_init]


# ═══════════════════════════════════════════════════════════════
# 4. EKF 跟踪器工厂
# ═══════════════════════════════════════════════════════════════

def _build_ekf_tracker(adaptive_noise_cfg):
    from src.ekf_mot.tracking.multi_object_tracker import MultiObjectTracker
    return MultiObjectTracker(
        n_init=2, max_age=15, dt=0.1,
        high_conf_threshold=0.5,
        low_conf_threshold=0.30,
        gating_threshold_confirmed=9.4877,
        iou_weight=0.4, mahal_weight=0.4, center_weight=0.2,
        cost_threshold_a=0.80,
        iou_threshold_b=0.35,
        iou_threshold_c=0.25,
        second_stage_match=True,
        lost_recovery_stage=True,
        cost_threshold_a2=0.90,
        min_create_score=0.30,
        anchor_mode="center",
        adaptive_noise_cfg=adaptive_noise_cfg,
        std_acc=3.0, std_yaw_rate=0.5, std_size=0.1,
        std_cx=5.0, std_cy=5.0, std_w=8.0, std_h=8.0,
        score_adaptive=True,
        lost_age_q_scale=1.3,
        init_std_cx=10.0, init_std_cy=10.0,
        init_std_v=8.0, init_std_theta=0.8, init_std_omega=0.3,
        init_std_w=15.0, init_std_h=15.0,
    )


# ═══════════════════════════════════════════════════════════════
# 5. 单序列实验
# ═══════════════════════════════════════════════════════════════

def run_sequence(
    group_idx: int,
    gt_frames: List[List[Dict]],
    det_frames: List[List[Dict]],
    iou_eval_thr: float = 0.5,
) -> Tuple[Dict, Optional[Dict]]:
    """
    运行一组（group_idx）在一条序列上的完整跟踪 + 评估。

    Returns:
        (metrics_dict, diag_dict_or_None)
    """
    from src.ekf_mot.metrics.tracking_metrics import TrackingEvaluator

    evaluator = TrackingEvaluator(iou_threshold=iou_eval_thr)

    if group_idx == 0:
        # G1: BaselineTracker
        tracker = BaselineTracker()
        for frame_id, (gt, dets) in enumerate(zip(gt_frames, det_frames)):
            pred = tracker.step(dets)
            evaluator.update(pred, [(a["id"], a["bbox"]) for a in gt])
        return evaluator.compute(), None

    # G2-G5: EKF
    from src.ekf_mot.core.types import Detection

    cfg = ADAPTIVE_CFGS[group_idx]  # None for G2
    tracker = _build_ekf_tracker(cfg)
    tracker.reset()

    all_diag: List[Dict] = []

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
        pred = [(t.track_id, t.get_bbox().tolist())
                for t in active if t.is_confirmed]
        evaluator.update(pred, [(a["id"], a["bbox"]) for a in gt])

        if cfg is not None:
            for t in active:
                d = t.get_adaptive_diagnostics()
                if d is not None:
                    all_diag.append(d)

    metrics = evaluator.compute()

    diag = None
    if all_diag:
        total_upd  = sum(d["total_updates"]          for d in all_diag)
        skip_cnt   = sum(d["skipped_update_count"]   for d in all_diag)
        abn_cnt    = sum(d["abnormal_update_count"]  for d in all_diag)
        nis_emas   = [d["avg_nis"]                   for d in all_diag]
        maneuvers  = [d["maneuver_memory"]            for d in all_diag]
        diag = {
            "total_updates":   total_upd,
            "skip_rate":       round(skip_cnt  / max(total_upd, 1), 4),
            "abnormal_rate":   round(abn_cnt   / max(total_upd, 1), 4),
            "avg_nis_ema":     round(float(np.mean(nis_emas))  if nis_emas  else 0.0, 4),
            "avg_maneuver":    round(float(np.mean(maneuvers)) if maneuvers else 0.0, 4),
        }

    return metrics, diag


# ═══════════════════════════════════════════════════════════════
# 6. 显著性检验
# ═══════════════════════════════════════════════════════════════

def _wilcoxon_or_ttest(a: List[float], b: List[float]) -> Tuple[float, str]:
    """Full Adaptive EKF(a) vs 对比组(b)，返回 (p_value, test_name)。"""
    if len(a) < 2:
        return float("nan"), "N/A"
    try:
        from scipy.stats import wilcoxon
        diffs = [x - y for x, y in zip(a, b)]
        if all(d == 0.0 for d in diffs):
            return 1.0, "wilcoxon"
        stat, p = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        return float(p), "wilcoxon"
    except ImportError:
        pass
    # 回退到配对 t 检验
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d  = float(np.std(diffs, ddof=1))
    if std_d < 1e-12:
        return (0.0 if mean_d > 0 else 1.0), "t-test"
    t_stat = mean_d / (std_d / math.sqrt(n))
    from math import erfc, sqrt
    p = 0.5 * erfc(t_stat / sqrt(2))  # one-sided
    return float(p), "t-test"


# ═══════════════════════════════════════════════════════════════
# 7. CSV 输出工具
# ═══════════════════════════════════════════════════════════════

def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info(f"  已保存: {path}")


# ═══════════════════════════════════════════════════════════════
# 8. 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="自适应噪声 EKF 五组消融实验")
    parser.add_argument(
        "--data", choices=["auto", "uadetrac", "synthetic"], default="auto",
        help="数据源：auto=优先 UA-DETRAC，synthetic=强制合成数据",
    )
    parser.add_argument("--sequences", type=int, default=8,
                        help="合成序列数量（仅 --data synthetic 时有效）")
    parser.add_argument("--frames", type=int, default=300,
                        help="每条合成序列帧数")
    parser.add_argument("--vehicles", type=int, default=5,
                        help="每条合成序列车辆数")
    parser.add_argument("--iou-eval", type=float, default=0.5,
                        help="评估 IoU 阈值")
    parser.add_argument("--n-seqs", type=int, default=8,
                        help="UA-DETRAC 序列数量（从全部可用 XML 中取前 N 条）")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="输出根目录")
    args = parser.parse_args()

    # ── 创建带时间戳的输出子目录 ──────────────────────────────
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    n_seqs_actual = args.n_seqs if args.data in ("auto", "uadetrac") else args.sequences
    run_name = f"{timestamp}_n{n_seqs_actual}"
    out_dir = Path(args.output) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {out_dir.resolve()}")

    # ── 确定序列列表 ──────────────────────────────────────────
    use_uadetrac = False
    sequences: List[Dict] = []   # [{"name": str, "gt": [...], "det": [...]}]

    if args.data in ("auto", "uadetrac"):
        # 扫描 Train + Test 全部可用 XML，按文件名排序后取前 n_seqs 条
        train_xmls = sorted(UADETRAC_TRAIN_DIR.glob("*.xml")) if UADETRAC_TRAIN_DIR.exists() else []
        test_xmls = sorted(UADETRAC_TEST_DIR.glob("*.xml")) if UADETRAC_TEST_DIR.exists() else []
        all_xmls = train_xmls + test_xmls
        selected = [p.stem for p in all_xmls[:args.n_seqs]]
        missing = 0
        for seq_name in selected:
            # 先尝试 Train，再尝试 Test
            xml_path = UADETRAC_TRAIN_DIR / f"{seq_name}.xml"
            if not xml_path.exists():
                xml_path = UADETRAC_TEST_DIR / f"{seq_name}.xml"
            if not xml_path.exists():
                missing += 1
                continue
            gt = _load_uadetrac_gt(xml_path, MAX_FRAMES_PER_SEQ)
            if gt is None:
                missing += 1
                continue
            rng = np.random.default_rng(seed=hash(seq_name) & 0xFFFFFFFF)
            det = [_gt_to_dets(frame, rng) for frame in gt]
            sequences.append({"name": seq_name, "gt": gt, "det": det})

        if sequences:
            use_uadetrac = True
            logger.info(f"UA-DETRAC: 加载 {len(sequences)}/{len(selected)} 条序列"
                        + (f"（{missing} 条缺失，已跳过）" if missing else ""))
        elif args.data == "uadetrac":
            logger.error("UA-DETRAC 数据集不可用，请运行 python scripts/check_uadetrac_subset.py 检查")
            sys.exit(1)

    if not sequences:
        logger.warning("UA-DETRAC 数据不可用，回退至合成序列")
        for i in range(args.sequences):
            gt, det = generate_synthetic_sequence(
                seq_id=i,
                num_frames=args.frames,
                num_vehicles=args.vehicles,
            )
            sequences.append({"name": f"Synthetic_{i:02d}", "gt": gt, "det": det})
        logger.info(f"合成序列：{len(sequences)} 条，每条 {args.frames} 帧，{args.vehicles} 辆车")

    data_tag = "uadetrac" if use_uadetrac else "synthetic"

    # ── 6 组 × N 序列实验 ─────────────────────────────────────
    #   raw_results[g][s] = (metrics_dict, diag_dict_or_None)
    raw_results: List[List[Tuple[Dict, Optional[Dict]]]] = [[] for _ in range(6)]

    n_total = 6 * len(sequences)
    done    = 0
    t0      = time.perf_counter()

    for g_idx, g_name in enumerate(GROUP_NAMES):
        logger.info(f"\n[{g_name}]")
        for s_idx, seq in enumerate(sequences):
            done += 1
            t_seq = time.perf_counter()
            metrics, diag = run_sequence(
                group_idx=g_idx,
                gt_frames=seq["gt"],
                det_frames=seq["det"],
                iou_eval_thr=args.iou_eval,
            )
            elapsed = time.perf_counter() - t_seq
            logger.info(
                f"  [{done}/{n_total}] {seq['name']:14s} | "
                f"MOTA={metrics['MOTA']:+.4f}  MOTP={metrics['MOTP']:.4f}  "
                f"IDSW={metrics['ID_Switch']:3d}  AvgLen={metrics['avg_track_length']:6.1f}  "
                f"({elapsed:.1f}s)"
            )
            raw_results[g_idx].append((metrics, diag))

    total_elapsed = time.perf_counter() - t0
    logger.info(f"\n总耗时: {total_elapsed:.1f}s")

    seq_names = [s["name"] for s in sequences]

    # ── 生成 ablation_metrics.csv ─────────────────────────────
    ablation_rows: List[Dict] = []
    for g_idx, g_name in enumerate(GROUP_NAMES):
        for s_idx, seq_name in enumerate(seq_names):
            m, _ = raw_results[g_idx][s_idx]
            ablation_rows.append({
                "group_id":        GROUP_IDS[g_idx],
                "group_name":      g_name,
                "sequence":        seq_name,
                "data_source":     data_tag,
                "MOTA":            m["MOTA"],
                "MOTP":            m["MOTP"],
                "ID_Switch":       m["ID_Switch"],
                "avg_track_length": m["avg_track_length"],
                "num_tracks":      m["num_tracks"],
                "TP":              m["TP"],
                "FP":              m["FP"],
                "FN":              m["FN"],
                "total_GT":        m["total_GT"],
                "num_frames":      m["num_frames"],
            })
    _write_csv(
        out_dir / "ablation_metrics.csv",
        ablation_rows,
        ["group_id", "group_name", "sequence", "data_source",
         "MOTA", "MOTP", "ID_Switch", "avg_track_length",
         "num_tracks", "TP", "FP", "FN", "total_GT", "num_frames"],
    )

    # ── 生成 main_metrics.csv ─────────────────────────────────
    main_rows: List[Dict] = []
    for g_idx, g_name in enumerate(GROUP_NAMES):
        ms = [raw_results[g_idx][s][0] for s in range(len(sequences))]
        main_rows.append({
            "group_id":          GROUP_IDS[g_idx],
            "group_name":        g_name,
            "data_source":       data_tag,
            "num_sequences":     len(sequences),
            "MOTA_mean":         round(float(np.mean([m["MOTA"]            for m in ms])), 4),
            "MOTA_std":          round(float(np.std( [m["MOTA"]            for m in ms])), 4),
            "MOTP_mean":         round(float(np.mean([m["MOTP"]            for m in ms])), 4),
            "MOTP_std":          round(float(np.std( [m["MOTP"]            for m in ms])), 4),
            "IDSW_mean":         round(float(np.mean([m["ID_Switch"]       for m in ms])), 2),
            "IDSW_std":          round(float(np.std( [m["ID_Switch"]       for m in ms])), 2),
            "AvgLen_mean":       round(float(np.mean([m["avg_track_length"] for m in ms])), 2),
            "AvgLen_std":        round(float(np.std( [m["avg_track_length"] for m in ms])), 2),
            "NumTracks_mean":    round(float(np.mean([m["num_tracks"]      for m in ms])), 1),
        })
    _write_csv(
        out_dir / "main_metrics.csv",
        main_rows,
        ["group_id", "group_name", "data_source", "num_sequences",
         "MOTA_mean", "MOTA_std", "MOTP_mean", "MOTP_std",
         "IDSW_mean", "IDSW_std", "AvgLen_mean", "AvgLen_std", "NumTracks_mean"],
    )

    # ── 生成 significance.csv ─────────────────────────────────
    # Full Adaptive EKF (G6, index=5) vs 其余 5 组
    g6_motas = [raw_results[5][s][0]["MOTA"] for s in range(len(sequences))]
    sig_rows: List[Dict] = []
    for g_idx in range(5):
        g_motas = [raw_results[g_idx][s][0]["MOTA"] for s in range(len(sequences))]
        p_val, test_name = _wilcoxon_or_ttest(g6_motas, g_motas)
        delta_mota = round(
            float(np.mean(g6_motas)) - float(np.mean(g_motas)), 4
        )
        sig_rows.append({
            "comparison":     f"G6 vs {GROUP_IDS[g_idx]}",
            "full_adaptive":  "Full Adaptive EKF",
            "baseline_group": GROUP_NAMES[g_idx],
            "metric":         "MOTA",
            "delta_mean":     delta_mota,
            "p_value":        round(p_val, 6) if not math.isnan(p_val) else "NaN",
            "significant_p05": "Yes" if (not math.isnan(p_val) and p_val < 0.05) else "No",
            "test":           test_name,
            "n_sequences":    len(sequences),
        })
    _write_csv(
        out_dir / "significance.csv",
        sig_rows,
        ["comparison", "full_adaptive", "baseline_group", "metric",
         "delta_mean", "p_value", "significant_p05", "test", "n_sequences"],
    )

    # ── 生成 diagnostics.csv ──────────────────────────────────
    diag_rows: List[Dict] = []
    for g_idx in [2, 3, 4, 5]:   # G3/G4/G5/G6
        for s_idx, seq_name in enumerate(seq_names):
            _, diag = raw_results[g_idx][s_idx]
            if diag is None:
                diag = {"total_updates": 0, "skip_rate": 0.0,
                        "abnormal_rate": 0.0, "avg_nis_ema": 0.0,
                        "avg_maneuver": 0.0}
            diag_rows.append({
                "group_id":      GROUP_IDS[g_idx],
                "group_name":    GROUP_NAMES[g_idx],
                "sequence":      seq_name,
                "total_updates": diag["total_updates"],
                "skip_rate":     diag["skip_rate"],
                "abnormal_rate": diag["abnormal_rate"],
                "avg_nis_ema":   diag["avg_nis_ema"],
                "avg_maneuver":  diag["avg_maneuver"],
            })
    _write_csv(
        out_dir / "diagnostics.csv",
        diag_rows,
        ["group_id", "group_name", "sequence",
         "total_updates", "skip_rate", "abnormal_rate",
         "avg_nis_ema", "avg_maneuver"],
    )

    # ── 终端汇总打印 ──────────────────────────────────────────
    logger.info("\n" + "=" * 72)
    logger.info(f"{'组别':<22} {'MOTA':>8} {'MOTP':>8} {'IDSW':>7} {'AvgLen':>8}")
    logger.info("-" * 72)
    for row in main_rows:
        logger.info(
            f"{row['group_name']:<22} "
            f"{row['MOTA_mean']:>+8.4f} "
            f"{row['MOTP_mean']:>8.4f} "
            f"{row['IDSW_mean']:>7.1f} "
            f"{row['AvgLen_mean']:>8.2f}"
        )
    logger.info("=" * 72)
    logger.info(f"\n显著性（G6 Full Adaptive EKF vs 其余，单侧 MOTA）：")
    for row in sig_rows:
        logger.info(
            f"  G6 vs {row['baseline_group']:<18s} "
            f"ΔMOTA={row['delta_mean']:+.4f}  "
            f"p={row['p_value']}  {'★' if row['significant_p05']=='Yes' else ''}"
        )
    logger.info(f"\n输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
