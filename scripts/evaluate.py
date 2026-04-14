"""
评估入口
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ekf_mot.metrics.prediction_metrics import PredictionMetrics
from src.ekf_mot.utils.file_io import save_json, load_json
from src.ekf_mot.utils.logger import setup_logger

logger = setup_logger("evaluate")


def evaluate(pred_path: str, gt_path: str = None, output: str = "outputs/metrics.json"):
    pred_data = load_json(pred_path)
    logger.info(f"加载预测结果: {pred_path} ({len(pred_data)} 帧)")

    metrics = {
        "total_frames": len(pred_data),
        "total_track_instances": sum(len(f.get("tracks", [])) for f in pred_data),
    }

    if gt_path and Path(gt_path).exists():
        # TODO: 实现完整的 MOTA/MOTP 评估
        logger.info(f"加载 GT: {gt_path}")
        logger.warning("完整 MOT 评估需要 TrackEval 框架，当前仅统计基础信息")

    save_json(metrics, output)
    logger.info(f"评估结果已保存: {output}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="预测结果 JSON 路径")
    parser.add_argument("--gt", default=None, help="GT 标注路径（可选）")
    parser.add_argument("--output", default="outputs/metrics.json")
    args = parser.parse_args()
    evaluate(args.pred, args.gt, args.output)
