"""
检测评估器 - 将 DetectionMetrics 封装为面向评估流程的高层接口

职责：
  - 接收检测器输出和标注数据
  - 调用 DetectionMetrics 累积统计
  - 支持按类别分别统计
  - 输出结构化评估报告（dict / JSON）

依赖 src/ekf_mot/metrics/detection_metrics.py 的核心计算逻辑。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..core.types import Detection
from ..metrics.detection_metrics import DetectionMetrics, match_detections

logger = logging.getLogger("ekf_mot.detection.evaluator")


class DetectionEvaluator:
    """
    检测评估器。

    用法：
        evaluator = DetectionEvaluator(iou_threshold=0.5)
        for frame_preds, frame_gts in zip(all_preds, all_gts):
            evaluator.update(frame_preds, frame_gts)
        report = evaluator.compute()
        evaluator.save_report("outputs/detection_metrics.json")
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        match_class: bool = False,
        per_class: bool = True,
    ) -> None:
        """
        Args:
            iou_threshold:  IoU 匹配阈值（默认 0.5 = AP50）
            match_class:    是否要求类别一致才能匹配
            per_class:      是否额外统计每个类别的独立指标
        """
        self.iou_threshold = iou_threshold
        self.match_class = match_class
        self.per_class = per_class

        # 全局指标累积器
        self._global = DetectionMetrics(iou_threshold, match_class)

        # 按类别独立累积器 {class_id: DetectionMetrics}
        self._per_class: Dict[int, DetectionMetrics] = {}
        self._class_names: Dict[int, str] = {}

        self._frame_count = 0

    # ──────────────────────────────────────────────────────────
    # 逐帧更新
    # ──────────────────────────────────────────────────────────

    def update(
        self,
        preds: List[Detection],
        gts: List[Detection],
    ) -> Dict[str, int]:
        """
        更新一帧的评估统计。

        Args:
            preds:  当前帧检测结果列表
            gts:    当前帧真实框列表

        Returns:
            当前帧的 {"tp": ..., "fp": ..., "fn": ...}
        """
        self._frame_count += 1

        # 全局统计
        frame_stats = self._global.update(preds, gts)

        # 按类别统计
        if self.per_class:
            # 收集所有出现过的类别
            all_class_ids = set(
                d.class_id for d in preds + gts
            )
            for cid in all_class_ids:
                # 收集类别名
                for d in preds + gts:
                    if d.class_id == cid:
                        self._class_names[cid] = d.class_name
                        break

                if cid not in self._per_class:
                    self._per_class[cid] = DetectionMetrics(
                        self.iou_threshold, match_class=True
                    )

                cls_preds = [d for d in preds if d.class_id == cid]
                cls_gts = [d for d in gts if d.class_id == cid]
                self._per_class[cid].update(cls_preds, cls_gts)

        return frame_stats

    # ──────────────────────────────────────────────────────────
    # 计算报告
    # ──────────────────────────────────────────────────────────

    def compute(self) -> Dict:
        """
        计算完整评估报告。

        Returns:
            {
              "iou_threshold": 0.5,
              "num_frames": ...,
              "global": {"precision": ..., "recall": ..., "f1": ..., "ap50": ..., ...},
              "per_class": {
                "0": {"class_name": "car", "precision": ..., ...},
                ...
              }
            }
        """
        global_result = self._global.compute()

        report = {
            "iou_threshold": self.iou_threshold,
            "num_frames": self._frame_count,
            "global": global_result,
        }

        if self.per_class and self._per_class:
            per_class_report = {}
            for cid, m in sorted(self._per_class.items()):
                cls_result = m.compute()
                cls_result["class_name"] = self._class_names.get(cid, str(cid))
                per_class_report[str(cid)] = cls_result
            report["per_class"] = per_class_report

        return report

    # ──────────────────────────────────────────────────────────
    # 保存报告
    # ──────────────────────────────────────────────────────────

    def save_report(self, output_path: str) -> Path:
        """
        将评估报告保存为 JSON 文件。

        Args:
            output_path:  输出文件路径

        Returns:
            实际写入的文件路径
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        report = self.compute()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"检测评估报告已保存: {out}")
        return out

    # ──────────────────────────────────────────────────────────
    # 打印摘要
    # ──────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """打印格式化的评估摘要"""
        report = self.compute()
        g = report["global"]
        print(f"\n{'='*60}")
        print(f"检测评估结果 (IoU threshold={self.iou_threshold})")
        print(f"{'='*60}")
        print(f"  处理帧数:    {report['num_frames']}")
        print(f"  TP / FP / FN: {g['tp']} / {g['fp']} / {g['fn']}")
        print(f"  Precision:   {g['precision']:.4f}")
        print(f"  Recall:      {g['recall']:.4f}")
        print(f"  F1:          {g['f1']:.4f}")
        print(f"  AP50:        {g['ap50']:.4f}")

        if "per_class" in report:
            print(f"\n  按类别：")
            for cid, cls_r in report["per_class"].items():
                print(
                    f"    [{cid}] {cls_r['class_name']:12s} "
                    f"P={cls_r['precision']:.3f}  R={cls_r['recall']:.3f}  "
                    f"F1={cls_r['f1']:.3f}  AP50={cls_r['ap50']:.3f}"
                )
        print(f"{'='*60}\n")

    def reset(self) -> None:
        """重置所有统计"""
        self._global.reset()
        self._per_class.clear()
        self._class_names.clear()
        self._frame_count = 0
