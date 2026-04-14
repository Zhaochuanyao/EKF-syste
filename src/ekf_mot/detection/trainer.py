"""
训练器模块 - 可选的本地微调入口
"""

from pathlib import Path
from typing import Optional
from ..utils.logger import get_logger

logger = get_logger("ekf_mot.detection.trainer")


class YOLOTrainer:
    """
    YOLOv8 轻量训练器（可选功能）。
    默认不执行训练，需要显式调用 train()。
    """

    def __init__(
        self,
        model: str = "yolov8n.pt",
        data_yaml: str = "",
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 8,
        device: str = "cpu",
        project: str = "outputs/train",
        name: str = "yolov8n_custom",
    ) -> None:
        self.model = model
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.project = project
        self.name = name

    def train(self) -> Optional[str]:
        """
        执行训练。

        Returns:
            训练结果目录路径，失败返回 None
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("请安装 ultralytics: pip install ultralytics")
            return None

        if not self.data_yaml or not Path(self.data_yaml).exists():
            logger.error(f"数据集配置文件不存在: {self.data_yaml}")
            return None

        logger.info(
            f"开始训练 | 模型: {self.model} | 数据: {self.data_yaml} | "
            f"Epochs: {self.epochs} | Batch: {self.batch} | Device: {self.device}"
        )
        logger.warning("CPU 训练速度较慢，建议使用 GPU 或减少 epochs")

        model = YOLO(self.model)
        results = model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            project=self.project,
            name=self.name,
            exist_ok=True,
        )
        save_dir = str(results.save_dir)
        logger.info(f"训练完成，结果保存至: {save_dir}")
        return save_dir
