"""
配置管理模块 - 支持 YAML 配置加载与多层合并
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典，override 中的值覆盖 base 中的值。
    对嵌套字典递归合并，而不是直接替换。
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """加载单个 YAML 文件"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(
    config_path: Union[str, Path],
    base_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    加载配置文件，支持通过 _base_ 字段继承基础配置。

    Args:
        config_path: 实验配置文件路径
        base_path: 基础配置文件路径（可选，优先级低于 _base_ 字段）

    Returns:
        合并后的配置字典
    """
    config_path = Path(config_path)
    exp_cfg = load_yaml(config_path)

    # 确定基础配置路径
    base_cfg: Dict[str, Any] = {}

    # 优先使用配置文件中的 _base_ 字段
    base_ref = exp_cfg.pop("_base_", None)
    if base_ref is not None:
        # _base_ 路径相对于项目根目录
        resolved = Path(base_ref)
        if not resolved.is_absolute():
            # 尝试相对于当前工作目录
            if resolved.exists():
                base_cfg = load_yaml(resolved)
            else:
                # 尝试相对于配置文件所在目录的上级
                alt = config_path.parent.parent / resolved
                if alt.exists():
                    base_cfg = load_yaml(alt)
    elif base_path is not None:
        base_cfg = load_yaml(base_path)

    # 合并：base_cfg 为底，exp_cfg 覆盖
    merged = _deep_merge(base_cfg, exp_cfg)
    return merged


class Config:
    """
    配置对象，支持通过属性访问和字典访问。

    用法:
        cfg = Config.from_yaml("configs/exp/demo_cpu.yaml")
        print(cfg.detector.conf)
        print(cfg["detector"]["conf"])
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data
        # 将顶层键转换为属性
        for key, val in data.items():
            if isinstance(val, dict):
                setattr(self, key, Config(val))
            else:
                setattr(self, key, val)

    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path],
        base_path: Optional[Union[str, Path]] = None,
    ) -> "Config":
        """从 YAML 文件加载配置"""
        data = load_config(config_path, base_path)
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        """安全获取配置值"""
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        """转换回字典"""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data})"


def get_default_config() -> Dict[str, Any]:
    """返回内置默认配置（不依赖文件）"""
    return {
        "runtime": {
            "frame_skip": 1,
            "max_frames": None,
            "seed": 42,
            "device": "cpu",
        },
        "detector": {
            "backend": "ultralytics",
            "model": "yolov8n",
            "weights": "weights/yolov8n.pt",
            "imgsz": 640,
            "conf": 0.35,
            "iou": 0.5,
            "max_det": 100,
            "classes": None,
            "half": False,
            "warmup_iters": 2,
        },
        "tracker": {
            "dt": 0.04,
            "auto_dt": True,
            "n_init": 3,
            "max_age": 20,
            "iou_threshold": 0.3,
            "max_iou_distance": 0.7,
            "max_mahal_distance": 9.4877,
            "gating_threshold": 9.4877,
            "second_stage_match": True,
            "second_stage_conf": 0.1,
            "cost_iou_weight": 0.5,
            "cost_mahal_weight": 0.5,
        },
        "ekf": {
            "process_noise": {
                "std_acc": 2.0,
                "std_yaw_rate": 0.5,
                "std_size": 0.1,
            },
            "measurement_noise": {
                "std_cx": 5.0,
                "std_cy": 5.0,
                "std_w": 10.0,
                "std_h": 10.0,
                "score_adaptive": True,
            },
            "initial_covariance": {
                "std_cx": 10.0,
                "std_cy": 10.0,
                "std_v": 5.0,
                "std_theta": 0.5,
                "std_omega": 0.2,
                "std_w": 20.0,
                "std_h": 20.0,
            },
            "omega_threshold": 0.001,
        },
        "prediction": {
            "future_steps": [1, 5, 10],
            "smooth_history": True,
            "smooth_window": 5,
        },
        "visualization": {
            "draw_bbox": True,
            "draw_tracks": True,
            "draw_future": True,
            "draw_covariance": True,
            "draw_id": True,
            "draw_score": True,
            "track_history_len": 30,
            "future_color": [0, 255, 255],
            "font_scale": 0.5,
            "line_thickness": 2,
        },
        "output": {
            "save_video": True,
            "save_json": True,
            "save_csv": True,
            "video_fps": 25,
            "video_codec": "mp4v",
            "output_dir": "outputs/",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "file": None,
        },
    }
