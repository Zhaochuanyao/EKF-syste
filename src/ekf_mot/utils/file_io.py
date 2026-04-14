"""文件 IO 工具"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """支持 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, cls=NumpyEncoder)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: List[Dict], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
