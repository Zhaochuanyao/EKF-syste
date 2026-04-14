"""随机种子工具"""
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
