"""设备检测工具"""
import platform


def get_device(requested: str = "cpu") -> str:
    """
    返回实际可用的设备字符串。
    本项目默认 CPU，若请求 cuda 但不可用则回退到 cpu。
    """
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_system_info() -> dict:
    """返回系统信息摘要"""
    info = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch"] = "not installed"
        info["cuda_available"] = False
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = "not installed"
    return info
