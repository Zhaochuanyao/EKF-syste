"""运行时指标 - FPS、模块耗时统计"""
from typing import Dict
import time


class RuntimeMetrics:
    def __init__(self) -> None:
        self._frame_times: list = []
        self._module_times: Dict[str, list] = {}
        self._start: float = 0.0

    def start_frame(self) -> None:
        self._start = time.perf_counter()

    def end_frame(self) -> float:
        elapsed = (time.perf_counter() - self._start) * 1000
        self._frame_times.append(elapsed)
        return elapsed

    def record(self, name: str, ms: float) -> None:
        self._module_times.setdefault(name, []).append(ms)

    def compute(self) -> Dict[str, float]:
        if not self._frame_times:
            return {}
        avg_ms = sum(self._frame_times) / len(self._frame_times)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        result = {
            "fps": fps,
            "avg_frame_ms": avg_ms,
            "total_frames": len(self._frame_times),
        }
        for name, times in self._module_times.items():
            result[f"{name}_avg_ms"] = sum(times) / len(times)
        return result

    def reset(self) -> None:
        self._frame_times.clear()
        self._module_times.clear()
