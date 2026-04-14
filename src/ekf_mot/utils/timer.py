"""
计时工具模块
"""

import time
from collections import defaultdict
from typing import Dict, Optional


class Timer:
    """
    简单计时器，支持多个命名计时段的统计。

    用法:
        timer = Timer()
        with timer.measure("detection"):
            results = detector.predict(frame)
        print(timer.stats())
    """

    def __init__(self) -> None:
        self._records: Dict[str, list] = defaultdict(list)
        self._start: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._start[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        elapsed = (time.perf_counter() - self._start[name]) * 1000  # ms
        self._records[name].append(elapsed)
        return elapsed

    class _MeasureContext:
        def __init__(self, timer: "Timer", name: str) -> None:
            self._timer = timer
            self._name = name

        def __enter__(self) -> "Timer._MeasureContext":
            self._timer.start(self._name)
            return self

        def __exit__(self, *_) -> None:
            self._timer.stop(self._name)

    def measure(self, name: str) -> "_MeasureContext":
        return self._MeasureContext(self, name)

    def mean(self, name: str) -> float:
        records = self._records.get(name, [])
        return sum(records) / len(records) if records else 0.0

    def total(self, name: str) -> float:
        return sum(self._records.get(name, []))

    def stats(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for name, records in self._records.items():
            result[name] = {
                "count": len(records),
                "mean_ms": sum(records) / len(records),
                "total_ms": sum(records),
                "min_ms": min(records),
                "max_ms": max(records),
            }
        return result

    def reset(self) -> None:
        self._records.clear()
        self._start.clear()
