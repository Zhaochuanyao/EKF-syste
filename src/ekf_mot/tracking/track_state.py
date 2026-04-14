"""
轨迹状态枚举定义
"""

from enum import Enum, auto


class TrackState(Enum):
    """
    轨迹生命周期状态机:

    Tentative  →  Confirmed  →  Lost  →  Removed
        ↑               ↓
        └───────────────┘ (重新命中)

    - Tentative : 新建轨迹，尚未被确认（命中次数不足）
    - Confirmed : 已确认轨迹，正在被稳定跟踪
    - Lost      : 暂时丢失，等待重新关联
    - Removed   : 已删除，不再参与跟踪
    """
    Tentative = auto()
    Confirmed = auto()
    Lost = auto()
    Removed = auto()

    @property
    def name_str(self) -> str:
        return self.name
