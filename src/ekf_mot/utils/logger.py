"""
日志工具模块
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ekf_mot",
    level: str = "INFO",
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    创建并配置 logger。

    Args:
        name: logger 名称
        level: 日志级别字符串
        fmt: 日志格式
        log_file: 日志文件路径（None 则只输出到控制台）

    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复添加 handler

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件 handler（可选）
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "ekf_mot") -> logging.Logger:
    """获取已存在的 logger，若不存在则创建默认 logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
