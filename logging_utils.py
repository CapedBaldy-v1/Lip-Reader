"""
Shared logging helpers for Swin-VALLR.
Creates per-run log files with detailed context.
"""

import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _log_file_path(base_dir: Path, component: str) -> Path:
    date_folder = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base_dir / "logs" / component / date_folder
    root.mkdir(parents=True, exist_ok=True)
    filename = f"{timestamp}_{os.getpid()}.txt"
    return root / filename


def setup_logging(
    component: str,
    base_dir: Optional[Path] = None,
    console_level: Optional[int] = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    base_dir = base_dir or Path(__file__).resolve().parent
    logger = logging.getLogger(component)

    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)
    log_path = _log_file_path(base_dir, component)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(process)d:%(threadName)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    if console_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    logger.propagate = False
    logger._configured = True
    logger.log_path = str(log_path)
    logger.debug("Log file created at %s", log_path)

    return logger


def log_system_info(logger: logging.Logger) -> None:
    logger.info("System platform: %s", platform.platform())
    logger.info("Python version: %s", sys.version.replace("\n", " "))
    logger.info("Python executable: %s", sys.executable)
    logger.info("Process ID: %s", os.getpid())
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Command line: %s", " ".join(sys.argv))


def log_exception(logger: logging.Logger, message: str) -> None:
    logger.error(message, exc_info=True)
