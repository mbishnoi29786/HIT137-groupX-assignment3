"""Centralized logging configuration for the AI Model Integration app."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
import platform
import sys
from typing import Optional


def _default_log_path() -> Path:
    project_root = Path(__file__).resolve().parent
    return project_root / "app.log"


def initialize_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure application logging once at start-up."""

    if logging.getLogger().handlers:
        return

    target_path = Path(log_path) if log_path else _default_log_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "detailed",
                "level": level,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "level": "DEBUG",
                "filename": str(target_path),
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging initialized. Platform=%s %s | Python=%s",
        platform.system(),
        platform.release(),
        sys.version.split()[0],
    )
