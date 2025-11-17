# app/core/logging.py
from __future__ import annotations

import logging
import os
import sys
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict
from app.core.config import settings

_LOGGING_CONFIGURED = False  # guard to avoid duplicate setup

class JSONFormatter(logging.Formatter):
    """Custom JSON Formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "extra_data"):
            log_data["extra_data"] = record.extra_data
        return json.dumps(log_data, ensure_ascii=False)

def _resolve_level(default: str = "INFO") -> int:
    level_name = str(getattr(settings, "LOG_LEVEL", default)).upper()
    return getattr(logging, level_name, logging.INFO)

def setup_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    # Resolve file path and ensure parent dir exists
    log_path = Path(str(settings.LOG_FILE_PATH)).expanduser()
    # If someone passed a directory, write app.log inside it
    if log_path.suffix == "":
        log_path = log_path / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(_resolve_level())

    # Start clean
    root.handlers.clear()

    # Formatter
    formatter = JSONFormatter() if getattr(settings, "LOG_FORMAT", "text") == "json" \
        else logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_resolve_level())
    console.setFormatter(formatter)
    root.addHandler(console)

    # File (with rotation)
    file_handler: RotatingFileHandler | None = None
    try:
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=int(getattr(settings, "LOG_MAX_SIZE", 10_000_000)),
            backupCount=int(getattr(settings, "LOG_BACKUP_COUNT", 5)),
            encoding="utf-8",
            delay=True,  # open lazily to reduce Windows lock contention
        )
    except PermissionError:
        # Fallback to temp if denied
        alt = Path(os.getenv("TEMP", os.getenv("TMP", "."))) / log_path.name
        try:
            file_handler = RotatingFileHandler(
                filename=str(alt),
                maxBytes=int(getattr(settings, "LOG_MAX_SIZE", 10_000_000)),
                backupCount=int(getattr(settings, "LOG_BACKUP_COUNT", 5)),
                encoding="utf-8",
                delay=True,
            )
            print(f"⚠️  Permission denied for {log_path}. Logging to {alt} instead.")
        except Exception as e:
            print(f"⚠️  File logging disabled (cannot open {log_path} or temp fallback): {e}")
            file_handler = None

    if file_handler:
        file_handler.setLevel(_resolve_level())
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
