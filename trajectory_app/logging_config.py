from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_ROOT_NAME = "trajectory_scribbler"


def configure_logging(
    *,
    log_dir: str | Path = "logs",
    level: int = logging.DEBUG,
    max_bytes: int = 2_000_000,
    backup_count: int = 5,
) -> Path:
    """
    Configure application logging to both stdout and a local rotating file.

    Calling this function more than once is safe: handlers owned by this
    configuration are replaced instead of duplicated.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "trajectory_scribbler.log"

    app_logger = logging.getLogger(_LOGGER_ROOT_NAME)
    app_logger.setLevel(level)
    app_logger.propagate = False

    # Avoid duplicate output if run_app() is called more than once in tests.
    for handler in list(app_logger.handlers):
        if getattr(handler, "_trajectory_scribbler_handler", False):
            app_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    stdout_handler._trajectory_scribbler_handler = True  # type: ignore[attr-defined]

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler._trajectory_scribbler_handler = True  # type: ignore[attr-defined]

    app_logger.addHandler(stdout_handler)
    app_logger.addHandler(file_handler)

    app_logger.info("Logging initialized: %s", log_file.resolve())
    return log_file


def get_logger(name: str) -> logging.Logger:
    """
    Return an application-scoped logger.

    Example:
        logger = get_logger(__name__)
    """
    if name.startswith(_LOGGER_ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_ROOT_NAME}.{name}")
