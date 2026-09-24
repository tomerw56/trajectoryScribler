from __future__ import annotations

import logging

from trajectory_app.logging_config import configure_logging, get_logger


def test_logging_writes_to_local_file(tmp_path):
    log_file = configure_logging(
        log_dir=tmp_path,
        level=logging.DEBUG,
        max_bytes=1024 * 1024,
        backup_count=1,
    )

    logger = get_logger("test")
    logger.info("hello-file-logger")

    for handler in logging.getLogger("trajectory_scribbler").handlers:
        handler.flush()

    assert log_file.exists()
    text = log_file.read_text(encoding="utf-8")
    assert "hello-file-logger" in text


def test_configure_logging_does_not_duplicate_owned_handlers(tmp_path):
    configure_logging(log_dir=tmp_path)
    configure_logging(log_dir=tmp_path)

    handlers = [
        handler
        for handler in logging.getLogger("trajectory_scribbler").handlers
        if getattr(handler, "_trajectory_scribbler_handler", False)
    ]
    assert len(handlers) == 2
