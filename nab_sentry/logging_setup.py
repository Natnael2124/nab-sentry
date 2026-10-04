"""Logging to the console and ``data/logs/nab_sentry.log`` (Requirement 13.7)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "nab_sentry"
LOG_FILE_NAME = "nab_sentry.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_HANDLER_TAG = "_nab_sentry_handler"


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure the ``nab_sentry`` logger with a console and a file handler.

    Idempotent: calling it again replaces the handlers it installed earlier (for
    example when ``logs_dir`` changes) instead of stacking duplicates.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    for h in list(logger.handlers):
        if getattr(h, _HANDLER_TAG, False):
            logger.removeHandler(h)
            h.close()

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(logs_dir / LOG_FILE_NAME, encoding="utf-8")
    for h in (console, file_handler):
        h.setFormatter(formatter)
        h.setLevel(level)
        setattr(h, _HANDLER_TAG, True)
        logger.addHandler(h)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Child logger of ``nab_sentry`` (e.g. ``get_logger("ingest")`` -> ``nab_sentry.ingest``)."""
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")
