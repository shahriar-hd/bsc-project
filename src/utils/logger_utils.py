"""
Shared logging utilities for all entry points (train, preprocessing, demo).

- setup_logger(log_path, cfg)  → console + file logger, levels from config
- log_banner(logger, text)     → section banner

train.py used to own these; they are now shared because preprocessing and
the demo also log to a file during long runs (and thesis experiments should
be able to point all three at the same logging config).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import Config, LogConfig


def _resolve_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def setup_logger(
    log_path: Optional[str] = None,
    cfg: Optional[Config] = None,
    name: str = "MTL",
) -> logging.Logger:
    """
    Configure a console + file logger and return it.

    Args:
        log_path: Absolute path of the log file. If None, only console output.
        cfg:      Optional Config; its `log` section controls format/levels.
        name:     Logger name. Callers that pass the same name get the same
                  instance back (logging caches by name), so repeated calls
                  must not add duplicate handlers — hence the guard below.

    Returns:
        Configured logger.
    """
    lc: LogConfig = (cfg.log if cfg is not None else LogConfig())
    fmt = logging.Formatter(fmt=lc.fmt, datefmt=lc.datefmt)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # avoid duplicate output via the root logger

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(_resolve_level(lc.console_level))
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_path is not None and not any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    ):
        from pathlib import Path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(_resolve_level(lc.file_level))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_banner(logger: logging.Logger, text: str, width: int = 60) -> None:
    """Print a section banner to logger."""
    sep = "─" * width
    logger.info(sep)
    logger.info(f"  {text}")
    logger.info(sep)
