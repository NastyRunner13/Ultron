"""Ultron logging — Loguru + Rich integration."""

from __future__ import annotations

import sys

from loguru import logger
from rich.console import Console

from ultron.core.settings import get_settings

# ── Rich console (shared across the project) ─────────────────────────────────
console = Console(highlight=True, markup=True)

_CONFIGURED = False


def setup_logging() -> None:
    """Configure Loguru with Rich-formatted output.

    Call once at application startup. Safe to call multiple times
    (subsequent calls are no-ops).
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    settings = get_settings()

    # Remove default Loguru handler
    logger.remove()

    # Add Rich-styled stderr handler
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Optional file handler
    if settings.log_file:
        logger.add(
            settings.log_file,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        )

    _CONFIGURED = True
    logger.debug("Logging configured — level={}", settings.log_level)
