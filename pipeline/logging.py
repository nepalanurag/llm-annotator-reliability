"""Structured logging for the pipeline. JSON to stderr by default."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(format: str = "json", level: str = "INFO") -> None:
    """Configure structlog. Call once at CLI startup.

    Args:
        format: "json" (default, for log aggregation) or "console" (local dev).
        level: standard logging level name.
    """
    if format not in ("json", "console"):
        raise ValueError(f"log format must be 'json' or 'console', got {format!r}")
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level: {level!r}")

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
