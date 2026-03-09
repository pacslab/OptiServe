"""Backward-compatible shim. The logging implementation now lives in
``optiserve.logging``; prefer ``from optiserve.logging import get_logger``.
This module is retained so existing ``from optiserve.utils.logger import logger``
imports keep working while modules migrate."""
from optiserve.logging import configure_logging, get_logger, logger

__all__ = ["logger", "get_logger", "configure_logging"]
