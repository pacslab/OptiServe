"""Logging for OptiServe.

Follows the standard library guidance for libraries: we attach a
:class:`~logging.NullHandler` to the package-root logger and do **not** call
``basicConfig`` at import time, so importing OptiServe never mutates the root
logger or emits output. Applications opt in to output via
:func:`configure_logging` (or their own logging setup).

Modules should obtain a namespaced logger with ``get_logger(__name__)`` so log
records are attributable to their source module.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "optiserve"

# Library best practice: a NullHandler on the package root prevents
# "No handlers could be found" warnings without imposing any configuration.
logging.getLogger(_ROOT_NAME).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under ``optiserve``.

    ``get_logger(__name__)`` yields e.g. ``optiserve.profiling.sampler``.
    ``get_logger()`` returns the package-root logger.
    """
    if name is None or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def configure_logging(level: int = logging.INFO, *, fmt: str | None = None) -> None:
    """Convenience for scripts/notebooks: attach a StreamHandler to the
    ``optiserve`` logger at ``level``. Idempotent — safe to call repeatedly."""
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.NullHandler
        ):
            return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


# Package-wide default logger (backward-compatible export).
logger = get_logger()
