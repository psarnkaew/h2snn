"""Logging of the stop conditions of spec v1.15 §6 (J10) in every trainer (card Part M, M6).

The exception is never caught for good: it is logged with its condition, value and epoch, then
re-raised, so the run stops as at `T_max`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from src.models.anchored_solve import PCGConvergenceError, StopConditionError


@contextmanager
def logged_stop_conditions(logger) -> Iterator[None]:
    try:
        yield
    except (StopConditionError, PCGConvergenceError) as exc:
        if logger is not None:
            epoch = getattr(exc, "epoch", None)
            value, limit = getattr(exc, "value", None), getattr(exc, "limit", None)
            logger.info("STOP CONDITION %s: %s > %s at epoch %s%s (spec v1.16 §6); run stopped.",
                        getattr(exc, "condition", type(exc).__name__),
                        "?" if value is None else f"{value:.6g}",
                        "?" if limit is None else f"{limit:g}", epoch,
                        " (evaluation forward)" if epoch is not None and epoch < 0 else "")
        raise


__all__ = ["logged_stop_conditions"]
