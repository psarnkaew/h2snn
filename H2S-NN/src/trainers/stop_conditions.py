"""Logging of the stop conditions of spec v1.15 §6 (J10) in every trainer (card Part M, M6).

The exception is never caught for good: it is logged with its condition, value and epoch, then
re-raised, so the run stops as at `T_max`."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from src.models.anchored_solve import StopConditionError


@contextmanager
def logged_stop_conditions(logger) -> Iterator[None]:
    try:
        yield
    except StopConditionError as exc:
        if logger is not None:
            logger.info("STOP CONDITION %s: %.6g > %g at epoch %d%s (spec v1.15 §6); run stopped.",
                        exc.condition, exc.value, exc.limit, exc.epoch,
                        " (evaluation forward)" if exc.epoch < 0 else "")
        raise


__all__ = ["logged_stop_conditions"]
