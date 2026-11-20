"""Per-epoch reporting from a baseline's training loop to the sweep that launched it.

Hyperband prunes a trial by comparing its metric against the other trials' metrics *at
the same step*, so it needs a live series, logged while the fit is still running. The
baseline campaign used to log once, after the fit returned, which left the
``early_terminate`` block in :func:`src.baselines.campaign.build_sweep_config` declaring
a pruning rule that could never fire: every trial ran to completion, whatever it looked
like at epoch 20.

H2SNN's own sweeps do not have that problem -- :class:`src.utils.logger.ExperimentLogger`
logs every epoch -- so the two halves of the results table were searched under different
rules. This module closes that gap: the shared training loops call a report each epoch,
and the sweep worker supplies one that logs under the sweep's own metric name.

The report is deliberately a plain callable rather than a logger object. Training loops
should not be able to reach into a W&B run, and the published-parameter runner, the unit
tests, and anything else running outside a sweep pass ``None`` and log nothing.
"""

from __future__ import annotations

from typing import Any, Protocol


class EpochReport(Protocol):
    """Called once per epoch, after that epoch's validation metrics exist.

    ``running_best`` is the best value of the selection monitor seen so far, which is
    what a pruner should compare across trials -- the current epoch's value alone is
    noisy enough that a good trial can be killed for one bad epoch.

    Card 8 C7 (amendment 6): ``extra`` carries the epoch's other logged values under their
    final keys (``train/loss``, ``val/loss``, ``test/<metric>``); ``prefix`` names the
    validation metrics (``val``, or ``val_subset`` for the B1a subset on H2GB). Both are
    logged only: no selection or early-stopping code reads them.
    """

    def __call__(
        self,
        *,
        epoch: int,
        monitor: float,
        running_best: float,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
        prefix: str = "val",
    ) -> None: ...


class WandbEpochReport:
    """Log each epoch to the active W&B run under the sweep's metric name.

    The key must match ``sweep_config["metric"]["name"]`` exactly or Hyperband has
    nothing to read, which is why the monitor name is passed in rather than guessed.
    """

    __slots__ = ("_best_key", "_current_key", "_wandb")

    def __init__(self, monitor: str) -> None:
        import wandb  # type: ignore[import-not-found]

        self._wandb = wandb
        key = monitor.removeprefix("val_")
        # Mirrors src/utils/wandb_campaign.py's end-of-trial summary key, so the live
        # series and the final value are one continuous curve rather than two.
        self._best_key = f"best/{key}"
        self._current_key = f"val/{key}"

    def __call__(
        self,
        *,
        epoch: int,
        monitor: float,
        running_best: float,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
        prefix: str = "val",
    ) -> None:
        if self._wandb.run is None:
            return
        payload = epoch_payload(self._best_key, epoch=epoch, monitor=monitor, running_best=running_best,
                                metrics=metrics, extra=extra, prefix=prefix)
        try:  # W&B never fails a run (card 8 C7)
            self._wandb.log(payload, step=epoch)
        except Exception as exc:  # noqa: BLE001
            print(f"wandb.log failed at epoch {epoch}: {type(exc).__name__}: {exc}", flush=True)


def epoch_payload(best_key: str, *, epoch: int, monitor: float, running_best: float, metrics: dict[str, float],
                  extra: dict[str, Any] | None = None, prefix: str = "val") -> dict[str, Any]:
    """The W&B payload of one epoch: `best/<monitor>` (running best), `<prefix>/<metric>` for every validation
    metric, and `extra` (card 8 C7: `train/loss`, `val/loss`, `test/<metric>`) with None values dropped."""
    key = best_key.removeprefix("best/")
    payload: dict[str, Any] = {best_key: running_best, f"{prefix}/{key}": monitor, "epoch": epoch}
    payload.update({f"{prefix}/{name}": value for name, value in metrics.items()})
    payload.update({k: v for k, v in (extra or {}).items() if v is not None})
    return payload


def report_epoch(
    report: EpochReport | None,
    *,
    epoch: int,
    monitor: float,
    running_best: float,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
    prefix: str = "val",
) -> None:
    """Call ``report`` if there is one. Keeps the null check out of the training loops. ``extra`` and ``prefix``
    (card 8 C7) are passed only when given, so a report written for the old call keeps working."""

    if report is None:
        return
    kwargs: dict[str, Any] = {}
    if extra is not None and _accepts(report, "extra"):
        kwargs["extra"] = extra
    if prefix != "val" and _accepts(report, "prefix"):
        kwargs["prefix"] = prefix
    report(epoch=epoch, monitor=monitor, running_best=running_best, metrics=metrics, **kwargs)


def _accepts(report: Any, name: str) -> bool:
    """Whether `report` takes keyword `name` (or `**kwargs`): a report written for the pre-C7 call (e.g. a test's
    `report(*, epoch, monitor, running_best, metrics)`) keeps receiving exactly those four (card 8 C7)."""
    import inspect

    try:
        parameters = inspect.signature(report).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD or p.name == name for p in parameters)


def prefixed_metrics(prefix: str, metrics: dict[str, Any] | None) -> dict[str, Any]:
    """`{f"{prefix}/{name}": value}` of a metric dict (the `test/*` part of an epoch's `extra`)."""
    return {f"{prefix}/{name}": value for name, value in (metrics or {}).items()}


__all__ = ["EpochReport", "WandbEpochReport", "epoch_payload", "prefixed_metrics", "report_epoch"]
