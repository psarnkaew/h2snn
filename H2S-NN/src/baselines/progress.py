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
    """

    def __call__(
        self,
        *,
        epoch: int,
        monitor: float,
        running_best: float,
        metrics: dict[str, float],
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
    ) -> None:
        if self._wandb.run is None:
            return
        payload: dict[str, Any] = {
            self._best_key: running_best,
            self._current_key: monitor,
            "epoch": epoch,
        }
        payload.update({f"val/{name}": value for name, value in metrics.items()})
        self._wandb.log(payload, step=epoch)


def report_epoch(
    report: EpochReport | None,
    *,
    epoch: int,
    monitor: float,
    running_best: float,
    metrics: dict[str, float],
) -> None:
    """Call ``report`` if there is one. Keeps the null check out of the training loops."""

    if report is None:
        return
    report(
        epoch=epoch, monitor=monitor, running_best=running_best, metrics=metrics
    )


__all__ = ["EpochReport", "WandbEpochReport", "report_epoch"]
