"""The training budget every baseline shares, per task.

Epochs and patience are *protocol*, not hyperparameters. When one model trains for 1000
epochs with patience 100 and another for 50 with patience 50, the results table compares
stopping budgets as much as it compares models, and the model that lost may simply have
been stopped sooner. They are therefore fixed per task here, deliberately absent from
every adapter's search space, and recorded on each run record so a reader can check that
the row they are looking at got the same budget as its neighbours.

The budget was previously a literal in each adapter's call to
:mod:`src.baselines.training`, copied from whatever the authors used. That is how ACM
node classification ended up giving H2SGNN 1000 epochs with patience 100 and GCN 300 with
patience 30, and how GCN/GAT graph classification ended up on 100/30 against everyone
else's 300/100.

A *published-parameter* run is the one case that must not be normalised: reproducing an
author's reported number means using the author's own budget, and
``configs/baselines/*/*/*.yaml`` carry ``epochs``/``early_stopping`` for exactly that
reason. :func:`resolve_budget` therefore honours an explicit value in the parameter set
and falls back to the shared budget otherwise -- which is what a tuned run always gets,
since neither key appears in any search space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    BaselineParams,
)

#: Epoch budget per task, shared by every model in that task's table.
TASK_EPOCHS: dict[str, int] = {
    NODE_CLASSIFICATION: 300,
    LINK_PREDICTION: 200,
    GRAPH_CLASSIFICATION: 300,
}

#: Early-stopping patience, shared by every model and every task. The manuscript's
#: experimental protocol states a patience of 100 epochs; this is that number, in the one
#: place that can make the statement true.
SHARED_PATIENCE = 100


@dataclass(frozen=True, slots=True)
class TrainingBudget:
    """How long one baseline is allowed to train, and how long it may stall."""

    epochs: int
    patience: int

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}.")


def default_budget(task: str) -> TrainingBudget:
    """The shared budget for ``task``, before any published-parameter override."""

    if task not in TASK_EPOCHS:
        raise ValueError(f"Unsupported task {task!r}.")
    return TrainingBudget(epochs=TASK_EPOCHS[task], patience=SHARED_PATIENCE)


def resolve_budget(task: str, params: BaselineParams) -> TrainingBudget:
    """The budget this run trains under.

    Published parameter sets carry the authors' own ``epochs``/``early_stopping`` and keep
    them. Tuned runs carry neither key -- they are not in any search space -- and get the
    shared budget.
    """

    budget = default_budget(task)
    return TrainingBudget(
        epochs=int(params.get("epochs", budget.epochs)),
        patience=int(params.get("early_stopping", budget.patience)),
    )


def budget_kwargs(task: str, params: BaselineParams) -> dict[str, Any]:
    """``epochs``/``patience``/``report`` for a :mod:`src.baselines.training` loop.

    Passed as ``**budget_kwargs(TASK, params)`` at every call site so that the three
    values a fair comparison requires to be identical cannot be set per adapter, and so
    that adding a fourth later does not mean revisiting thirty call sites.
    """

    budget = resolve_budget(task, params)
    return {
        "epochs": budget.epochs,
        "patience": budget.patience,
        "report": params.report,
    }


def record_budget(record: dict[str, float], task: str, params: BaselineParams) -> None:
    """Stamp the budget onto a finished run record, in place.

    ``budget_source`` says why this row trained for this long: ``"published"`` (the
    authors' own budget, reproduced), ``"campaign_cli"`` (a campaign-wide ``--epochs``
    or ``--patience``, applied to every model in the run), or ``"campaign_shared"`` (the
    task default). Two rows of the same table with different ``budget_epochs`` are not
    comparable, and this is what makes that visible without re-reading the code.
    """

    budget = resolve_budget(task, params)
    overridden = "epochs" in params.values or "early_stopping" in params.values
    record["budget_epochs"] = float(budget.epochs)
    record["budget_patience"] = float(budget.patience)
    if not overridden:
        record["budget_source"] = "campaign_shared"
    else:
        record["budget_source"] = (
            "published" if params.source == "published" else "campaign_cli"
        )


__all__ = [
    "SHARED_PATIENCE",
    "TASK_EPOCHS",
    "TrainingBudget",
    "budget_kwargs",
    "default_budget",
    "record_budget",
    "resolve_budget",
]
