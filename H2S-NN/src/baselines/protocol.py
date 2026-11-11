"""The contract every baseline adapter implements.

Task policy, fixed for the whole campaign:

* **Node classification** runs the authors' own model *including their head*. That is what
  "running the baseline" means, and it is the only way the published-hyperparameter cells
  are meaningful.
* **Link prediction** and **graph classification** use the authors' encoder plus a head we
  supply (see :mod:`src.baselines.heads`), trained end to end. Most of these models never
  implemented those tasks, so without a shared head the comparison would measure whoever
  wrote the better decoder rather than the encoder.

Every adapter implements the single entry point :meth:`BaselineAdapter.fit_task` and owns
its own training loop. An earlier design exposed a separate ``node_embeddings`` hook and
let this module drive training, but some adapters do not factor that way -- SeHGNN
precomputes meta-path features before any model exists (and LatGRL, removed in spec v1.15,
trained an unsupervised encoder before a probe) -- so the shared loop would have
been bypassed by exactly the cases it was meant to unify. Attaching the shared head is
therefore the adapter's job on LP and GC; :mod:`src.baselines.heads` supplies it so every
adapter attaches the *same* one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from src.baselines.progress import EpochReport

# Task identifiers, matching src/configs/config.py: DATASET_TASK_TYPES.
NODE_CLASSIFICATION = "node_classification"
LINK_PREDICTION = "link_prediction"
GRAPH_CLASSIFICATION = "graph_classification"
TASKS = (NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION)


@dataclass(slots=True)
class BaselineParams:
    """One resolved hyperparameter assignment for a single (model, task, dataset) cell.

    ``source`` records where the values came from so the results table can state which
    cells inherited published hyperparameters and which were searched:
    ``"published"`` (transcribed from the authors, see configs/baselines/) or
    ``"tuned"`` (selected on validation by a W&B sweep).

    ``report`` is not a hyperparameter and never enters ``values``: it is the per-epoch
    callback the sweep worker installs so Hyperband can see a trial before it finishes
    (see :mod:`src.baselines.progress`). It rides along here because this object is
    already threaded from :func:`src.baselines.campaign.fit_cell` down to every training
    loop, which is exactly the path the callback needs, and it is ``None`` outside a
    sweep.
    """

    values: dict[str, Any] = field(default_factory=dict)
    source: str = "tuned"
    report: EpochReport | None = None

    def __post_init__(self) -> None:
        if self.source not in {"published", "tuned"}:
            raise ValueError(
                f"BaselineParams.source must be 'published' or 'tuned', got {self.source!r}."
            )

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise KeyError(
                f"Required hyperparameter {key!r} is missing from this {self.source} "
                "parameter set."
            )
        return self.values[key]


@runtime_checkable
class BaselineAdapter(Protocol):
    """Bridge between one vendored baseline repository and the shared campaign."""

    #: Registry key, e.g. ``"gcn"``. Also used in experiment and W&B run names.
    name: str

    def supports(self, task: str, dataset: str) -> bool:
        """Whether this cell is runnable at all.

        A ``False`` here becomes an explicit em-dash in the results table, never a silent
        omission (e.g. a dataset whose data are pending).
        """
        ...

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        """W&B ``parameters`` block for this cell, in shared-backbone form."""
        ...

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        """Authors' hyperparameters for this cell, or ``None`` if they published none."""
        ...

    def fit_task(
        self,
        bundle: Any,
        params: BaselineParams,
        *,
        task: str,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        """Train and evaluate one cell, returning the metric record for the run report.

        Keys must match ``src.utils.run_reports.report_metrics_for_task`` for the task,
        plus the validation metric named by ``default_monitor_for_task``.
        """
        ...


class UnsupportedCell(RuntimeError):
    """Raised when a (model, task, dataset) cell is requested but not runnable.

    Carries the reason so the results table footnote and the error message stay in sync.
    """

    def __init__(self, model: str, task: str, dataset: str, reason: str) -> None:
        super().__init__(f"{model} does not support {task} on {dataset}: {reason}")
        self.model = model
        self.task = task
        self.dataset = dataset
        self.reason = reason
