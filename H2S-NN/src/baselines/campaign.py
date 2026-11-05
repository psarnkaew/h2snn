"""Running baseline cells: unit enumeration, sweeps, and multi-seed repeats.

The two entry scripts (``scripts/run_baseline_tuning.py`` and
``scripts/run_baseline_published.py``) are argument parsing and object construction only;
everything they do lives here, and the W&B orchestration itself is
:mod:`src.utils.wandb_campaign`, shared with the H2SNN campaign.

**A unit** is one tunable slot. What that means depends on the task:

**A unit** is one seed, for all three tasks. Five units per cell.

Graph classification follows HINPool (ref/HINPOOL.pdf, hinpool/utils/utils.py:18-48):
a stratified 80/10/10 holdout redrawn per seed, rather than cross-validation. Each unit
is therefore a different stratified split *and* a different initialisation, which is
what their 'varied stratified splits and random seeds' describes.

Hyperparameters are selected per unit. At five units that is five sweeps per cell,
which is affordable; the repeated-CV script remains available as a secondary protocol.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.baselines.budget import TASK_EPOCHS, record_budget
from src.baselines.link_audit import record_link_split_overlap
from src.baselines.progress import EpochReport
from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    BaselineParams,
)
from src.baselines.registry import get_adapter, unsupported_reason
from src.baselines.runtime.dependencies import dgl_cuda_report, groups_for
from src.utils.run_reports import default_monitor_for_task
from src.utils.wandb_campaign import CampaignUnit

#: Units per cell, for every task. Mirrors the dataset presets; kept here so unit
#: enumeration does not need to build a config just to count units.
DEFAULT_NUM_RUNS = 5


@dataclass(frozen=True, slots=True)
class Cell:
    """One (model, task, dataset) entry of the results table."""

    model: str
    task: str
    dataset: str

    @property
    def token(self) -> str:
        return f"{self.model}_{self.task}_{self.dataset}"


def build_units(
    task: str, *, base_seed: int = 42, num_runs: int = DEFAULT_NUM_RUNS
) -> list[CampaignUnit]:
    """Enumerate the tunable slots for one task: one per seed.

    Graph classification redraws its stratified 80/10/10 split from the seed, so a unit
    varies both the split and the initialisation there; node classification and link
    prediction hold a fixed benchmark split and vary only the initialisation.
    """

    return [
        CampaignUnit(
            run_index=index,
            num_runs=num_runs,
            seed=base_seed + index,
            label="run",
        )
        for index in range(num_runs)
    ]


def build_raw_base(
    cell: Cell,
    unit: CampaignUnit,
    *,
    device: str | None = None,
    epochs: int | None = None,
    patience: int | None = None,
) -> dict[str, Any]:
    """The JSON-serialisable description a worker needs to rebuild this trial.

    Deliberately small: model name, cell coordinates, the unit's split position, and the
    campaign-level execution settings. The bundle is rebuilt from it in the worker rather
    than pickled across the process boundary.

    ``device``, ``epochs``, and ``patience`` are the CLI's, and they must travel in here
    to have any effect: the worker is a *fresh process* that reads this payload out of a
    context file and never sees the parent's ``argparse.Namespace``. Leaving them out is
    how ``--device cpu`` came to be parsed and then ignored while the worker picked CUDA
    on its own, and how ``--epochs`` came to size the Hyperband bracket without changing
    how long anything actually trained.

    They are campaign-wide, applying to every model in the run, so an override cannot
    reintroduce the per-adapter budget asymmetry that :mod:`src.baselines.budget` exists
    to prevent.
    """

    payload: dict[str, Any] = {
        "model": cell.model,
        "task": cell.task,
        "dataset": cell.dataset,
        "seed": unit.seed,
        "run_index": unit.run_index,
    }
    if device is not None:
        payload["device"] = device
    if epochs is not None:
        payload["epochs"] = int(epochs)
    if patience is not None:
        payload["patience"] = int(patience)
    payload.update(unit.metadata)
    return payload


def resolve_device(raw_base: dict[str, Any]) -> Any:
    """The device this trial trains on: the campaign's ``--device``, else CUDA if present."""

    import torch

    requested = raw_base.get("device")
    if requested:
        return torch.device(str(requested))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def budget_overrides(raw_base: dict[str, Any]) -> dict[str, Any]:
    """Campaign-level ``--epochs``/``--patience``, in the key names the budget reads."""

    overrides: dict[str, Any] = {}
    if raw_base.get("epochs") is not None:
        overrides["epochs"] = int(raw_base["epochs"])
    if raw_base.get("patience") is not None:
        overrides["early_stopping"] = int(raw_base["patience"])
    return overrides


def bundle_overrides(raw_base: dict[str, Any]) -> dict[str, Any]:
    """Extra config overrides for this unit.

    None are needed: every task derives its split position from ``run_index`` and
    ``seed``, which :func:`src.baselines.bundles.build_bundle` already takes directly.
    Kept as a seam because a secondary protocol (the repeated-CV script) does need to
    pin ``repetition_index``/``fold_index``.
    """

    return {}


def build_sweep_config(
    cell: Cell,
    *,
    monitor: str,
    n_trials: int,
    epochs: int,
    hyperband_min_iter: int = 20,
    hyperband_eta: int = 3,
) -> dict[str, Any]:
    """Bayes + Hyperband over the adapter's own search space.

    Same shape as the H2SNN campaign's sweeps, so the two sets of results are produced
    under the same optimiser and pruning rules.
    """

    if epochs <= hyperband_min_iter:
        raise ValueError(
            "epochs must exceed hyperband_min_iter so Hyperband has a valid stopping "
            f"interval; got {epochs} <= {hyperband_min_iter}."
        )
    adapter = get_adapter(cell.model)
    parameters = copy.deepcopy(adapter.search_space(cell.task, cell.dataset))
    if not parameters:
        raise ValueError(f"{cell.token}: adapter returned an empty search space.")

    return {
        "name": f"{cell.token}_sweep",
        "method": "bayes",
        "metric": {"name": f"best/{monitor.removeprefix('val_')}", "goal": "maximize"},
        "parameters": parameters,
        "early_terminate": {
            "type": "hyperband",
            "min_iter": hyperband_min_iter,
            "eta": hyperband_eta,
            "max_iter": epochs,
        },
    }


def fit_cell(
    raw_base: dict[str, Any],
    params: dict[str, Any],
    *,
    source: str,
    device: Any,
    report: EpochReport | None = None,
) -> dict[str, Any]:
    """Build the bundle for one unit and run the adapter on it.

    The single place a baseline is actually fitted, shared by the sweep worker and the
    published-parameter runner so the two cannot drift apart.

    ``report`` is the sweep's per-epoch callback (:mod:`src.baselines.progress`); the
    published runner passes ``None`` and trains silently.
    """

    from src.baselines.bundles import build_bundle

    cell = Cell(
        model=str(raw_base["model"]),
        task=str(raw_base["task"]),
        dataset=str(raw_base["dataset"]),
    )
    reason = unsupported_reason(cell.model, cell.task, cell.dataset)
    if reason:
        raise ValueError(f"{cell.token} is not a runnable cell: {reason}")

    seed = int(raw_base["seed"])
    bundle = build_bundle(
        cell.task,
        cell.dataset,
        run_index=int(raw_base.get("run_index", 0)),
        seed=seed,
        overrides=bundle_overrides(raw_base) or None,
    )
    adapter = get_adapter(cell.model)
    values = dict(params)
    # A published parameter set already carries the authors' own budget; overriding it
    # from the CLI would silently stop reproducing the number it exists to reproduce.
    if source != "published":
        values.update(budget_overrides(raw_base))
    baseline_params = BaselineParams(values=values, source=source, report=report)
    record = adapter.fit_task(
        bundle,
        baseline_params,
        task=cell.task,
        seed=seed,
        device=device,
    )
    record.setdefault("model", cell.model)
    record.setdefault("dataset", cell.dataset)
    record.setdefault("task", cell.task)
    # The DGL-backed adapters route their device through `resolve_dgl_device`, which
    # returns CPU without raising when the installed DGL has no CUDA support. The run
    # then produces correct numbers on the wrong hardware, so the record has to say
    # which: a row that trained on CPU is not comparable on runtime to one beside it
    # that did not, and nothing else in the record distinguishes the two.
    record["device_requested"] = getattr(device, "type", str(device))
    if "dgl" in groups_for(cell.model, cell.task):
        record["dgl_cuda"] = float(dgl_cuda_report()[0])
    # Stamped centrally so every row of every table carries the budget it trained under,
    # and a reader can check that two rows in the same table got the same one.
    record_budget(record, cell.task, baseline_params)
    if cell.task == LINK_PREDICTION:
        # HGB's split leaves part of each held-out set visible to the encoder; measured
        # here so the number is on the record rather than left to be discovered later.
        record_link_split_overlap(record, bundle.data)
    return record


def monitor_for(task: str) -> str:
    """``macro_f1`` for node and graph classification, ``auroc`` for link prediction."""

    return default_monitor_for_task(task)


def default_epochs(task: str) -> int:
    """Epoch budget per task, from the single budget table every adapter also reads.

    Used here only to give Hyperband its ``max_iter``; the value that actually reaches a
    training loop comes from :func:`src.baselines.budget.resolve_budget`. Reading both
    from :data:`src.baselines.budget.TASK_EPOCHS` is what keeps the pruning bracket and
    the real epoch count from drifting apart.
    """

    if task not in TASK_EPOCHS:
        raise ValueError(f"Unsupported task {task!r}.")
    return TASK_EPOCHS[task]


def summarize_cell(
    records: list[dict[str, Any]], task: str, path: Path
) -> dict[str, Any]:
    """Aggregate one cell's unit records and write them alongside the summary.

    Mean and standard deviation over the five runs, for every task. Graph
    classification reported two standard deviations while it used repeated
    cross-validation; under HINPool's holdout there is no repetition structure to
    average over, so there is one.

    Graph classification reports **two** standard deviations, as TUDataset does
    (``gnn_evaluation.py`` returns ``s_1`` and ``s_2``): one over the five
    repetition-means, and one over all fifty individual runs. They answer different
    questions -- how much the protocol's outcome moves if you rerun it, versus how much a
    single fold varies -- and quoting only the first understates the spread. The other two
    tasks have no repetition structure, so they report the plain mean and standard
    deviation over seeds.
    """

    import json
    from statistics import mean, stdev

    from src.utils.run_reports import report_metrics_for_task

    metrics = report_metrics_for_task(task)
    summary: dict[str, Any] = {
        "task": task,
        "num_units": len(records),
        "metrics": {},
    }

    for metric in metrics:
        values = [float(record.get(metric, 0.0)) for record in records]
        summary["metrics"][metric] = {
            "mean": mean(values) if values else 0.0,
            "std": stdev(values) if len(values) > 1 else 0.0,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "units": records}, handle, indent=2)
    return summary


def state_path_for(state_dir: Path, cell: Cell, unit: CampaignUnit) -> Path:
    return state_dir / cell.token / f"tuning_state_{unit.token}.json"


__all__ = [
    "Cell",
    "DEFAULT_NUM_RUNS",
    "budget_overrides",
    "build_raw_base",
    "build_sweep_config",
    "build_units",
    "bundle_overrides",
    "default_epochs",
    "fit_cell",
    "monitor_for",
    "resolve_device",
    "state_path_for",
    "summarize_cell",
]
