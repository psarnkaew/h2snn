"""The campaign's cell matrix: which model runs on which task and dataset, and why not.

Every cell is either runnable by a named adapter or explicitly unsupported with a reason.
A results table renders an unsupported cell as an em-dash and the reason as a footnote;
nothing is ever silently absent.

Adapters are imported lazily so a missing vendor clone or optional dependency breaks only
the cells that need it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    BaselineAdapter,
    UnsupportedCell,
)

DATASETS: dict[str, tuple[str, ...]] = {
    NODE_CLASSIFICATION: ("acm", "dblp", "imdb"),
    LINK_PREDICTION: ("lastfm", "youtube", "amazon", "pubmed"),
    GRAPH_CLASSIFICATION: ("mutag", "proteins", "enzymes", "nci1"),
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One baseline row: where its adapter lives and which tasks it covers."""

    name: str
    display: str
    group: str
    module: str
    tasks: tuple[str, ...]
    #: (task, dataset) -> reason, for cells this model cannot run.
    unsupported: tuple[tuple[str, str, str], ...] = ()


# Groups follow the manuscript's four-way split (homogeneous/heterogeneous x
# homophilic/heterophilic).
HOM_HOM = "Hom./hom."
HOM_HET = "Hom./het."
HET_HOM = "Het./hom."
HET_HET = "Het./het."

_ALL = (NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION)

#: Kept here rather than imported from the adapter so building the cell matrix does not
#: require importing a vendor-dependent module.
LATGRL_FEATURELESS_REASON = (
    "LatGRL builds its latent graphs from feature cosine similarity "
    "(module/graph_generating.py:30-64); this dataset has no node attributes, so that "
    "similarity is the identity and the low-pass latent graph is zeroed by the diagonal "
    "removal on the next line"
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("gcn", "GCN", HOM_HOM, "src.baselines.adapters.hgb_gnn", _ALL),
    ModelSpec("gat", "GAT", HOM_HOM, "src.baselines.adapters.hgb_gnn", _ALL),
    ModelSpec(
        "chebnetii", "ChebNetII", HOM_HOM, "src.baselines.adapters.chebnetii",
        (NODE_CLASSIFICATION,),
    ),
    ModelSpec("nsd", "NSD", HOM_HET, "src.baselines.adapters.nsd", _ALL),
    ModelSpec("csnn", "CSNN", HOM_HET, "src.baselines.adapters.csnn", _ALL),
    ModelSpec("han", "HAN", HET_HOM, "src.baselines.adapters.hgb_han", _ALL),
    ModelSpec(
        "sehgnn", "SeHGNN", HET_HOM, "src.baselines.adapters.sehgnn",
        (NODE_CLASSIFICATION,),
    ),
    ModelSpec(
        "hinormer", "HiNormer", HET_HOM, "src.baselines.adapters.hinormer",
        (NODE_CLASSIFICATION,),
    ),
    ModelSpec("pshgcn", "PSHGCN", HET_HOM, "src.baselines.adapters.pshgcn", _ALL),
    ModelSpec("h2gformer", "H2G-former", HET_HET, "src.baselines.adapters.h2gformer", _ALL),
    ModelSpec("hetero2net", "Hetero$^2$Net", HET_HET, "src.baselines.adapters.hetero2net", _ALL),
    ModelSpec("h2sgnn", "H2SGNN", HET_HET, "src.baselines.adapters.h2sgnn", _ALL),
    ModelSpec(
        "latgrl", "LatGRL", HET_HET, "src.baselines.adapters.latgrl",
        (NODE_CLASSIFICATION, LINK_PREDICTION),
        unsupported=tuple(
            (
                GRAPH_CLASSIFICATION,
                dataset,
                "unsupervised node-level objective with no graph-level signal; its dense "
                "O(N^2) latent-graph construction also does not fit the TU union "
                "(NCI1 60.3 GB, PROTEINS 7.6 GB)",
            )
            for dataset in DATASETS[GRAPH_CLASSIFICATION]
        )
        + tuple(
            (LINK_PREDICTION, dataset, LATGRL_FEATURELESS_REASON)
            # LastFM and YouTube have no node attributes at all, and LatGRL's latent
            # graphs are built from feature similarity. See _latgrl_lp.
            for dataset in ("lastfm", "youtube")
        ),
    ),
)

#: HetSheaf has no public implementation; its rows are cited from the paper's Table 2
#: rather than run. Kept here so table builders can render the row without an adapter.
CITED_ONLY: dict[str, str] = {
    "hetsheaf": "no public implementation; values cited from the original paper's Table 2",
}

_BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in MODELS}


def model_spec(name: str) -> ModelSpec:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown baseline {name!r}. Known: {', '.join(sorted(_BY_NAME))}."
        ) from exc


def models_for_task(task: str) -> tuple[ModelSpec, ...]:
    return tuple(spec for spec in MODELS if task in spec.tasks)


def unsupported_reason(name: str, task: str, dataset: str) -> str | None:
    """Return why a cell is not runnable, or ``None`` if it is."""

    spec = model_spec(name)
    if dataset not in DATASETS.get(task, ()):
        return f"{dataset} is not a {task.replace('_', ' ')} dataset in this campaign"
    # A specific reason always wins over the generic "no implementation" fallback: the
    # table footnote should say *why* LatGRL has no graph-classification cell, not just
    # that it has none.
    for entry_task, entry_dataset, reason in spec.unsupported:
        if entry_task == task and entry_dataset == dataset:
            return reason
    if task not in spec.tasks:
        return f"{spec.display} has no {task.replace('_', ' ')} implementation"
    return None


def get_adapter(name: str) -> BaselineAdapter:
    """Import and return one adapter instance."""

    spec = model_spec(name)
    module = importlib.import_module(spec.module)
    adapter = getattr(module, "ADAPTER", None)
    if adapter is None:
        raise AttributeError(f"{spec.module} does not expose an ADAPTER instance.")
    # Several adapters back more than one row (hgb_gnn serves both GCN and GAT), so the
    # registry name is authoritative rather than whatever the module happens to call it.
    if isinstance(adapter, dict):
        adapter = adapter[name]
    return adapter


def iter_cells(
    tasks: tuple[str, ...] = _ALL,
    models: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate every campaign cell with its runnable/unsupported status.

    This is what the notebooks' registry-view cell renders and what the completeness test
    walks, so the matrix has exactly one definition.
    """

    selected = models or tuple(spec.name for spec in MODELS)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for dataset in DATASETS[task]:
            for name in selected:
                spec = model_spec(name)
                if task not in spec.tasks and dataset not in DATASETS[task]:
                    continue
                reason = unsupported_reason(name, task, dataset)
                rows.append(
                    {
                        "model": name,
                        "display": spec.display,
                        "group": spec.group,
                        "task": task,
                        "dataset": dataset,
                        "runnable": reason is None,
                        "reason": reason,
                    }
                )
    return rows


def campaign_summary() -> dict[str, int]:
    """Counts used by the plan and by the notebooks' status cell."""

    cells = iter_cells()
    runnable = [cell for cell in cells if cell["runnable"]]
    return {
        "total_cells": len(cells),
        "runnable_cells": len(runnable),
        "unsupported_cells": len(cells) - len(runnable),
    }


__all__ = [
    "CITED_ONLY",
    "DATASETS",
    "MODELS",
    "ModelSpec",
    "UnsupportedCell",
    "campaign_summary",
    "get_adapter",
    "iter_cells",
    "model_spec",
    "models_for_task",
    "unsupported_reason",
]
