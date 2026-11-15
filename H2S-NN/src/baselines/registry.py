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

# Spec v1.15 §6 (card Part M, M11): NC = HGB 4 + H2GB 8; LP = the 6 local tasks + ogbn-mag P-P and
# H-Pokec U-U; GC = TU 4. YouTube and Amazon are removed (spec v1.5).
H2GB_NC: tuple[str, ...] = ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem",
                            "ieee-cis-g", "h-pokec", "pdns")
LARGE_LP: tuple[str, ...] = ("ogbn-mag_pp", "h-pokec_uu")
DERIVED_LP: tuple[str, ...] = ("acm_pa", "dblp_ap", "imdb_ma", "freebase_bb")

DATASETS: dict[str, tuple[str, ...]] = {
    NODE_CLASSIFICATION: ("acm", "dblp", "imdb", "freebase") + H2GB_NC,
    LINK_PREDICTION: DERIVED_LP + ("pubmed", "lastfm") + LARGE_LP,
    GRAPH_CLASSIFICATION: ("mutag", "proteins", "enzymes", "nci1"),
}

DATA_PENDING_REASON = "data pending (lab cluster, after mini-batch design)"
#: Part M3b B2: H2GB NC datasets with a loader (`src/data/h2gb_nc.py`) and a count record equal to H2GB's
#: Table 1 (`docs/partM3b_evidence/h2gb_nc_counts.{json,md}`); their NC cell is no longer data-pending.
H2GB_NC_LOADED: tuple[str, ...] = ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem",
                                   "ieee-cis-g", "h-pokec", "pdns")  # PDNS: card baselines-h2gb C2
#: Part M3b: a baseline runs on a loaded H2GB NC dataset only once its adapter reads the shared batch range
#: and epoch definition and consumes the H2GB bundle (card B8 decides per adapter); until then the cell says so.
H2GB_ADAPTER_PENDING_REASON = ("adapter pending: the H2GB bundle, batch range and epoch definition are not "
                               "wired into this adapter yet (Part M3b B8)")
#: Adapters wired for the loaded H2GB NC datasets (Part M3b B8).
H2GB_NC_WIRED_ADAPTERS: frozenset[str] = frozenset()
#: Part M3b B6: large LP tasks whose split and negative files exist (`data/LP/large/<task>/{split,negatives}.npz`,
#: written by `scripts/lp/build_large_lp.py`; jobs 16779 / 16780, sha256 in `meta.json` and
#: `docs/partM3b_evidence/b6_build_<task>.json`); their LP cell is no longer data-pending.
LARGE_LP_BUILT: tuple[str, ...] = ("ogbn-mag_pp", "h-pokec_uu")
#: Part M3b: a baseline runs on a large LP task only once its adapter reads the task files and the mini-batch
#: budget (8,192 positives per step, 40 epochs; card B8 decides per adapter); until then the cell says so.
LARGE_LP_ADAPTER_PENDING_REASON = ("adapter pending: the large-LP task files, mini-batch step and epoch definition "
                                   "are not wired into this adapter yet (Part M3b B8)")
#: Adapters wired for the built large LP tasks (Part M3b B8).
LARGE_LP_WIRED_ADAPTERS: frozenset[str] = frozenset()
#: Card baselines-h2gb (C0, C1, C4): why each adapter cannot yet run on the loaded H2GB NC datasets
#: (`docs/baselines_h2gb_evidence/c0_preconditions.md`; report `docs/H2S-NN_baselines_h2gb_report.md`).
_S1 = ("sampler depth undecided (card baselines-h2gb C0 stop S1: H2GB's hgt sampler runs as many iterations as its "
       "YAML list is long, not the method's layers; at depth = propagation steps HGB's GCN differs from its full-batch "
       "output under full fan-out, C1)")
H2GB_NC_ADAPTER_PENDING: dict[str, str] = {
    "gcn": f"adapter pending: {_S1}",
    "gat": f"adapter pending: {_S1}",
    "han": (f"adapter pending: {_S1}; and HAN's semantic attention averages over every node of the sampled "
            "subgraph (C0 S5: HGB model_hetero.py:28, PyG HANConv), so no sampled output equals the full-batch one"),
    "h2gformer": f"adapter pending: {_S1}",
    "nsd": f"adapter pending: class C, D1 second branch (H2GB's hgt sampler): {_S1}",
    "csnn": f"adapter pending: class C, D1 second branch (H2GB's hgt sampler): {_S1}",
    "h2sgnn": ("adapter pending: its forward propagates inside the model (H2SGNN/model.py:67-86; C0 S4), so D1 sends "
               f"it to H2GB's hgt sampler: {_S1}"),
    "pshgcn": ("adapter pending: card baselines-h2gb C0 stop S3: its large-graph scheme (PSHGCN/ogbn-mag) builds a "
               "model with MAG's node type P and input widths 128/256 hard-coded (ogbn-mag/model.py:57-66); its HGB "
               "model propagates inside the forward (hgb/model.py:45-110); reaching a batched scheme needs a model edit"),
    "hinormer": ("adapter pending: part 2 E0 stop H1: H2GB's own HINormerModel has its transformer and feature encoder "
                 "commented out (hinormer_model.py:221-222, 274-289: a linear head on the ego node after full-graph GCN "
                 "layers) and its `hin` loader yields the whole split as one step (custom_sampler.py:678-685); configs "
                 "exist for 4 of the 8 datasets; decision S2 needs revisiting"),
    "sehgnn": "adapter pending (card baselines-h2gb C4)",
    "chebnetii": "adapter pending (card baselines-h2gb C4)",
    "hetero2net": ("adapter pending: card baselines-h2gb C4 did not wire its NeighborLoader path (Hetero2Net/main.py:"
                   "147-175); on ogbn-mag/mag-year it also waits for `mag_metapath2vec_emb.zip` (author download)"),
}
#: Cells wired on the loaded H2GB NC datasets (card baselines-h2gb C4): adapter and its tests in place, no open
#: author decision.
H2GB_NC_WIRED_CELLS: frozenset[tuple[str, str]] = frozenset({
    # part 2 E1 (author decisions S1, S5): class A on H2GB's hgt sampler with H2GB's YAML fan-out
    *((m, d) for m in ("gcn", "gat", "han") for d in H2GB_NC),
    *(("h2gformer", d) for d in H2GB_NC if d != "mag-year"),
    ("sehgnn", "ieee-cis-g"),
    ("chebnetii", "ogbn-mag"), ("chebnetii", "mag-year"),
    ("chebnetii", "oag-cs"), ("chebnetii", "oag-eng"), ("chebnetii", "oag-chem"),
})
_SEHGNN_MAG = ("adapter in place, pending author decision (card baselines-h2gb C4-a): SeHGNN's own MAG branch feeds "
               "author/field/institution with random-uniform or ComplEx embeddings (SeHGNN/ogbn/utils.py:345-354), "
               "H2GB's graph carries averaged paper features")
_SEHGNN_PARALLEL = ("adapter in place, pending author decision (card baselines-h2gb C4-b): parallel relations between "
                    "one pair of node types are merged into one (SeHGNN names keys by one letter per type, "
                    "SeHGNN/ogbn/utils.py:56-63, and would keep only the last)")
_CHEB_WIDTHS = ("pending author decision (card baselines-h2gb C4-c): ChebNetII's ogb scheme takes one feature matrix "
                "(ChebNetII/ogb/processing.py:73-75); this dataset has featureless types or several feature widths")
#: Per-cell reasons on the loaded H2GB NC datasets (they win over the adapter's general reason).
H2GB_NC_CELL_REASONS: dict[tuple[str, str], str] = {
    ("h2gformer", "mag-year"): ("adapter pending: part 2 E1 stop item: H2GB's mag-year H2Gformer config enables "
                                "MetaPath2Vec, but H2GB's preprocess_Metapath has no schema for the name 'mag-year' "
                                "(encoding_generator.py:187-436; mag_dataset.py:67), so H2GB's own code cannot run it"),
    ("sehgnn", "ogbn-mag"): _SEHGNN_MAG, ("sehgnn", "mag-year"): _SEHGNN_MAG,
    **{("sehgnn", d): _SEHGNN_PARALLEL for d in ("oag-cs", "oag-eng", "oag-chem", "pdns")},
    **{("chebnetii", d): _CHEB_WIDTHS for d in ("ieee-cis-g", "h-pokec", "pdns")},
}
#: SeHGNN on H-Pokec (author, 2026-11-16; Part M3b B7).
SEHGNN_HPOKEC_NA = ("N/A: out of memory under its own code (stored propagated features 1,214-29,685 GiB at "
                    "num_hops 2-4, Part M3b B7; author decision 2026-11-16)")
#: Card baselines-h2gb C6: link-level sampling on the task's message graph and the task files are in place
#: (`src/baselines/large_lp_minibatch.py`, held-out-fact test and mutation); no LP adapter is wired yet.
_LP_S1 = ("adapter pending: link-level neighbour sampling on the message graph is in place (card baselines-h2gb C6), "
          "but its fan-out and depth are the class's, undecided (C0 stop S1)")
LARGE_LP_ADAPTER_PENDING: dict[str, str] = {
    **{name: _LP_S1 for name in ("gcn", "gat", "nsd", "csnn", "han", "h2gformer")},
    "h2sgnn": f"{_LP_S1}; H2SGNN is not decoupled (C0 S4), so D2's per-node precomputation does not apply",
    "pshgcn": ("adapter pending: D2's decoupled branch needs PSHGCN's batched scheme, which needs a model edit "
               "(card baselines-h2gb C0 stop S3)"),
    "hetero2net": ("adapter pending: card baselines-h2gb did not wire Hetero2Net's NeighborLoader path (C4); the "
                   "link-level sampler on the message graph is in place (C6)"),
}
#: Cells that no model can run yet, whatever the model: (task, dataset) -> reason.
DATASET_PENDING: dict[tuple[str, str], str] = {
    **{(NODE_CLASSIFICATION, dataset): DATA_PENDING_REASON for dataset in H2GB_NC
       if dataset not in H2GB_NC_LOADED},
    **{(LINK_PREDICTION, dataset): DATA_PENDING_REASON for dataset in LARGE_LP if dataset not in LARGE_LP_BUILT},
    # Freebase NC is loadable since Part P P3 (H2S-NN loader, bundles HGB_RAW_DIRNAME).
    # The derived LP tasks are wired into every LP adapter (Part P P2): no pending reason.
}


#: Part P P3: Freebase NC needs the authors' own Freebase preprocessing for these three (their
#: HGB data_loader branch: SeHGNN's hgb/, PSHGCN's and H2SGNN's loaders via _hgb_vendor); their
#: clones were empty where Part P ran, so it is not wired.
FREEBASE_VENDOR_REASON = (
    "Freebase NC not wired: needs the authors' own Freebase preprocessing (their HGB loader "
    "branch); the vendor clone was not available when Part P ran (UNVERIFIED whether their code "
    "has a Freebase branch)")

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
        unsupported=((NODE_CLASSIFICATION, "freebase", FREEBASE_VENDOR_REASON),
                     (NODE_CLASSIFICATION, "h-pokec", SEHGNN_HPOKEC_NA)),
    ),
    ModelSpec(
        "hinormer", "HiNormer", HET_HOM, "src.baselines.adapters.hinormer",
        (NODE_CLASSIFICATION,),
    ),
    ModelSpec("pshgcn", "PSHGCN", HET_HOM, "src.baselines.adapters.pshgcn", _ALL,
              unsupported=((NODE_CLASSIFICATION, "freebase", FREEBASE_VENDOR_REASON),)),
    ModelSpec("h2gformer", "H2G-former", HET_HET, "src.baselines.adapters.h2gformer", _ALL),
    ModelSpec("hetero2net", "Hetero$^2$Net", HET_HET, "src.baselines.adapters.hetero2net", _ALL),
    ModelSpec("h2sgnn", "H2SGNN", HET_HET, "src.baselines.adapters.h2sgnn", _ALL,
              unsupported=((NODE_CLASSIFICATION, "freebase", FREEBASE_VENDOR_REASON),)),
)

#: Baselines of the protocol whose adapter is not written yet (spec v1.15 §6: HetSheaf is rerun
#: on our pipeline; its adapter is the next card). Not cited: no number is quoted in a table.
ADAPTER_PENDING: dict[str, str] = {
    "hetsheaf": "adapter pending (next card): HetSheaf is rerun under the protocol (spec v1.15 §6)",
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
    # table footnote should say *why* a model has no cell, not just that it has none.
    for entry_task, entry_dataset, reason in spec.unsupported:
        if entry_task == task and entry_dataset == dataset:
            return reason
    if task not in spec.tasks:
        return f"{spec.display} has no {task.replace('_', ' ')} implementation"
    if (task, dataset) in DATASET_PENDING:
        return DATASET_PENDING[(task, dataset)]
    if task == NODE_CLASSIFICATION and dataset in H2GB_NC_LOADED and name not in H2GB_NC_WIRED_ADAPTERS:
        if (name, dataset) in H2GB_NC_WIRED_CELLS:
            return None
        return H2GB_NC_CELL_REASONS.get((name, dataset),
                                        H2GB_NC_ADAPTER_PENDING.get(name, H2GB_ADAPTER_PENDING_REASON))
    if task == LINK_PREDICTION and dataset in LARGE_LP_BUILT and name not in LARGE_LP_WIRED_ADAPTERS:
        return LARGE_LP_ADAPTER_PENDING.get(name, LARGE_LP_ADAPTER_PENDING_REASON)
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
    "ADAPTER_PENDING",
    "DATASETS",
    "DATASET_PENDING",
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
