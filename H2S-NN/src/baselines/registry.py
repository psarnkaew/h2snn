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
# Card 15: acm_pp (ACM paper-cite-paper) replaces acm_pa, which is ill-posed under the LP protocol (cards 11 A, 12 L).
DERIVED_LP: tuple[str, ...] = ("acm_pp", "dblp_ap", "imdb_ma", "freebase_bb")

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
    "hinormer": ("not used: H2GB's own HINormerModel (transformer and encoder commented out, hinormer_model.py:221-222, "
                 "274-289); the original HINormer runs instead (card 1c F1, decision G-1)"),
    "sehgnn": "adapter pending (card baselines-h2gb C4)",
    "chebnetii": "adapter pending (card baselines-h2gb C4)",
    "hetero2net": ("adapter in place, pending author (waiting for `mag_metapath2vec_emb.zip` under "
                   "<data root>/NC/mag/hetero2net/, author download: https://data.pyg.org/datasets/mag_metapath2vec_emb.zip; "
                   "card 1c F4)"),
}
#: Cells wired on the loaded H2GB NC datasets (card baselines-h2gb C4): adapter and its tests in place, no open
#: author decision.
H2GB_NC_WIRED_CELLS: frozenset[tuple[str, str]] = frozenset({
    # part 2 E1 (author decisions S1, S5): class A on H2GB's hgt sampler with H2GB's YAML fan-out
    *((m, d) for m in ("gcn", "gat", "han") for d in H2GB_NC),
    *(("h2gformer", d) for d in H2GB_NC),  # mag-year: card 1c F6 (G-7)
    # part 2 E3 (D1 second branch, S1): class C on the hgt sampler with the dataset's GCN fan-out
    *((m, d) for m in ("nsd", "csnn") for d in H2GB_NC),
    # card 1c F1 (G-1): the original HINormer, class B
    *(("hinormer", d) for d in H2GB_NC),
    ("sehgnn", "ieee-cis-g"),
    ("sehgnn", "ogbn-mag"), ("sehgnn", "mag-year"),  # card 1c F3 (G-4)
    # card 1c F4 (G-5 amended): Hetero2Net without a metapath list (its FreeBase branch)
    *(("hetero2net", d) for d in ("oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns")),
    ("hetero2net", "ogbn-mag"), ("hetero2net", "mag-year"),  # card 1d H3: the author's embedding file in place
    ("pshgcn", "ogbn-mag"), ("pshgcn", "mag-year"),  # card 1d H1 (G-2, D1-b): their ogbn-mag code with sparse_tools
    # part 2 E2: decision C4-b (parallel relations merged)
    ("sehgnn", "oag-cs"), ("sehgnn", "oag-eng"), ("sehgnn", "oag-chem"), ("sehgnn", "pdns"),
    ("chebnetii", "ogbn-mag"), ("chebnetii", "mag-year"),
    ("chebnetii", "oag-cs"), ("chebnetii", "oag-eng"), ("chebnetii", "oag-chem"),
    ("chebnetii", "pdns"),  # part 2 E2: decision C4-c (per-type projection after per-block bases)
    ("chebnetii", "ieee-cis-g"), ("chebnetii", "h-pokec"),  # card 1c F5 (G-6): projection first
})
#: Per-cell reasons on the loaded H2GB NC datasets (they win over the adapter's general reason).
H2GB_NC_CELL_REASONS: dict[tuple[str, str], str] = {
}
#: SeHGNN on H-Pokec (author, 2026-11-16; Part M3b B7).
SEHGNN_HPOKEC_NA = ("N/A: out of memory under its own code (stored propagated features 1,214-29,685 GiB at "
                    "num_hops 2-4, Part M3b B7; author decision 2026-11-16)")
#: HAN on Freebase LP (card 1e F3.2, author 2026-11-18; F0 `p6_f0_diagnosis.md` item 3, F1 `p6_f1_stop_items.md`).
HAN_FREEBASE_LP_NA = ("N/A: OOM on a 24 GB GPU at num_heads 4, the grid minimum; the width is a secondary memory term "
                      "(card 1e F0 item 3; author decision 2026-11-18)")
#: Card baselines-h2gb C6: link-level sampling on the task's message graph and the task files are in place
#: (`src/baselines/large_lp_minibatch.py`, held-out-fact test and mutation); no LP adapter is wired yet.
#: Part 2 E4: large-LP cells wired (link-level sampling on the message graph, the local-LP encoder, shared decoder).
LARGE_LP_WIRED_CELLS: frozenset[tuple[str, str]] = frozenset({
    # card 1c F7 (G-8, G-10): large_lp_training.py; notebooks regenerated (G-9)
    *((m, t) for m in ("gcn", "gat", "nsd", "csnn", "han", "h2gformer") for t in ("ogbn-mag_pp", "h-pokec_uu")),
    ("hetero2net", "h-pokec_uu"),
    ("hetero2net", "ogbn-mag_pp"),  # card 1d H3
})
_LP_NOT_WIRED = ("adapter pending: part 2 E4 did not wire this method's large-LP path (its local-LP encoder on link-level "
                 "subgraphs); the sampler, the message graph and the loop are in place (large_lp_training.py)")
LARGE_LP_CELL_REASONS: dict[tuple[str, str], str] = {
}
LARGE_LP_ADAPTER_PENDING: dict[str, str] = {
    **{name: _LP_NOT_WIRED for name in ("gcn", "gat", "nsd", "csnn", "han", "h2gformer")},
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


#: Part P P3: Freebase NC needed the authors' own Freebase preprocessing for SeHGNN, PSHGCN and H2SGNN; wired in card 3
#: Part B (no cell uses this reason any more; kept for the record).
FREEBASE_VENDOR_REASON = (
    "Freebase NC not wired: needs the authors' own Freebase preprocessing (their HGB loader "
    "branch); the vendor clone was not available when Part P ran (UNVERIFIED whether their code "
    "has a Freebase branch)")

#: Card 3 Part A A3 (H-4): HetSheaf as class C on the 8 H2GB NC datasets (H2GB's hgt sampler with the dataset's GCN list,
#: S1; their chain on the whole graph, type ids per subgraph) and both large LP tasks on the shared loop (card 1c F7:
#: ogbn-mag P-P per-hop link sampler, D2; H-Pokec U-U H2GB's hgt sampler with its per-type lists, G-8; author 2026-11-16).
H2GB_NC_WIRED_CELLS = H2GB_NC_WIRED_CELLS | frozenset(("hetsheaf", dataset) for dataset in H2GB_NC)
LARGE_LP_WIRED_CELLS = LARGE_LP_WIRED_CELLS | frozenset({("hetsheaf", "ogbn-mag_pp"), ("hetsheaf", "h-pokec_uu")})

#: Card 3 Part B (G-2, G-3; F-1 generic branch, `adapters/_hgb_generic.py`): PSHGCN on the 6 non-MAG H2GB NC datasets
#: (their decoupled scheme, `minibatch.train_index_minibatch`) and H2SGNN on all 8 (class C on H2GB's `hgt` sampler,
#: `minibatch.train_minibatch_node_classification`); both on the two large LP tasks (shared loop). Freebase NC: the
#: three `FREEBASE_VENDOR_REASON` entries removed (SeHGNN its own Freebase branch; PSHGCN, H2SGNN the generic branch).
H2GB_NC_WIRED_CELLS = H2GB_NC_WIRED_CELLS | frozenset(
    {*(("pshgcn", d) for d in ("oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns")),
     *(("h2sgnn", d) for d in H2GB_NC)})
LARGE_LP_WIRED_CELLS = LARGE_LP_WIRED_CELLS | frozenset(
    (m, t) for m in ("pshgcn", "h2sgnn") for t in ("ogbn-mag_pp", "h-pokec_uu"))


#: Card 8 C2/C3/C4 (author 2026-11-20): R-GCN and MAGNN on the H2GB NC datasets (R-GCN: H2GB's own R-GCN; MAGNN: probe
#: cells, the t-X-t rule) and on the two large LP tasks (`fit_large_lp`).
H2GB_NC_WIRED_CELLS = H2GB_NC_WIRED_CELLS | frozenset((m, d) for m in ("rgcn", "magnn") for d in H2GB_NC)
LARGE_LP_WIRED_CELLS = LARGE_LP_WIRED_CELLS | frozenset((m, t) for m in ("rgcn", "magnn") for t in LARGE_LP)
#: Card 8 C4 (author 2026-11-20, GC option (a)): the GC baseline rows are HetSheaf's published Table 4 numbers; no
#: baseline GC cell runs (the adapters keep their GC code).
GC_OPTION_A_REASON = "GC option (a): HetSheaf published number (author 2026-11-20; no baseline GC run)"
#: Card 8 C3: MAGNN NC on ACM, DBLP, IMDB holds HGB's published number; MAGNN's own preprocessing of these datasets
#: (ACM's added self-loops and sub-graph, IMDB's full-batch MAGNN_nc) is not wired.
MAGNN_HGB_NC_NOT_WIRED = ("N/A here: the cell holds HGB's published number (register); MAGNN's own ACM/DBLP/IMDB "
                          "preprocessing is not wired (card 8 C3)")


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
    # card 8 C2: R-GCN on NC and LP (GC: HetSheaf's published number)
    ModelSpec("rgcn", "R-GCN", HET_HOM, "src.baselines.adapters.rgcn", (NODE_CLASSIFICATION, LINK_PREDICTION)),
    ModelSpec("han", "HAN", HET_HOM, "src.baselines.adapters.hgb_han", _ALL,
              unsupported=((LINK_PREDICTION, "freebase_bb", HAN_FREEBASE_LP_NA),)),  # card 1e F3.2
    # card 8 C3: MAGNN on NC and LP (not GC: the TU graphs define no meta-paths)
    ModelSpec("magnn", "MAGNN", HET_HOM, "src.baselines.adapters.magnn", (NODE_CLASSIFICATION, LINK_PREDICTION),
              unsupported=tuple((NODE_CLASSIFICATION, d, MAGNN_HGB_NC_NOT_WIRED) for d in ("acm", "dblp", "imdb"))),
    ModelSpec(
        "sehgnn", "SeHGNN", HET_HOM, "src.baselines.adapters.sehgnn",
        (NODE_CLASSIFICATION,),
        unsupported=((NODE_CLASSIFICATION, "h-pokec", SEHGNN_HPOKEC_NA),),
    ),
    ModelSpec(
        "hinormer", "HiNormer", HET_HOM, "src.baselines.adapters.hinormer",
        (NODE_CLASSIFICATION,),
    ),
    ModelSpec("pshgcn", "PSHGCN", HET_HOM, "src.baselines.adapters.pshgcn", _ALL),
    ModelSpec("h2gformer", "H2G-former", HET_HET, "src.baselines.adapters.h2gformer", _ALL),
    ModelSpec("hetero2net", "Hetero$^2$Net", HET_HET, "src.baselines.adapters.hetero2net", _ALL),
    ModelSpec("h2sgnn", "H2SGNN", HET_HET, "src.baselines.adapters.h2sgnn", _ALL),
    # Card 3 Part A (H-1..H-4): HetSheaf rerun under the shared protocol (spec v1.22 §6, C7); heterogeneous and
    # heterophily-aimed as its paper states. GC cells: card 3 A2.
    ModelSpec("hetsheaf", "HetSheaf", HET_HET, "src.baselines.adapters.hetsheaf", _ALL),
)

#: Baselines of the protocol whose adapter is not written yet. Not cited: no number is quoted in a table.
#: Card 3 A1: HetSheaf's entry replaced by its ModelSpec row.
ADAPTER_PENDING: dict[str, str] = {}

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
    if task == GRAPH_CLASSIFICATION:  # card 8 C4: GC option (a)
        return GC_OPTION_A_REASON
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
        if (name, dataset) in LARGE_LP_WIRED_CELLS:
            return None
        return LARGE_LP_CELL_REASONS.get((name, dataset),
                                         LARGE_LP_ADAPTER_PENDING.get(name, LARGE_LP_ADAPTER_PENDING_REASON))
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
