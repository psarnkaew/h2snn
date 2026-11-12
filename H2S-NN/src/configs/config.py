from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


def _reject_unknown_keys(section: str, raw: Mapping[str, Any], config_class: type) -> None:
    """Fail loudly on keys the dataclass does not define (e.g. removed v1 settings)."""
    unknown = sorted(set(raw) - set(config_class.__dataclass_fields__))
    if unknown:
        raise ValueError(
            f"Unknown {section} key(s): {', '.join(unknown)}. They are not settings of the "
            "current model; remove them (v1 and AMPSS settings were deleted, see "
            "docs/H2S-NN_v2_plan.md D6)."
        )


# Settings that are ablation controls only: reported as separate experiments and never part
# of a tuning search space, sweep config or tuning notebook (plan §9.1, §4.6).
NEVER_TUNED_DATA_KEYS: tuple[str, ...] = ("fact_layer",)
# D10: neither the coupling mode nor the gate init is ever a search-space parameter.
NEVER_TUNED_MODEL_KEYS: tuple[str, ...] = ("coupling", "coupling_gate_init")
# "pair_attention" (plan §12.3, Part F U4b) is the test of assumption A4, not a model option.
COUPLING_MODES: tuple[str, ...] = ("cooperative", "listen_only", "symmetric", "pair_attention")
# D12 G0b diagnostics (gap ladder): never a search-space parameter; identity_maps is also
# refused in a tuning base config (it is a control, not a model option).
NEVER_TUNED_DIAGNOSTIC_KEYS: tuple[str, ...] = ("identity_maps", "feature_normalization")
# D12 comparison arms (C8, C9): never a search-space parameter; the non-default values are refused
# in a tuning base config (evidence rows, never tuned, never defaults).
NEVER_TUNED_D12_KEYS: tuple[str, ...] = ("geometry_context", "encoder")
ENCODERS: tuple[str, ...] = ("filter_bank", "first_order")
# Part H (WISP, spec v1.3 §3.2): the path source and the WISP method constants are never tuned.
# Spec v1.15 J2 (card Part M, M10): the support `wisp_max_length` (l_max in {2, 3, 4, 5}) and the
# precision `wisp_walks_per_anchor` (r in {8, 16, 32, 64}) ARE tuned; they left this list.
PATH_SOURCES: tuple[str, ...] = ("schemas", "wisp")
NEVER_TUNED_WISP_KEYS: tuple[str, ...] = ("path_source", "path_weight", "path_weight_init",
                                           "path_weight_input", "wisp_proposal",
                                           "path_weight_context")
# Spec v1.15 J3 / experiment_settings.html §4.2 (second table): keys that are not components of
# the method are never in H2S-NN's search space (the tuner refuses them by exact name).
NEVER_TUNED_LEGACY_KEYS: tuple[str, ...] = (
    "K", "layer_dropout", "eta_init", "gamma_init", "coupling_gate_init", "encoder", "coupling",
    "path_sampler", "face_sample_k", "exclude_identity_metapaths", "metapath_templates",
    "feature_normalization", "per_side_scale", "identity_maps", "h0_only_readout",
    "direction_blind_maps", "free_path_maps", "endpoint_edges", "holonomy_edges",
    "no_path_cells", "uniform_path_gate", "num_label_prop_steps",
    "compute_relation_diagnostics", "checkpointing", "node_edge_memory_guard_fraction",
    "overlap_sampling", "resample_paths")
# Part H3: q = p ("uniform") or the adaptive defensive proposal (variance only; fixed by a gate).
WISP_PROPOSALS: tuple[str, ...] = ("uniform", "adaptive")
# Cell weight (Part H): "exp" = w_f = exp(g(xi_f)), gamma removed (it is the bias of g; spec v1.3
# §3.2c); "sigmoid_gamma" = gamma * sigmoid(g) (the D11/D12 behaviour, comparison arm).
PATH_WEIGHTS: tuple[str, ...] = ("exp", "sigmoid_gamma")
# The C2 weight ladder (spec v1.3 §5): inputs of g, nested restrictions on identical cells.
PATH_WEIGHT_INPUTS: tuple[str, ...] = ("full", "endpoints", "schema", "uniform")
FEATURE_NORMALIZATIONS: tuple[str, ...] = ("none", "row")
# v3-B (spec v1.11 R13, §6; card Part L): "chebyshev" = the v1.2-1.10 propagation (gated
# Chebyshev filter bank / first-order update; legacy, bit-identical); "anchored" = one anchored
# quadratic solve `(P^T P + tau L_hat) Y* = P^T h0` on the three-level sheaf energy (PCG, implicit
# backward). Never tuned.
PROPAGATIONS: tuple[str, ...] = ("chebyshev", "anchored")
# Spec v1.15 §6 GC (J6): pooling before the one Linear layer of the graph readout; tuned (M10).
GRAPH_POOLINGS: tuple[str, ...] = ("mean", "max", "add", "attention")
# v3-B arms (spec §6): every arm is a configuration of the one model under `propagation: anchored`
# (same WISP draw, maps, anchor, head). Comparison rows; "full" is the model; never tuned.
V3B_ARMS: tuple[str, ...] = ("full", "covering_only", "hub_A", "closure_only", "no_path_cells",
                             "instance_features", "free_cell_maps")
# Card §3: the solver tolerance, its schedule, T_max and tau's initialisation are fixed, never in a
# search space; the propagation and the arm are comparison switches.
NEVER_TUNED_V3B_KEYS: tuple[str, ...] = ("propagation", "arm", "tau_init", "solver_tolerance",
                                         "solver_warmup_tolerance", "solver_warmup_epochs",
                                         "solver_max_iterations", "tau_stop",
                                         "log_weight_range_stop")


def reject_never_tuned_settings(
    raw_config: Mapping[str, Any] | None,
    search_parameters: Any = (),
) -> None:
    """Refuse a tuning setup that touches a never-tuned ablation control.

    Raises if the base config sets `data.fact_layer: untied`, or if any search-space
    parameter name mentions `fact_layer` (with any values). The model's default is the tied
    layer, so the untied control is reachable only by setting it explicitly.
    """
    data = (raw_config or {}).get("data") or {}
    coupling = str(((raw_config or {}).get("ablation") or {}).get("coupling", "cooperative"))
    propagation = str(((raw_config or {}).get("model") or {}).get("propagation", "chebyshev"))
    # v3-B (spec v1.11 G4, D-B1): the anchored model has no gates, so `symmetric` is the model's
    # own coupling there (ModelConfig refuses any other); the rule below is the legacy one.
    if propagation == "anchored" and coupling == "symmetric":
        pass
    elif coupling != "cooperative":
        raise ValueError(
            f"ablation.coupling: {coupling} is an ablation / test row (plan §10.5, §12.3) and must "
            "not be used in tuning; tune the cooperative model and report the row separately."
        )
    optim = (raw_config or {}).get("optim") or {}
    # Spec v1.15 §6 (J4): no learning-rate scheduler and no gradient clipping for any method.
    if str(optim.get("scheduler", "none")) != "none" or float(optim.get("grad_clip", 0.0)) > 0:
        raise ValueError(
            f"optim.scheduler {optim.get('scheduler')!r} / optim.grad_clip {optim.get('grad_clip')!r}: "
            "the shared protocol has no scheduler and no gradient clipping (spec v1.15 §6); "
            "remove them from a tuning base config.")
    if str(data.get("fact_layer", "")).lower() == "untied":
        raise ValueError(
            "data.fact_layer: untied is an ablation control and must not be used in tuning "
            "(plan §9.1); tune the tied model and report untied separately."
        )
    if ((raw_config or {}).get("ablation") or {}).get("identity_maps", False):
        raise ValueError(
            "ablation.identity_maps is the sheaf-free diagnostic of the D12 gap ladder and must "
            "not be used in tuning."
        )
    model = (raw_config or {}).get("model") or {}
    if model.get("geometry_context", True) is not True:
        raise ValueError(
            "model.geometry_context: false is the C9 comparison arm and must not be used in tuning."
        )
    if str(model.get("encoder", "filter_bank")) != "filter_bank":
        raise ValueError(
            "model.encoder: first_order is the C8 comparison arm and must not be used in tuning."
        )
    if str(model.get("path_weight", "exp")) != "exp":
        raise ValueError(
            "model.path_weight: sigmoid_gamma is a comparison arm (Part H) and must not be used in "
            "tuning."
        )
    if str(model.get("arm", "full")) != "full":
        raise ValueError(
            f"model.arm: {model.get('arm')} is a v3-B comparison arm (spec v1.11 §6) and must not be "
            "used in tuning; tune the full model and report the arm separately."
        )
    if str(((raw_config or {}).get("ablation") or {}).get("path_weight_input", "full")) != "full":
        raise ValueError(
            "ablation.path_weight_input other than full is a C2 ladder arm and must not be used in "
            "tuning."
        )
    dataset = str(data.get("dataset_name", "acm")).lower()
    node_classification = (str(data.get("task_type", "")) or DATASET_TASK_TYPES.get(dataset, "")
                           ) == "node_classification"
    if str(data.get("path_source", "")) == "schemas" and node_classification:
        raise ValueError(
            "data.path_source: schemas is the fixed-literature-schema comparison row for node "
            "classification (spec v1.4 E7) and must not be used in tuning."
        )
    # Spec v1.15 J2: l_max and r are tuned (no pre-registered value, no calibrated-r rule in a
    # tuning base config); the proposal stays fixed (spec §6 WISP settings).
    if str(data.get("wisp_proposal", "uniform")) != "uniform":
        raise ValueError("data.wisp_proposal: adaptive is not the pre-registered proposal "
                         "(spec §6) and must not be used in a tuning base config.")
    if ((raw_config or {}).get("ablation") or {}).get("path_weight_context", False) is not False:
        raise ValueError(
            "ablation.path_weight_context: true is the spec v1.4 appendix pair (g reads [h0, c]) "
            "and must not be used in tuning."
        )
    for name in search_parameters:
        # exact last component: `model.encoder_dropout` is an ordinary hyperparameter
        if str(name).split(".")[-1] in NEVER_TUNED_D12_KEYS:
            raise ValueError(
                f"tuning search space contains {name!r}: the D12 comparison switches "
                "(geometry_context, encoder) are never tuned."
            )
        if any(key in str(name) for key in NEVER_TUNED_DIAGNOSTIC_KEYS):
            raise ValueError(
                f"tuning search space contains {name!r}: identity_maps and feature_normalization "
                "are D12 diagnostics, never tuned."
            )
        if str(name).split(".")[-1] in NEVER_TUNED_V3B_KEYS:
            raise ValueError(
                f"tuning search space contains {name!r}: the v3-B propagation, arm, tau init and "
                f"solver settings ({', '.join(NEVER_TUNED_V3B_KEYS)}) are fixed, never tuned "
                "(card Part L §3)."
            )
        if str(name).split(".")[-1] in NEVER_TUNED_LEGACY_KEYS:
            raise ValueError(
                f"tuning search space contains {name!r}: not a component of the method (spec "
                "v1.15 J3; experiment_settings.html §4.2); never in H2S-NN's search space.")
        if str(name).split(".")[-1] in NEVER_TUNED_WISP_KEYS:
            raise ValueError(
                f"tuning search space contains {name!r}: the WISP settings "
                f"({', '.join(NEVER_TUNED_WISP_KEYS)}) are pre-registered or comparison arms, "
                "never tuned (spec v1.4 §6)."
            )
        if any(key in str(name) for key in NEVER_TUNED_DATA_KEYS):
            raise ValueError(
                f"tuning search space contains {name!r}: fact_layer is never tuned (plan §9.1)."
            )
        if any(key in str(name) for key in NEVER_TUNED_MODEL_KEYS):
            raise ValueError(
                f"tuning search space contains {name!r}: ablation.coupling and "
                "model.coupling_gate_init are never tuned (plan §10.5)."
            )


@dataclass(slots=True)
class MetaPathConfig:
    name: str
    node_types: tuple[str, ...]
    sample_per_anchor: int | None = None
    # Optional explicit hops [(relation_name, "+" | "-"), ...], one per node-type
    # transition (D7). Needed when a transition resolves to more than one (relation,
    # direction) under the active fact layer, e.g. ACM paper -> paper (cite +, cite -).
    hops: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self) -> None:
        if self.hops is None:
            return
        hops = tuple((str(hop[0]), str(hop[1])) for hop in self.hops)
        if any(len(hop) != 2 for hop in self.hops) or any(s not in {"+", "-"} for _, s in hops):
            raise ValueError(
                f"metapath {self.name!r}: hops must be [relation_name, '+' | '-'] pairs, "
                f"got {self.hops!r}."
            )
        if len(hops) != len(self.node_types) - 1:
            raise ValueError(
                f"metapath {self.name!r}: {len(hops)} hops for {len(self.node_types)} node types."
            )
        self.hops = hops

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetaPathConfig":
        _reject_unknown_keys("data.metapath_templates[]", raw, cls)
        hops = raw.get("hops")
        return cls(
            name=str(raw["name"]),
            node_types=tuple(str(value) for value in raw["node_types"]),
            sample_per_anchor=raw.get("sample_per_anchor"),
            hops=None if hops is None else tuple(tuple(hop) for hop in hops),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_types": list(self.node_types),
            "sample_per_anchor": self.sample_per_anchor,
            "hops": None if self.hops is None else [list(hop) for hop in self.hops],
        }


NODE_CLASSIFICATION_PRESETS: dict[str, dict[str, Any]] = {
    "acm": {
        "target_node_type": "paper",
        "metapath_templates": (
            MetaPathConfig(name="PAP", node_types=("paper", "author", "paper")),
            MetaPathConfig(name="PSP", node_types=("paper", "subject", "paper")),
        ),
    },
    "dblp": {
        "target_node_type": "author",
        "metapath_templates": (
            MetaPathConfig(name="APA", node_types=("author", "paper", "author")),
            MetaPathConfig(
                name="APTPA",
                node_types=("author", "paper", "term", "paper", "author"),
                sample_per_anchor=50,
            ),
        ),
    },
    "imdb": {
        "target_node_type": "movie",
        "metapath_templates": (
            MetaPathConfig(name="MAM", node_types=("movie", "actor", "movie")),
            MetaPathConfig(name="MDM", node_types=("movie", "director", "movie")),
        ),
    },
}

# The literature meta-path list of the fixed-literature-schema row and of the schema read-out
# (spec v1.19 §6, O-1 = (a), author 2026-11-13; card Part O O2/O3). One table: its source and its
# per-dataset entries. HGB's published meta-path table, the paths that start at the target type,
# on ACM, DBLP, IMDB and Freebase only (no row on H2GB datasets). Relation-level definitions from
# HGB's code (docs/predefined_metapaths.md §2): MAGNN NC/benchmark/methods/MAGNN/run_ACM.py:20
# (ACM [2,3] [4,5] [0,2,3] [0,4,5] [1,2,3] [1,4,5] over link ids 0 paper-cite-paper, 1
# paper-ref-paper, 2/3 paper-author, 4/5 paper-subject), run_DBLP.py:20, run_IMDB.py:19 (target
# movie: [0,1] [2,3]); HAN NC/benchmark/methods/HAN/utils.py:234 for Freebase (DGL edge names: 'ab'
# = link a->b stored, a reversed link otherwise, utils.py:180-201). Hops name the relations as
# PyG's HGBDataset names them (`<src>__<rel>__<dst>`; IMDB's movie->actor is `movie__>actorh__actor`
# in HGB's info.dat); '+' along the stored relation, '-' against it. The old
# `NODE_CLASSIFICATION_PRESETS` lists stay for the legacy schema rows only (`resolved()`).
LITERATURE_SCHEMA_SOURCE: str = (
    "HGB published meta-path table, paths starting at the target type (spec v1.19 §6); relation "
    "level from HGB's code (MAGNN run_ACM.py:20, run_DBLP.py:20, run_IMDB.py:19; HAN "
    "utils.py:234 for Freebase; docs/predefined_metapaths.md §2)")


def _literature_path(name: str, node_types: tuple[str, ...],
                     hops: tuple[tuple[str, str], ...]) -> MetaPathConfig:
    return MetaPathConfig(name=name, node_types=node_types, hops=hops)


LITERATURE_METAPATHS: dict[str, tuple[MetaPathConfig, ...]] = {
    "acm": (  # target paper
        _literature_path("PAP", ("paper", "author", "paper"),
                         (("paper__to__author", "+"), ("author__to__paper", "+"))),
        _literature_path("PSP", ("paper", "subject", "paper"),
                         (("paper__to__subject", "+"), ("subject__to__paper", "+"))),
        _literature_path("PcPAP", ("paper", "paper", "author", "paper"),
                         (("paper__cite__paper", "+"), ("paper__to__author", "+"),
                          ("author__to__paper", "+"))),
        _literature_path("PcPSP", ("paper", "paper", "subject", "paper"),
                         (("paper__cite__paper", "+"), ("paper__to__subject", "+"),
                          ("subject__to__paper", "+"))),
        _literature_path("PrPAP", ("paper", "paper", "author", "paper"),
                         (("paper__ref__paper", "+"), ("paper__to__author", "+"),
                          ("author__to__paper", "+"))),
        _literature_path("PrPSP", ("paper", "paper", "subject", "paper"),
                         (("paper__ref__paper", "+"), ("paper__to__subject", "+"),
                          ("subject__to__paper", "+"))),
    ),
    "dblp": (  # target author
        _literature_path("APA", ("author", "paper", "author"),
                         (("author__to__paper", "+"), ("paper__to__author", "+"))),
        _literature_path("APTPA", ("author", "paper", "term", "paper", "author"),
                         (("author__to__paper", "+"), ("paper__to__term", "+"),
                          ("term__to__paper", "+"), ("paper__to__author", "+"))),
        _literature_path("APVPA", ("author", "paper", "venue", "paper", "author"),
                         (("author__to__paper", "+"), ("paper__to__venue", "+"),
                          ("venue__to__paper", "+"), ("paper__to__author", "+"))),
    ),
    "imdb": (  # target movie
        _literature_path("MDM", ("movie", "director", "movie"),
                         (("movie__to__director", "+"), ("director__to__movie", "+"))),
        _literature_path("MAM", ("movie", "actor", "movie"),
                         (("movie__>actorh__actor", "+"), ("actor__to__movie", "+"))),
    ),
    "freebase": (  # target book; HAN's codes ['00','00'] ['01','10'] ['05','52','20'] ['04','40']
        # ['04','43','30'] ['06','61','10'] ['07','70'] (utils.py:234)
        _literature_path("BB", ("book", "book", "book"),
                         (("book__and__book", "+"), ("book__and__book", "+"))),
        _literature_path("BFB", ("book", "film", "book"),
                         (("book__to__film", "+"), ("book__to__film", "-"))),
        _literature_path("BLMB", ("book", "location", "music", "book"),
                         (("book__on__location", "+"), ("music__on__location", "-"),
                          ("music__in__book", "+"))),
        _literature_path("BPB", ("book", "people", "book"),
                         (("people__to__book", "-"), ("people__to__book", "+"))),
        _literature_path("BPSB", ("book", "people", "sports", "book"),
                         (("people__to__book", "-"), ("people__to__sports", "+"),
                          ("book__on__sports", "-"))),
        _literature_path("BOFB", ("book", "organization", "film", "book"),
                         (("book__about__organization", "+"), ("organization__in__film", "+"),
                          ("book__to__film", "-"))),
        _literature_path("BUB", ("book", "business", "book"),
                         (("business__about__book", "-"), ("business__about__book", "+"))),
    ),
}


def literature_walk_split(walks_per_anchor: int, num_paths: int) -> tuple[int, ...]:
    """Walks per anchor of each listed meta-path in the fixed-literature-schema row (spec v1.19
    §6, O-2): the tuned r split evenly over the listed paths, at least one each. With r = q k + s
    (0 <= s < k) the first s paths (table order) get q + 1 and the rest q, so the walks sum to r.
    Raises when r < k (one walk each would exceed r; unreachable for r in {8, 16, 32, 64})."""
    if num_paths < 1:
        raise ValueError("the literature list is empty.")
    if walks_per_anchor < num_paths:
        raise ValueError(
            f"walks per anchor r = {walks_per_anchor} cannot give each of the {num_paths} listed "
            "meta-paths at least one walk while summing to r (spec v1.19 §6).")
    base, extra = divmod(walks_per_anchor, num_paths)
    return tuple(base + (1 if i < extra else 0) for i in range(num_paths))


def literature_lengths(dataset_name: str) -> tuple[int, ...]:
    """Distinct lengths (hops) of the dataset's listed meta-paths, ascending."""
    return tuple(sorted({len(t.node_types) - 1
                         for t in LITERATURE_METAPATHS[dataset_name.lower()]}))

# Backward-compatible alias — kept because it's the name older configs/tooling import.
# r (walks per anchor) by dataset and l_max, by the H1/H6 incidence rule: the smallest r whose
# evaluation-draw incidences (seed 42) reach the dataset's schemas-run budget (ACM 403,122, DBLP
# 862,788, IMDB 143,052). docs/WISP_evidence/k3_calibrate_r.json (spec v1.6 E11). DBLP has no
# entry at l_max 2: even every walk (revisits allowed) gives at most 511,887 incidences.
WISP_WALKS_PER_ANCHOR: dict[str, dict[int, int]] = {
    "acm": {2: 64, 3: 47, 4: 42, 5: 38, 6: 36},
    "dblp": {3: 93, 4: 65, 5: 55, 6: 48},
    "imdb": {2: 25, 3: 13, 4: 12, 5: 11, 6: 11},
}

DATASET_PRESETS = NODE_CLASSIFICATION_PRESETS

LINK_PREDICTION_PRESETS: dict[str, dict[str, Any]] = {
    "lastfm": {
        # HGB local-format LastFM (data/LP/LastFM): node.dat/link.dat/link.dat.test,
        # relation 0 is "user-artist" — the sole relation in link.dat.test.
        "target_relation": ("user", "rel_0", "artist"),
        "neg_sampling_mode": "resample_each_epoch",
        "link_val_ratio": 0.10,
    },
    "amazonbook": {
        "target_relation": ("user", "rates", "book"),
        "neg_sampling_mode": "resample_each_epoch",
        "link_val_ratio": 0.09,
        "link_test_ratio": 0.10,
    },
    "pubmed": {
        "target_relation": ("disease", "rel_2", "disease"),
        "neg_sampling_mode": "resample_each_epoch",
        "link_val_ratio": 0.10,
    },
    "youtube": {
        "target_relation": ("user", "rel_0", "user"),
        "neg_sampling_mode": "resample_each_epoch",
        "link_val_ratio": 0.10,
    },
    "amazon": {
        # HGB's official Amazon LP benchmark (data/amazon): single node type
        # "product", two relations (rel_0 = co-view, rel_1 = co-purchase) both
        # pooled as one "product-product" target — see target_relation_ids in
        # _LOCAL_HGB_LINK_DATASETS. target_relation here names the first one;
        # the loader pools both.
        "target_relation": ("product", "rel_0", "product"),
        "neg_sampling_mode": "resample_each_epoch",
        # HGB holds out a validation edge with probability 1 - train_ratio = 0.10
        # (HGB/LP/benchmark/scripts/data_loader.py:44), which is what LastFM, PubMed, and
        # YouTube above already use. Amazon was the one dataset left at 0.09, so it was
        # the one dataset whose split could not be described as HGB's.
        #
        # The rate is not the realised fraction: `_split_edge_index_hgb_by_head` keeps
        # the first edge of every head node in train unconditionally, exactly as HGB
        # does, so the draw only applies to the remainder and the realised validation
        # share lands below 0.10.
        "link_val_ratio": 0.10,
        "link_test_ratio": 0.10,
    },
}

#: Shared by every graph-classification dataset.
#
# HINPool's evaluation protocol (references/background/HINPOOL.pdf, and hinpool/utils/utils.py:18-48):
# a stratified 80/10/10 holdout repeated over five seeded runs, with 100-epoch early
# stopping on validation. They argue explicitly against 10-fold CV as usually practised,
# because reporting the best test score reached during training leaks through model
# selection -- which is what tud_benchmark's inner loop does
# (gnn_evaluation.py: `if val_acc > best_val_acc: best_test = test(test_loader)`).
# Their four datasets are ours, so matching their protocol makes the numbers directly
# comparable to their Table 2.
_GRAPH_HOLDOUT_PRESET: dict[str, Any] = {
    "split_protocol": "stratified_holdout",
    "num_runs": 5,
}

# Spec v1.15 §6 GC (HetSheaf's protocol in every point, author D7): one random 80/10/10 split
# (`torch.randperm` with seed 42; not stratified, not CV), the same for every training seed;
# hetsheaf/datasets/graph_classification.py:240-251 (commit 6964e1e).
_GRAPH_HETSHEAF_PRESET: dict[str, Any] = {
    "split_protocol": "hetsheaf_random",
    "split_seed": 42,
}

GRAPH_CLASSIFICATION_PRESETS: dict[str, dict[str, Any]] = {
    "mutag": _GRAPH_HETSHEAF_PRESET,
    "proteins": _GRAPH_HETSHEAF_PRESET,
    "enzymes": _GRAPH_HETSHEAF_PRESET,
    "nci1": _GRAPH_HETSHEAF_PRESET,
}

# Spec v1.15 §6, J4 (card Part M, M9): epochs, early stopping and the selection metric per task
# and source protocol, shared by H2S-NN and every baseline (src/baselines/budget.py reads this
# table). `patience: None` = no early stopping (the best validation epoch is selected).
# HGB NC: run_new.py, max 300 epochs, patience 30; HGB LP: 40 epochs, patience 40; H2GB NC: 500
# epochs, no early stopping; GC: HetSheaf (early stopping on validation LOSS, patience 50, test
# with the final weights; epochs 200 = HetSheaf's gc_config.yaml, 500 = its sweep: open O4).
# NC/LP monitor the validation metric (NC Macro-F1, LP ROC-AUC), not the validation loss.
@dataclass(frozen=True, slots=True)
class ProtocolBudget:
    source: str
    epochs: int
    patience: int | None
    monitor: str
    mode: str


PROTOCOL_BUDGETS: dict[tuple[str, str], ProtocolBudget] = {
    ("node_classification", "hgb"): ProtocolBudget("hgb", 300, 30, "macro_f1", "max"),
    ("node_classification", "h2gb"): ProtocolBudget("h2gb", 500, None, "macro_f1", "max"),
    ("link_prediction", "hgb"): ProtocolBudget("hgb", 40, 40, "auroc", "max"),
    ("graph_classification", "hetsheaf"): ProtocolBudget("hetsheaf", 200, 50, "loss", "min"),
}
# H2GB node-classification datasets (spec v1.15 §6; data pending, M11).
H2GB_NC_DATASETS: tuple[str, ...] = ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem",
                                     "ieee-cis-g", "h-pokec", "pdns")


def protocol_budget(task_type: str, dataset_name: str = "") -> ProtocolBudget:
    """The budget of `task_type` under its source protocol (H2GB for the H2GB NC datasets)."""
    if task_type == "node_classification":
        source = "h2gb" if dataset_name.lower() in H2GB_NC_DATASETS else "hgb"
    elif task_type == "link_prediction":
        source = "hgb"
    elif task_type == "graph_classification":
        source = "hetsheaf"
    else:
        raise ValueError(f"Unsupported task {task_type!r}.")
    return PROTOCOL_BUDGETS[(task_type, source)]


DATASET_TASK_TYPES: dict[str, str] = {
    "acm": "node_classification",
    "dblp": "node_classification",
    "imdb": "node_classification",
    "lastfm": "link_prediction",
    "amazonbook": "link_prediction",
    "pubmed": "link_prediction",
    "youtube": "link_prediction",
    "amazon": "link_prediction",
    "mutag": "graph_classification",
    "proteins": "graph_classification",
    "enzymes": "graph_classification",
    "nci1": "graph_classification",
}


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "acm"
    root: Path = Path("data")
    task_type: str = ""   # "" -> inferred from DATASET_TASK_TYPES in resolved()

    # --- node classification ---
    target_node_type: str = ""
    metapath_templates: tuple[MetaPathConfig, ...] = field(default_factory=tuple)
    num_label_prop_steps: int = 3
    acm_val_ratio: float = 0.2
    acm_split_seed: int = 42
    face_sample_k: int = 50
    exclude_identity_metapaths: bool = True
    # Per-relation agreement statistics used only for reporting (see
    # src/data/preprocessing.py: compute_relation_diagnostics). Off by default: the model
    # does not consume them, and computing the structural atoms costs a Python-level
    # per-node two-hop pass. scripts/h2snn_paper_studies.py turns it on.
    compute_relation_diagnostics: bool = False
    # Input-feature preprocessing (D12 G0b diagnostic): "none" = raw HGB features (default,
    # unchanged behaviour); "row" = PyG NormalizeFeatures per node type before stacking (what
    # HetSheaf applies). Node classification only; never tuned (`reject_never_tuned_settings`).
    feature_normalization: str = "none"

    # --- all tasks: fact layer (plan §9.1) ---
    # "tied": one oriented 1-cell per fact, rho from exact edge-set equality (the model).
    # "untied": every stored edge is its own 1-cell (ablation control; never tuned, see
    # `reject_never_tuned_settings`).
    fact_layer: str = "tied"
    # Path-cell builder (plan §9.2): "walk" (r walks per anchor and schema, merged weights,
    # the model), "dfs" (pre-D7 seeded DFS with a quota; untied only), "uniform_instance"
    # (ablation: draws uniform over instances, L <= 2). Baseline bundles pin untied + dfs +
    # no resampling in src/baselines/bundles.py::_base_config (plan §9.5, option A).
    path_sampler: str = "walk"
    # Protocol (plan §9.2): training redraws the path cells every epoch (seed = seed +
    # epoch); validation and test always use the draw with seed `seed`.
    resample_paths: bool = True
    # Path source (Part H; spec v1.15 J1): "wisp" = schema-free walks from the stated target
    # measure (spec §3.2), the path source of every task (anchors: NC target type, LP both
    # endpoint types on the message graph, GC every node); "schemas" = the D8 walk sampler over the
    # literature presets (NC comparison row) or, for LP/GC, the baseline bundles' pinned DFS row.
    # "" (unset) resolves to wisp. `wisp_max_length` is the support l_max; `wisp_walks_per_anchor`
    # is r (precision): 0 (unset) resolves for NC to the calibrated table; LP and GC must set it.
    path_source: str = ""
    wisp_max_length: int = 4
    wisp_walks_per_anchor: int = 0
    # Training draws from 1/2 p + 1/2 q with q from the previous epoch's path mass ("adaptive",
    # Part H3) or q = p ("uniform"); the evaluation draw always uses p. Never tuned.
    wisp_proposal: str = "uniform"
    # Part I (I6): build epoch e+1's training draw in a worker thread while epoch e trains
    # (node-classification trainer, resample_paths only). Changes no result: the seed schedule
    # and the proposal inputs are those of the sequential path (bit-identity test).
    overlap_sampling: bool = True

    # --- link prediction ---
    target_relation: tuple[str, str, str] = ("", "", "")
    decoder_type: str = "dot"                    # "dot" | "distmult" | "concat_mlp"
    # HGB's runner re-draws its training negatives every epoch (spec v1.16 K3): the default.
    neg_sampling_mode: str = "resample_each_epoch"   # "resample_each_epoch" | "fixed"
    neg_sampling_ratio: int = 1
    # Positive links per optimizer step (spec v1.15 §6, J5: HGB's LP default 8,192; not tuned).
    # For H2S-NN one step is one solve on the whole message graph.
    link_batch_size: int = 8192
    link_val_ratio: float = 0.09
    link_test_ratio: float = 0.10

    # --- graph classification ---
    # "hetsheaf_random" (the protocol, spec v1.15 §6) | legacy: "stratified_holdout" |
    # "precomputed_folds" | "stratified_kfold" | "repeated_kfold"
    split_protocol: str = "stratified_holdout"
    num_folds: int = 10
    fold_index: int = 0
    num_repetitions: int = 1
    repetition_index: int = 0
    split_seed: int = 42
    num_runs: int = 5
    run_index: int = 0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # Graphs per batch (spec v1.15 §6: batch size 1 as HetSheaf; not tuned).
    graph_batch_size: int = 1

    def __post_init__(self) -> None:
        """Value checks that need no task (review MINOR-7): direct construction cannot bypass
        them. The task-dependent resolution of path_source and r stays in `resolved()`."""
        if self.path_source not in ("",) + PATH_SOURCES:
            raise ValueError(
                f"data.path_source must be one of {PATH_SOURCES} (or unset), got {self.path_source!r}."
            )
        if self.wisp_max_length < 2:
            raise ValueError(f"data.wisp_max_length must be >= 2, got {self.wisp_max_length}.")
        if self.wisp_walks_per_anchor < 0:
            raise ValueError("data.wisp_walks_per_anchor must be >= 1, or 0 for the dataset's "
                             f"calibrated r; got {self.wisp_walks_per_anchor}.")
        if self.wisp_proposal not in WISP_PROPOSALS:
            raise ValueError(
                f"data.wisp_proposal must be one of {WISP_PROPOSALS}, got {self.wisp_proposal!r}."
            )
        if not isinstance(self.overlap_sampling, bool):
            raise ValueError(
                f"data.overlap_sampling must be a boolean, got {self.overlap_sampling!r}.")
        # path_source: schemas with r >= 1 is the fixed-literature-schema row (spec v1.19 §6; card
        # Part O O3): the tuned `full` configuration with only the path source changed, so its
        # tuned l_max is carried along (not read by the schema draw) and r is the row's walk
        # budget. With r = 0 it is the legacy schema row (old presets, `face_sample_k` quotas).
        if self.path_source == "schemas" and (
            (self.wisp_walks_per_anchor == 0 and self.wisp_max_length != 4)
            or self.wisp_proposal != "uniform"
        ):
            raise ValueError(
                "data.wisp_max_length and wisp_proposal configure path_source: wisp; under "
                "path_source: schemas the proposal must be uniform and, without r (the legacy "
                "schema row), l_max must stay unset (got "
                f"{self.wisp_max_length}, {self.wisp_walks_per_anchor}, {self.wisp_proposal!r})."
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataConfig":
        _reject_unknown_keys("data", raw, cls)
        config = cls(
            dataset_name=str(raw.get("dataset_name", "acm")).lower(),
            root=Path(raw.get("root", "data")),
            task_type=str(raw.get("task_type", "")),
            target_node_type=str(raw.get("target_node_type", "")),
            metapath_templates=tuple(
                MetaPathConfig.from_dict(item)
                for item in raw.get("metapath_templates", [])
            ),
            num_label_prop_steps=int(raw.get("num_label_prop_steps", 3)),
            acm_val_ratio=float(raw.get("acm_val_ratio", 0.2)),
            acm_split_seed=int(raw.get("acm_split_seed", 42)),
            face_sample_k=int(raw.get("face_sample_k", 50)),
            exclude_identity_metapaths=bool(
                raw.get("exclude_identity_metapaths", True)
            ),
            compute_relation_diagnostics=bool(
                raw.get("compute_relation_diagnostics", False)
            ),
            feature_normalization=str(raw.get("feature_normalization", "none")),
            fact_layer=str(raw.get("fact_layer", "tied")),
            path_sampler=str(raw.get("path_sampler", "walk")),
            resample_paths=raw.get("resample_paths", True),
            path_source=str(raw.get("path_source", "")),
            wisp_max_length=int(raw.get("wisp_max_length", 4)),
            wisp_walks_per_anchor=int(raw.get("wisp_walks_per_anchor", 0)),
            wisp_proposal=str(raw.get("wisp_proposal", "uniform")),
            # Not cast with bool(): a string such as "false" must fail validation.
            overlap_sampling=raw.get("overlap_sampling", True),
            target_relation=tuple(raw.get("target_relation", ("", "", ""))),
            decoder_type=str(raw.get("decoder_type", "dot")),
            neg_sampling_mode=str(raw.get("neg_sampling_mode", "resample_each_epoch")),
            neg_sampling_ratio=int(raw.get("neg_sampling_ratio", 1)),
            link_val_ratio=float(raw.get("link_val_ratio", 0.09)),
            link_test_ratio=float(raw.get("link_test_ratio", 0.10)),
            link_batch_size=int(raw.get("link_batch_size", 8192)),
            split_protocol=str(raw.get("split_protocol", "stratified_holdout")),
            num_folds=int(raw.get("num_folds", 10)),
            fold_index=int(raw.get("fold_index", 0)),
            num_repetitions=int(raw.get("num_repetitions", 1)),
            repetition_index=int(raw.get("repetition_index", 0)),
            split_seed=int(raw.get("split_seed", 42)),
            num_runs=int(raw.get("num_runs", 5)),
            run_index=int(raw.get("run_index", 0)),
            train_ratio=float(raw.get("train_ratio", 0.8)),
            val_ratio=float(raw.get("val_ratio", 0.1)),
            test_ratio=float(raw.get("test_ratio", 0.1)),
            graph_batch_size=int(raw.get("graph_batch_size", 1)),
        )
        # The raw keys travel as an argument rather than a field: a dataset preset
        # may only fill in what the caller left unset, and persisting that on the
        # dataclass would put it in every saved config and in the W&B resume
        # fingerprint.
        return config.resolved(frozenset(raw))

    def resolved(self, explicit: frozenset[str] = frozenset()) -> "DataConfig":
        """Infer the task, validate, and fill unset fields from the dataset preset.

        ``explicit`` names the keys the caller supplied. Anything listed there keeps
        its value; only the rest may be filled from a preset. An empty set means
        'nothing was named', so the preset applies throughout -- the behaviour every
        caller had before graph classification acquired a preset that disagrees with
        the dataclass defaults.
        """
        dataset_name = self.dataset_name.lower()
        if self.face_sample_k <= 0:
            raise ValueError("data.face_sample_k must be positive.")
        if self.fact_layer not in {"tied", "untied"}:
            raise ValueError(
                f"data.fact_layer must be 'tied' or 'untied', got {self.fact_layer!r}."
            )
        if self.path_sampler not in {"walk", "dfs", "uniform_instance"}:
            raise ValueError(
                "data.path_sampler must be 'walk', 'dfs' or 'uniform_instance', got "
                f"{self.path_sampler!r}."
            )
        if not isinstance(self.resample_paths, bool):
            raise ValueError(
                f"data.resample_paths must be a boolean, got {self.resample_paths!r}."
            )
        if self.resample_paths and self.path_sampler == "dfs":
            raise ValueError(
                "data.resample_paths needs a sampler with a draw seed (walk or "
                "uniform_instance); path_sampler: dfs keeps its fixed seed."
            )
        if self.fact_layer == "tied" and self.path_sampler == "dfs":
            raise ValueError(
                "data.path_sampler: dfs walks stored out-edges only and cannot follow tied "
                "facts; use path_sampler: walk (or fact_layer: untied for the old behaviour)."
            )
        task_type = self.task_type or DATASET_TASK_TYPES.get(dataset_name, "")
        if not task_type:
            raise ValueError(
                f"Unknown dataset '{self.dataset_name}'; set data.task_type explicitly."
            )
        # Spec v1.15 J1: WISP is the path source of every task; unset path_source resolves to
        # wisp. Unset r resolves to the calibrated table for node classification only; link
        # prediction and graph classification have no table and must name r (it is tuned, J2).
        path_source = self.path_source or "wisp"
        if path_source not in PATH_SOURCES:
            raise ValueError(
                f"data.path_source must be one of {PATH_SOURCES}, got {self.path_source!r}."
            )
        walks = self.wisp_walks_per_anchor
        # LP/GC: r stays 0 (unset) here, so data-only uses (negatives, validation files) load;
        # building a WISP draw with r unset raises (`src/data/wisp_tasks.py`).
        if path_source == "wisp" and walks == 0 and task_type == "node_classification":
            table = WISP_WALKS_PER_ANCHOR.get(dataset_name, {})
            if self.wisp_max_length not in table:
                raise ValueError(
                    f"no calibrated data.wisp_walks_per_anchor for dataset {dataset_name!r} at "
                    f"wisp_max_length {self.wisp_max_length} (calibrated: {sorted(table)}); set r "
                    "explicitly (spec v1.6 E11).")
            walks = table[self.wisp_max_length]
        if self.wisp_max_length < 2:
            raise ValueError(f"data.wisp_max_length must be >= 2, got {self.wisp_max_length}.")
        if walks < (1 if path_source == "wisp" and task_type == "node_classification" else 0):
            raise ValueError(
                f"data.wisp_walks_per_anchor must be >= 1 under path_source: wisp (0 = the "
                f"dataset's calibrated r), got {walks} for dataset {self.dataset_name!r}."
            )
        if self.wisp_proposal not in WISP_PROPOSALS:
            raise ValueError(
                f"data.wisp_proposal must be one of {WISP_PROPOSALS}, got {self.wisp_proposal!r}."
            )
        if self.wisp_proposal == "adaptive" and (path_source != "wisp"
                                                 or not self.resample_paths):
            raise ValueError("data.wisp_proposal: adaptive changes the training draws of "
                             "path_source: wisp; it needs resample_paths: true.")
        if path_source == "schemas" and task_type != "node_classification" and (
                self.path_sampler != "dfs"):
            raise ValueError(
                "data.path_source: schemas for link prediction and graph classification exists "
                "only as the baseline bundles' pinned pre-D7 DFS row (path_sampler: dfs; their "
                "adapters never read path cells). The length-2 schema-free walks were replaced by "
                "WISP (spec v1.15 J1): use path_source: wisp.")
        if path_source == "schemas" and walks >= 1:  # the fixed-literature-schema row
            if task_type != "node_classification":
                raise ValueError("the fixed-literature-schema row (path_source: schemas with "
                                 "wisp_walks_per_anchor r >= 1) exists for node classification "
                                 "only (spec v1.19 §6).")
            listed = LITERATURE_METAPATHS.get(dataset_name, ())
            if not listed:
                raise ValueError(
                    f"no literature meta-path list for dataset {dataset_name!r}: the "
                    "fixed-literature-schema row exists on "
                    f"{sorted(LITERATURE_METAPATHS)} only (spec v1.19 §6).")
            if self.metapath_templates:
                raise ValueError(
                    "the fixed-literature-schema row draws the list LITERATURE_METAPATHS "
                    f"({LITERATURE_SCHEMA_SOURCE}); data.metapath_templates must stay unset.")
            if self.fact_layer != "tied" or self.path_sampler != "walk":
                raise ValueError("the fixed-literature-schema row is the main model with another "
                                 "path source: fact_layer tied, path_sampler walk.")
            literature_walk_split(walks, len(listed))  # raises when r < number of paths
        if path_source == "wisp":
            if self.fact_layer != "tied":
                raise ValueError("data.path_source: wisp needs fact_layer: tied (it walks facts in "
                                 "both directions); the untied and dfs rows name "
                                 "path_source: schemas.")
            if self.path_sampler != "walk":
                raise ValueError("data.path_sampler selects the schema sampler's variant; it must stay "
                                 "'walk' under path_source: wisp.")
        if self.feature_normalization not in FEATURE_NORMALIZATIONS:
            raise ValueError(
                f"data.feature_normalization must be one of {FEATURE_NORMALIZATIONS}, "
                f"got {self.feature_normalization!r}."
            )
        if not self.overlap_sampling and task_type != "node_classification":
            raise ValueError("data.overlap_sampling belongs to the node-classification trainer; "
                             "link prediction and graph classification do not read it.")
        if self.feature_normalization != "none" and task_type != "node_classification":
            raise ValueError("data.feature_normalization is implemented for node classification only.")

        if (path_source, walks) != (self.path_source, self.wisp_walks_per_anchor):
            return replace(self, path_source=path_source,
                           wisp_walks_per_anchor=walks).resolved(explicit)
        if task_type == "node_classification":
            if not 0.0 < self.acm_val_ratio < 1.0:
                raise ValueError("data.acm_val_ratio must be in (0, 1).")
            return self._resolved_node_classification(dataset_name, task_type)
        if task_type == "link_prediction":
            if self.decoder_type not in {"dot", "distmult", "concat_mlp"}:
                raise ValueError("Unsupported data.decoder_type.")
            if self.link_batch_size < 1:
                raise ValueError("data.link_batch_size must be >= 1.")
            if not (
                0.0 < self.link_val_ratio < 1.0
                and 0.0 < self.link_test_ratio < 1.0
                and self.link_val_ratio + self.link_test_ratio < 1.0
            ):
                raise ValueError(
                    "Link validation/test ratios must be positive and sum to less than 1."
                )
            return self._resolved_link_prediction(dataset_name, task_type, explicit)
        if task_type == "graph_classification":
            if self.graph_batch_size < 1:
                raise ValueError("data.graph_batch_size must be >= 1.")
            supported_split_protocols = {
                "hetsheaf_random",
                "stratified_holdout",
                "precomputed_folds",
                "stratified_kfold",
                "repeated_kfold",
            }
            if self.split_protocol not in supported_split_protocols:
                raise ValueError(
                    f"Unsupported data.split_protocol '{self.split_protocol}'."
                )
            if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0.0:
                raise ValueError("Graph train/validation/test ratios must be positive.")
            if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-8:
                raise ValueError("Graph train/validation/test ratios must sum to 1.")
            if self.split_protocol not in ("stratified_holdout", "hetsheaf_random"):
                if self.num_folds < 2:
                    raise ValueError("data.num_folds must be at least 2 for K-fold protocols.")
                if not 0 <= self.fold_index < self.num_folds:
                    raise ValueError(
                        "data.fold_index must be in [0, data.num_folds) for K-fold protocols."
                    )
            if self.split_protocol == "repeated_kfold":
                if self.num_repetitions <= 0:
                    raise ValueError("data.num_repetitions must be positive.")
                if self.split_seed < 0:
                    raise ValueError("data.split_seed must be non-negative.")
                if not 0 <= self.repetition_index < self.num_repetitions:
                    raise ValueError(
                        "data.repetition_index must be in [0, data.num_repetitions)."
                    )
            return self._resolved_graph_classification(
                dataset_name, task_type, explicit
            )
        raise ValueError(f"Unsupported task_type '{task_type}'.")

    def _resolved_node_classification(self, dataset_name: str, task_type: str) -> "DataConfig":
        if dataset_name not in NODE_CLASSIFICATION_PRESETS:
            raise ValueError(f"Unsupported node-classification dataset '{self.dataset_name}'.")

        preset = NODE_CLASSIFICATION_PRESETS[dataset_name]
        target_node_type = self.target_node_type or preset["target_node_type"]
        # Part O (O3): the old preset lists fill only the legacy schema rows (path_source:
        # schemas without r: the baseline bundles' pinned DFS row and the schema tests). WISP
        # reads no list (spec R8) and the fixed-literature-schema row reads LITERATURE_METAPATHS,
        # so neither carries the presets in its resolved (and saved) configuration.
        legacy_schemas = self.path_source == "schemas" and self.wisp_walks_per_anchor == 0
        if self.metapath_templates or not legacy_schemas:
            templates = self.metapath_templates
        else:
            templates = preset["metapath_templates"]

        templates = tuple(
            MetaPathConfig(
                name=template.name,
                node_types=template.node_types,
                sample_per_anchor=(
                    self.face_sample_k
                    if template.sample_per_anchor is None
                    else template.sample_per_anchor
                ),
                hops=template.hops,
            )
            for template in templates
        )
        if self.fact_layer == "untied" and any(
            sign == "-" for template in templates for _, sign in (template.hops or ())
        ):
            raise ValueError(
                "explicit '-' hops need fact_layer: tied; untied walks stored directions only."
            )

        return replace(
            self,
            dataset_name=dataset_name,
            task_type=task_type,
            target_node_type=target_node_type,
            metapath_templates=templates,
        )

    def _resolved_link_prediction(
        self, dataset_name: str, task_type: str, explicit: frozenset[str] = frozenset()
    ) -> "DataConfig":
        if self.path_sampler == "uniform_instance":
            raise ValueError(
                "link prediction has no schemas; path_sampler: uniform_instance needs one."
            )
        preset = LINK_PREDICTION_PRESETS.get(dataset_name, {})
        target_relation = (
            self.target_relation
            if any(self.target_relation)
            else preset.get("target_relation", ("", "", ""))
        )
        if not any(target_relation):
            raise ValueError(
                f"No target_relation for link-prediction dataset '{self.dataset_name}'; "
                "set data.target_relation explicitly."
            )
        # An explicitly supplied key wins; the preset only fills gaps. Comparing against
        # the dataclass default instead -- as this did -- silently discards a config that
        # asks for the same number the default happens to hold, which is how a deliberate
        # `link_val_ratio: 0.09` would have been overwritten the moment a preset moved to
        # 0.10. Same rule the graph-classification resolver already uses.
        def _pick(key: str, current: Any, fallback: Any) -> Any:
            if key in explicit:
                return current
            return preset.get(key, fallback)

        neg_sampling_mode = _pick(
            "neg_sampling_mode", self.neg_sampling_mode, "resample_each_epoch"
        )
        link_val_ratio = _pick("link_val_ratio", self.link_val_ratio, 0.09)
        link_test_ratio = _pick("link_test_ratio", self.link_test_ratio, 0.10)
        return replace(
            self,
            dataset_name=dataset_name,
            task_type=task_type,
            target_relation=tuple(target_relation),
            neg_sampling_mode=neg_sampling_mode,
            link_val_ratio=link_val_ratio,
            link_test_ratio=link_test_ratio,
            metapath_templates=(),
        )

    def _resolved_graph_classification(
        self, dataset_name: str, task_type: str, explicit: frozenset[str]
    ) -> "DataConfig":
        preset = GRAPH_CLASSIFICATION_PRESETS.get(dataset_name, {})

        def resolve(key: str, current: object) -> object:
            # An explicitly named value always wins; the preset only fills gaps.
            # Without this a config asking for stratified_holdout -- the dataclass
            # default -- is indistinguishable from one that said nothing, and the
            # preset silently overrides a deliberate choice.
            if key in explicit:
                return current
            return preset.get(key, current)

        return replace(
            self,
            dataset_name=dataset_name,
            task_type=task_type,
            split_protocol=str(resolve("split_protocol", self.split_protocol)),
            num_runs=int(resolve("num_runs", self.num_runs)),
            split_seed=int(resolve("split_seed", self.split_seed)),
            metapath_templates=(),
        )
        num_folds = self.num_folds if self.num_folds != 10 else preset.get("num_folds", 10)
        num_repetitions = (
            self.num_repetitions
            if self.num_repetitions != 1
            else preset.get("num_repetitions", 1)
        )
        return replace(
            self,
            dataset_name=dataset_name,
            task_type=task_type,
            split_protocol=split_protocol,
            num_folds=num_folds,
            num_repetitions=num_repetitions,
            metapath_templates=(),
        )


@dataclass(slots=True)
class ModelConfig:
    """H2S-NN (plan §4): node diffusion with the vertex-2-cell sheaf Laplacian.

    The task heads (node, link, graph) are chosen by `data.task_type`; ablations live in
    `AblationConfig`.
    """

    d: int = 64
    d_r: int = 16
    encoder_dropout: float = 0.1
    # Hidden width of the node->edge map MLPs (and of the free-map ablation's MLP).
    edge_mlp_hidden_dim: int = 128
    # Hidden width of the positional path encoder that feeds the path gate w_f.
    face_mlp_hidden_dim: int = 128
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.1
    # Graph classification readout (spec v1.15 §6, J6): pooling before one Linear layer
    # ("mean" | "max" | "add" | "attention" = HetSheaf's AttentionPooling); tuned (M10).
    graph_pooling: str = "mean"
    # Diffusion steps x <- x - eta D^{-1/2} (Delta_0 + gamma L_P) D^{-1/2} x.
    K: int = 5
    layer_dropout: float = 0.0
    orthogonal_map: str = "fasthpp"
    # CUDA allocation guard of the dense map family ([M, d, d] maps).
    node_edge_memory_guard_fraction: float = 0.20
    # Node->edge restriction maps (decision D1). "factorized": F = c R_{relation,side} H_k(u),
    # matrix-free. "dense": explicit [M, d, d] FastHPP maps with the same scale law.
    map_family: str = "factorized"
    # Householder reflections k per incidence in the factorized family, 1 <= k <= d.
    householder_k: int = 2
    # Lower bound eps of the conformal scale c = softplus(.) + eps.
    map_scale_eps: float = 0.1
    # Initial gamma = softplus(theta_gamma) > 0 and eta = sigmoid(theta_eta) in (0, 1).
    gamma_init: float = 1.0
    eta_init: float = 0.5
    # D10: initial value of the listen / broadcast gates g, b in (0, 1) (plan §10.2). 0.6 since
    # D11 (plan §12.4): at 0.95 the sigmoid is flat and the gates did not move. Never tuned.
    coupling_gate_init: float = 0.6
    # Activation checkpointing of the matrix-free map and Laplacian applications.
    checkpointing: bool = True
    # D12 (spec v1.2 R10, Eq. 9): the geometry (map MLPs, gate heads, path encoder) is computed
    # once from [h0, c], c = per-(relation, role) mean of the neighbours' h0 over facts. false =
    # geometry from h0 alone (the D11 model; comparison arm C9). Never tuned.
    geometry_context: bool = True
    # D12 (spec v1.2 R9): "filter_bank" = x = sum_k T_k(A - I)(h0 W_k), the method;
    # "first_order" = K steps x <- x - eta A x (the D11 update; nested arm C8, Prop. 7). eta exists
    # only under first_order; filter_bank starts at the first-order update with eta = eta_init.
    # Never tuned.
    encoder: str = "filter_bank"
    # Part H (spec v1.3 §3.2c): "exp" = w_f = exp(g(xi_f)) with the bias b of g initialised to
    # log(path_weight_init) and no gamma; "sigmoid_gamma" = gamma * sigmoid(g) (comparison arm,
    # the D12 model bit for bit). Never tuned.
    path_weight: str = "exp"
    path_weight_init: float = 0.5
    # v3-B (spec v1.11 R13; card Part L). "chebyshev": the legacy propagation above (K steps or the
    # filter bank, gates, eta), unchanged. "anchored": Y* = argmin 1/2||x - h0||^2 + tau/2 Y^T L_hat Y
    # on the three-level energy (nodes, facts, cells), solved by PCG with an implicit backward; no
    # K, eta, gamma or gates. Never tuned.
    propagation: str = "chebyshev"
    # v3-B arm under `propagation: anchored` (spec §6): full | covering_only (no node-cell term) |
    # hub_A (fact states eliminated, no fact-cell term) | closure_only (holonomy edges, no cells) |
    # no_path_cells (beta = 0) | instance_features (beta = 0 in the solve, cell features to the head)
    # | free_cell_maps (free R_{e,f}; Thm 3 control). Comparison rows; never tuned.
    arm: str = "full"
    # tau = exp(theta), theta initialised to log(tau_init); own optimizer group (D-B4). Fixed.
    tau_init: float = 1.0
    # PCG relative tolerance (spec §6: 1e-3 for the first `solver_warmup_epochs` epochs, then 1e-4;
    # a fixed schedule) and T_max (a stop condition: the solve raises when it is reached). Fixed,
    # never tuned (card §3).
    solver_tolerance: float = 1e-4
    solver_warmup_tolerance: float = 1e-3
    solver_warmup_epochs: int = 10
    solver_max_iterations: int = 100
    # Stop conditions (spec v1.15 §3, §6; J10): tau > tau_stop and a log-weight range of g above
    # log_weight_range_stop raise (StopConditionError), as T_max does. Fixed, never tuned.
    tau_stop: float = 50.0
    log_weight_range_stop: float = 30.0

    def __post_init__(self) -> None:
        if self.graph_pooling not in GRAPH_POOLINGS:
            raise ValueError(
                f"model.graph_pooling must be one of {GRAPH_POOLINGS}, got {self.graph_pooling!r}.")
        if self.propagation not in PROPAGATIONS:
            raise ValueError(
                f"model.propagation must be one of {PROPAGATIONS}, got {self.propagation!r}."
            )
        if self.arm not in V3B_ARMS:
            raise ValueError(f"model.arm must be one of {V3B_ARMS}, got {self.arm!r}.")
        if self.propagation != "anchored" and self.arm != "full":
            raise ValueError(
                f"model.arm: {self.arm} is a v3-B arm and exists only under model.propagation: "
                "anchored (the legacy propagation keeps its ablation switches)."
            )
        if not self.tau_init > 0.0:
            raise ValueError(f"model.tau_init must be positive, got {self.tau_init}.")
        for name in ("tau_stop", "log_weight_range_stop"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"model.{name} must be positive, got {getattr(self, name)}.")
        for name in ("solver_tolerance", "solver_warmup_tolerance"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"model.{name} must be in (0, 1), got {value}.")
        for name, lower in (("solver_warmup_epochs", 0), ("solver_max_iterations", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"model.{name} must be an integer >= {lower}, got {value!r}.")
        if self.propagation == "anchored" and self.path_weight != "exp":
            raise ValueError("model.propagation: anchored needs model.path_weight: exp (beta_f = "
                             "w_f m_f with w_f = exp(g); there is no gamma).")
        if self.path_weight not in PATH_WEIGHTS:
            raise ValueError(
                f"model.path_weight must be one of {PATH_WEIGHTS}, got {self.path_weight!r}."
            )
        # A value the active weight mode would ignore is rejected (the saved defaults pass, so
        # configs round-trip through save_config).
        if self.path_weight == "exp" and self.gamma_init != 1.0:
            raise ValueError("model.gamma_init is rejected under model.path_weight: exp (there is "
                             "no gamma; the scale is the bias of g, set by path_weight_init).")
        if self.path_weight == "sigmoid_gamma" and self.path_weight_init != 0.5:
            raise ValueError("model.path_weight_init exists only under model.path_weight: exp.")
        if not self.path_weight_init > 0.0:
            raise ValueError(
                f"model.path_weight_init must be positive, got {self.path_weight_init}."
            )
        if self.encoder not in ENCODERS:
            raise ValueError(f"model.encoder must be one of {ENCODERS}, got {self.encoder!r}.")
        if not isinstance(self.geometry_context, bool):
            raise ValueError(
                f"model.geometry_context must be a boolean, got {self.geometry_context!r}."
            )
        if self.orthogonal_map != "fasthpp":
            raise ValueError(
                f"Unsupported model.orthogonal_map '{self.orthogonal_map}'. "
                "Only 'fasthpp' is implemented."
            )
        if not 0.0 < self.node_edge_memory_guard_fraction <= 1.0:
            raise ValueError(
                "model.node_edge_memory_guard_fraction must be in (0, 1], "
                f"got {self.node_edge_memory_guard_fraction}."
            )
        if self.map_family not in {"dense", "factorized"}:
            raise ValueError(
                f"Unsupported model.map_family '{self.map_family}'. "
                "Supported values: dense, factorized."
            )
        if isinstance(self.householder_k, bool) or not isinstance(self.householder_k, int):
            raise ValueError(
                f"model.householder_k must be an integer, got {self.householder_k!r}."
            )
        if self.householder_k < 1:
            raise ValueError(f"model.householder_k must be >= 1, got {self.householder_k}.")
        if self.map_family == "factorized" and self.householder_k > self.d:
            # d reflections already generate O(d), so k > d adds parameters, not reach.
            raise ValueError(
                f"model.householder_k must be <= d={self.d} for map_family='factorized', "
                f"got {self.householder_k}."
            )
        if not self.map_scale_eps > 0.0:
            raise ValueError(f"model.map_scale_eps must be positive, got {self.map_scale_eps}.")
        if not 0.0 < self.coupling_gate_init < 1.0:
            raise ValueError(
                f"model.coupling_gate_init must be in (0, 1), got {self.coupling_gate_init}."
            )
        if not 0.0 < self.eta_init < 1.0:
            raise ValueError(f"model.eta_init must be in (0, 1), got {self.eta_init}.")
        if not self.gamma_init > 0.0:
            raise ValueError(
                f"model.gamma_init must be positive (gamma = softplus(theta)), got "
                f"{self.gamma_init}; use ablation.no_path_cells for gamma = 0."
            )
        if not isinstance(self.checkpointing, bool):
            raise ValueError(f"model.checkpointing must be a boolean, got {self.checkpointing!r}.")
        if self.K < 0:
            raise ValueError(f"model.K must be non-negative, got {self.K}.")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        _reject_unknown_keys("model", raw, cls)
        path_weight = str(raw.get("path_weight", "exp"))
        return cls(
            d=int(raw.get("d", 64)),
            d_r=int(raw.get("d_r", 16)),
            encoder_dropout=float(raw.get("encoder_dropout", 0.1)),
            edge_mlp_hidden_dim=int(raw.get("edge_mlp_hidden_dim", 128)),
            face_mlp_hidden_dim=int(raw.get("face_mlp_hidden_dim", 128)),
            classifier_hidden_dim=int(raw.get("classifier_hidden_dim", 128)),
            classifier_dropout=float(raw.get("classifier_dropout", 0.1)),
            graph_pooling=str(raw.get("graph_pooling", "mean")),
            K=int(raw.get("K", 5)),
            layer_dropout=float(raw.get("layer_dropout", 0.0)),
            orthogonal_map=str(raw.get("orthogonal_map", "fasthpp")),
            node_edge_memory_guard_fraction=float(
                raw.get("node_edge_memory_guard_fraction", 0.20)
            ),
            map_family=str(raw.get("map_family", "factorized")),
            # Not cast with int(): 2.9 or `true` must fail validation, not become 2 or 1.
            householder_k=raw.get("householder_k", 2),
            map_scale_eps=float(raw.get("map_scale_eps", 0.1)),
            gamma_init=float(raw.get("gamma_init", 1.0)),
            eta_init=float(raw.get("eta_init", 0.5)),
            coupling_gate_init=float(raw.get("coupling_gate_init", 0.6)),
            # Not cast with bool(): a string such as "false" must fail validation.
            checkpointing=raw.get("checkpointing", True),
            geometry_context=raw.get("geometry_context", True),
            encoder=str(raw.get("encoder", "filter_bank")),
            path_weight=path_weight,
            path_weight_init=float(raw.get("path_weight_init", 0.5)),
            propagation=str(raw.get("propagation", "chebyshev")),
            arm=str(raw.get("arm", "full")),
            tau_init=float(raw.get("tau_init", 1.0)),
            solver_tolerance=float(raw.get("solver_tolerance", 1e-4)),
            solver_warmup_tolerance=float(raw.get("solver_warmup_tolerance", 1e-3)),
            # Not cast with int(): 2.9 or `true` must fail validation.
            solver_warmup_epochs=raw.get("solver_warmup_epochs", 10),
            solver_max_iterations=raw.get("solver_max_iterations", 100),
            tau_stop=float(raw.get("tau_stop", 50.0)),
            log_weight_range_stop=float(raw.get("log_weight_range_stop", 30.0)),
        )


@dataclass(slots=True)
class AblationConfig:
    """Ablation switches of plan §4.6, all off by default (the full model)."""

    # gamma = 0: no path coupling, path operator not built.
    no_path_cells: bool = False
    # Per-incidence orthogonal maps instead of the transport forced by functoriality.
    free_path_maps: bool = False
    # Each path replaced by one gated edge (v_0, v_L) with endpoint-only maps.
    endpoint_edges: bool = False
    # Each path replaced by a closure edge (v_L, v_0) whose relative map is the holonomy.
    holonomy_edges: bool = False
    # Part H, the C2 weight ladder (spec v1.3 §5): the inputs of g. "full" = all nodes' geometry
    # and the actions; "endpoints" = v_0, v_L and the actions (hidden width matched to full's
    # parameter count within 5 %); "schema" = the actions only; "uniform" = no input (w_f = exp(b),
    # or 1 under sigmoid_gamma: the former `uniform_path_gate`). Same cells, same transports.
    # Never tuned.
    path_weight_input: str = "full"
    # Spec v1.4 (E1-E2, R12): g reads each instance node's own h0 (false, the model), or the
    # v1.3 input [h0, c] (true: the appendix pair, full and endpoints only; needs
    # geometry_context). Maps and gates read [h0, c] either way. Never tuned.
    path_weight_context: bool = False
    # D1's per-incidence conformal scales instead of one scale per edge (D1a).
    per_side_scale: bool = False
    # Read out from h0 only (sanity control: is diffusion used at all?).
    h0_only_readout: bool = False
    # Same frame on both sides, no side code and a symmetrized c_e for EVERY relation
    # (shared roles everywhere; the analogue of DSNN's q = 0).
    direction_blind_maps: bool = False
    # D10 (plan §10): node-level listen / broadcast gates, x <- x - eta G Delta B x.
    # "cooperative": both gates; "listen_only": B = I; "symmetric": no gate heads, all gates
    # 1 (the pre-D10 model). Never tuned (see `reject_never_tuned_settings`).
    coupling: str = "cooperative"
    # (D11) "pair_attention" replaces the gates by GATv2-style pair attention on the same sheaf:
    # the test of assumption A4 (plan §12.3). Evidence row only; never a default, never tuned.
    # (D12 G0b) Every restriction map F = I with scale 1: Delta_0 becomes the graph Laplacian
    # over facts (x) I_d, the sheaf-free control of the gap ladder. Diagnostic only; needs
    # no_path_cells; never tuned (`reject_never_tuned_settings`).
    identity_maps: bool = False

    def __post_init__(self) -> None:
        for name in type(self).__dataclass_fields__:
            if name not in ("coupling", "path_weight_input") and not isinstance(
                    getattr(self, name), bool):
                raise ValueError(f"ablation.{name} must be a boolean, got {getattr(self, name)!r}.")
        if self.coupling not in COUPLING_MODES:
            raise ValueError(
                f"ablation.coupling must be one of {COUPLING_MODES}, got {self.coupling!r}."
            )
        if self.path_weight_input not in PATH_WEIGHT_INPUTS:
            raise ValueError(
                f"ablation.path_weight_input must be one of {PATH_WEIGHT_INPUTS}, got "
                f"{self.path_weight_input!r}."
            )
        if self.coupling == "pair_attention" and (
            self.per_side_scale or self.free_path_maps or self.endpoint_edges
            or self.holonomy_edges or self.path_weight_input != "full" or self.h0_only_readout
        ):
            raise ValueError(
                "ablation.coupling: pair_attention needs orthogonal transports and the path "
                "encoder; it cannot be combined with per_side_scale, free_path_maps, "
                "endpoint_edges, holonomy_edges, path_weight_input other than full or "
                "h0_only_readout."
            )
        if self.path_weight_input != "full" and (
            self.no_path_cells or self.free_path_maps or self.endpoint_edges or self.holonomy_edges
        ):
            raise ValueError(
                "ablation.path_weight_input restricts g on identical cells (the C2 ladder); it "
                "cannot be combined with no_path_cells, free_path_maps, endpoint_edges or "
                "holonomy_edges."
            )
        if self.path_weight_context and (
            self.no_path_cells or self.path_weight_input not in ("full", "endpoints")
            or self.endpoint_edges or self.free_path_maps or self.coupling == "pair_attention"
        ):
            raise ValueError(
                "ablation.path_weight_context (g reads [h0, c], the spec v1.4 appendix pair) needs "
                "path cells and path_weight_input full or endpoints, on the cell ladder (not "
                "endpoint_edges, free_path_maps or pair_attention, spec v1.6); schema and uniform "
                "read no node at all."
            )
        if self.identity_maps and (
            not self.no_path_cells or self.h0_only_readout or self.coupling == "pair_attention"
        ):
            raise ValueError(
                "ablation.identity_maps (sheaf-free diagnostic) needs no_path_cells: true and "
                "cannot be combined with h0_only_readout or coupling: pair_attention."
            )
        if self.endpoint_edges and self.holonomy_edges:
            raise ValueError("ablation.endpoint_edges and ablation.holonomy_edges are exclusive.")
        if self.free_path_maps and (
            self.endpoint_edges or self.holonomy_edges or self.per_side_scale
        ):
            raise ValueError(
                "ablation.free_path_maps ablates the path cells under per-edge scales: it "
                "cannot be combined with endpoint_edges, holonomy_edges or per_side_scale."
            )
        if self.no_path_cells and (
            self.free_path_maps or self.endpoint_edges or self.holonomy_edges
        ):
            raise ValueError(
                "ablation.no_path_cells removes the path term; it cannot be combined with "
                "free_path_maps, endpoint_edges or holonomy_edges."
            )

    @property
    def map_scale(self) -> str:
        return "per_side" if self.per_side_scale else "per_edge"

    @property
    def path_mode(self) -> str:
        if self.endpoint_edges:
            return "endpoint_edges"
        if self.holonomy_edges:
            return "holonomy_edges"
        return "cells"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AblationConfig":
        _reject_unknown_keys("ablation", raw, cls)
        defaults = cls()
        # Not cast with bool(): a string such as "false" must fail validation.
        return cls(**{name: raw.get(name, getattr(defaults, name))
                      for name in cls.__dataclass_fields__})


@dataclass(slots=True)
class OptimConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Spec v1.15 §6 (J4): no learning-rate scheduler and no gradient clipping for any method.
    # `grad_clip <= 0` means no clipping.
    scheduler: str = "none"
    grad_clip: float = 0.0
    # Learning rate of the diffusion scalars theta_eta, theta_gamma and the coupling-gate heads,
    # as a multiple of `lr` (own parameter group, no weight decay). Full-batch training takes one optimizer step
    # per epoch, so at the base rate these scalars barely move (plan §4.7).
    scalar_lr_scale: float = 10.0

    def __post_init__(self) -> None:
        if not self.scalar_lr_scale > 0.0:
            raise ValueError(
                f"optim.scalar_lr_scale must be positive, got {self.scalar_lr_scale}."
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OptimConfig":
        _reject_unknown_keys("optim", raw, cls)
        return cls(
            name=str(raw.get("name", "adamw")),
            lr=float(raw.get("lr", 1e-3)),
            weight_decay=float(raw.get("weight_decay", 1e-4)),
            scheduler=str(raw.get("scheduler", "none")),
            grad_clip=float(raw.get("grad_clip", 0.0)),
            scalar_lr_scale=float(raw.get("scalar_lr_scale", 10.0)),
        )


@dataclass(slots=True)
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    epochs: int = 200
    seed: int = 42
    device: str = "cuda"
    exp_name: str = "h2snn"
    output_dir: Path = Path("experiments")
    log_every: int = 10
    checkpoint_metric: str = "acc"
    checkpoint_mode: str = "max"
    early_stopping_patience: int = 100
    use_wandb: bool = False
    wandb_project: str = "h2snn"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    wandb_group: str = ""
    wandb_tags: tuple[str, ...] = field(default_factory=tuple)
    wandb_log_artifacts: bool = False

    def __post_init__(self) -> None:
        if self.ablation.path_weight_context and not self.model.geometry_context:
            raise ValueError(
                "ablation.path_weight_context lets g read [h0, c]; c exists only under "
                "model.geometry_context: true."
            )
        if self.model.propagation == "anchored":
            # v3-B (D-B1): no gates; the arms replace the legacy path switches; orthogonal maps
            # need one scale per fact (D1a); the sheaf-free diagnostic has no anchored form.
            if self.ablation.coupling != "symmetric":
                raise ValueError(
                    "model.propagation: anchored has no listen / broadcast gates (spec v1.11 D-B1, "
                    "L1'); set ablation.coupling: symmetric."
                )
            switched = [name for name in ("no_path_cells", "free_path_maps", "endpoint_edges",
                                          "holonomy_edges", "identity_maps", "per_side_scale")
                        if getattr(self.ablation, name)]
            if switched:
                raise ValueError(
                    f"ablation.{', '.join(switched)} belong to the legacy propagation; under "
                    "model.propagation: anchored use model.arm (no_path_cells, closure_only, "
                    "free_cell_maps, ...) with per-edge scales."
                )

    @property
    def exp_dir(self) -> Path:
        return self.output_dir / self.exp_name

    @property
    def task_type(self) -> str:
        return self.data.task_type

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        _reject_unknown_keys("top-level config", raw, cls)
        data = DataConfig.from_dict(raw.get("data", {}))
        # Spec v1.15 §6 (J4, M9): unset epochs, patience and selection metric come from the
        # task's source protocol; an explicit value is kept (smoke runs, tests).
        budget = protocol_budget(data.task_type, data.dataset_name)
        default_patience = budget.patience if budget.patience is not None else budget.epochs
        default_epochs = budget.epochs
        config = cls(
            data=data,
            model=ModelConfig.from_dict(raw.get("model", {})),
            ablation=AblationConfig.from_dict(raw.get("ablation", {})),
            optim=OptimConfig.from_dict(raw.get("optim", {})),
            epochs=int(raw.get("epochs", default_epochs)),
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "cuda")),
            exp_name=str(raw.get("exp_name", "h2snn")),
            output_dir=Path(raw.get("output_dir", "experiments")),
            log_every=int(raw.get("log_every", 10)),
            # GC selects configurations by validation Macro-F1 (its best.pt artifact too); its
            # early stopping on validation loss is the trainer's (M5).
            checkpoint_metric=str(raw.get("checkpoint_metric", budget.monitor
                                          if budget.monitor != "loss" else "macro_f1")),
            checkpoint_mode=str(raw.get("checkpoint_mode", "max")),
            early_stopping_patience=int(raw.get(
                "early_stopping_patience",
                default_patience if "epochs" not in raw or budget.patience is not None
                else int(raw["epochs"]))),
            use_wandb=bool(raw.get("use_wandb", False)),
            wandb_project=str(raw.get("wandb_project", "h2snn")),
            wandb_entity=str(raw.get("wandb_entity", "")),
            wandb_mode=str(raw.get("wandb_mode", "online")),
            wandb_group=str(raw.get("wandb_group", "")),
            wandb_tags=tuple(str(item) for item in raw.get("wandb_tags", [])),
            wandb_log_artifacts=bool(raw.get("wandb_log_artifacts", False)),
        )
        if config.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be positive, "
                f"got {config.early_stopping_patience}."
            )
        return config


# ----------------------------------------------------------------- fixed-literature-schema row
def is_literature_row(data: DataConfig) -> bool:
    """A resolved configuration of the fixed-literature-schema row (spec v1.19 §6): node
    classification, path_source: schemas, r >= 1 (the legacy schema rows have r = 0)."""
    return (data.task_type == "node_classification" and data.path_source == "schemas"
            and data.wisp_walks_per_anchor >= 1)


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            out.update(_flatten(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: payload}


def fixed_literature_schema_config(full: TrainConfig) -> TrainConfig:
    """The fixed-literature-schema row of a tuned `full` configuration (spec v1.19 §6; card Part
    O O3): the same configuration with one change, `data.path_source: schemas`; the row then draws
    the dataset's `LITERATURE_METAPATHS` with the tuned r split evenly (`literature_walk_split`).
    Built through the saved-configuration round trip (`dataclass_to_dict` -> `from_dict`), the
    path a server run takes. Raises unless `full` is a resolved v3-B anchored `full` WISP node-
    classification configuration of a dataset with a list, and if any setting other than the path
    source differs between the two configurations. The run location (`exp_name`, `output_dir`) is
    the caller's: the row must not share the main run's directory."""
    from src.utils.misc import dataclass_to_dict  # lazy: misc imports torch, config does not

    data = full.data
    if data.task_type != "node_classification" or data.path_source != "wisp":
        raise ValueError("the fixed-literature-schema row is built from a resolved WISP node-"
                         f"classification configuration (got {data.task_type!r}, "
                         f"{data.path_source!r}).")
    if full.model.propagation != "anchored" or full.model.arm != "full":
        raise ValueError("the fixed-literature-schema row is built from the v3-B main model "
                         f"(propagation anchored, arm full), got {full.model.propagation!r}, "
                         f"{full.model.arm!r}.")
    if data.dataset_name not in LITERATURE_METAPATHS:
        raise ValueError(f"no literature list for {data.dataset_name!r} "
                         f"(the row exists on {sorted(LITERATURE_METAPATHS)} only).")
    before = dataclass_to_dict(full)
    payload = dataclass_to_dict(full)
    payload["data"]["path_source"] = "schemas"
    row = TrainConfig.from_dict(payload)
    old, new = _flatten(before), _flatten(dataclass_to_dict(row))
    changed = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
    if changed != ["data.path_source"]:
        raise ValueError("the fixed-literature-schema row may differ from its full configuration "
                         f"in data.path_source only; differing settings: {changed}.")
    return row
