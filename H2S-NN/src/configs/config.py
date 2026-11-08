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
# Part H (WISP, spec v1.3 §3.2): the path source and its support are never tuned; the walks per
# anchor `r` (precision) is calibrated to an incidence budget, not tuned.
PATH_SOURCES: tuple[str, ...] = ("schemas", "wisp")
NEVER_TUNED_WISP_KEYS: tuple[str, ...] = ("path_source", "wisp_max_length", "path_weight",
                                           "path_weight_init", "path_weight_input",
                                           "wisp_proposal", "path_weight_context",
                                           "wisp_walks_per_anchor")
# Part H3: q = p ("uniform") or the adaptive defensive proposal (variance only; fixed by a gate).
WISP_PROPOSALS: tuple[str, ...] = ("uniform", "adaptive")
# Cell weight (Part H): "exp" = w_f = exp(g(xi_f)), gamma removed (it is the bias of g; spec v1.3
# §3.2c); "sigmoid_gamma" = gamma * sigmoid(g) (the D11/D12 behaviour, comparison arm).
PATH_WEIGHTS: tuple[str, ...] = ("exp", "sigmoid_gamma")
# The C2 weight ladder (spec v1.3 §5): inputs of g, nested restrictions on identical cells.
PATH_WEIGHT_INPUTS: tuple[str, ...] = ("full", "endpoints", "schema", "uniform")
FEATURE_NORMALIZATIONS: tuple[str, ...] = ("none", "row")


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
    if coupling != "cooperative":
        raise ValueError(
            f"ablation.coupling: {coupling} is an ablation / test row (plan §10.5, §12.3) and must "
            "not be used in tuning; tune the cooperative model and report the row separately."
        )
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
    if node_classification:
        # Spec v1.6 §6 / E11: l_max, the proposal and r are pre-registered, never tuned.
        if int(data.get("wisp_max_length", 4)) != 4:
            raise ValueError("data.wisp_max_length is pre-registered (4 on HGB, spec §6) and must "
                             "not differ in a tuning base config (the U3 curve is analysis).")
        if str(data.get("wisp_proposal", "uniform")) != "uniform":
            raise ValueError("data.wisp_proposal: adaptive is not the pre-registered proposal "
                             "(spec §6) and must not be used in a tuning base config.")
        walks = int(data.get("wisp_walks_per_anchor", 0))
        calibrated = WISP_WALKS_PER_ANCHOR.get(dataset, {}).get(4)
        if walks not in (0, calibrated):
            raise ValueError(
                f"data.wisp_walks_per_anchor {walks} is neither 0 nor the calibrated value "
                f"{calibrated} for {dataset!r} (spec v1.6 E11: r is never tuned).")
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
        # HGB local-format LastFM (data/LastFM): node.dat/link.dat/link.dat.test,
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

GRAPH_CLASSIFICATION_PRESETS: dict[str, dict[str, Any]] = {
    "mutag": _GRAPH_HOLDOUT_PRESET,
    "proteins": _GRAPH_HOLDOUT_PRESET,
    "enzymes": _GRAPH_HOLDOUT_PRESET,
    "nci1": _GRAPH_HOLDOUT_PRESET,
}

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
    # Path source (Part H; default flipped by spec v1.4 E7): "wisp" = schema-free walks from the
    # stated target measure (spec §3.2), node classification only; "schemas" = the D8 walk
    # sampler over the literature presets (the fixed-literature-schema comparison row). "" (unset)
    # resolves by task: wisp for node classification, schemas for link prediction and graph
    # classification (their samplers do not read it). `wisp_max_length` is the support l_max;
    # `wisp_walks_per_anchor` is r (precision): 0 (unset) resolves under wisp to the dataset's
    # incidence-matching value (spec §6; NODE_CLASSIFICATION_PRESETS).
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
    neg_sampling_mode: str = "fixed"              # "fixed" | "resample_each_epoch"
    neg_sampling_ratio: int = 1
    link_val_ratio: float = 0.09
    link_test_ratio: float = 0.10

    # --- graph classification ---
    split_protocol: str = "stratified_holdout"    # "stratified_holdout" | "precomputed_folds" | "stratified_kfold" | "repeated_kfold"
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
        if self.path_source == "schemas" and (
            self.wisp_max_length != 4 or self.wisp_walks_per_anchor != 0
            or self.wisp_proposal != "uniform"
        ):
            raise ValueError(
                "data.wisp_max_length, wisp_walks_per_anchor and wisp_proposal configure "
                "path_source: wisp; they are rejected under path_source: schemas (got "
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
            neg_sampling_mode=str(raw.get("neg_sampling_mode", "fixed")),
            neg_sampling_ratio=int(raw.get("neg_sampling_ratio", 1)),
            link_val_ratio=float(raw.get("link_val_ratio", 0.09)),
            link_test_ratio=float(raw.get("link_test_ratio", 0.10)),
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
        # Spec v1.4 E7: unset path_source resolves by task; unset r resolves to the preset.
        path_source = self.path_source or (
            "wisp" if task_type == "node_classification" else "schemas")
        if path_source not in PATH_SOURCES:
            raise ValueError(
                f"data.path_source must be one of {PATH_SOURCES}, got {self.path_source!r}."
            )
        walks = self.wisp_walks_per_anchor
        if path_source == "wisp" and walks == 0:
            table = WISP_WALKS_PER_ANCHOR.get(dataset_name, {})
            if self.wisp_max_length not in table:
                raise ValueError(
                    f"no calibrated data.wisp_walks_per_anchor for dataset {dataset_name!r} at "
                    f"wisp_max_length {self.wisp_max_length} (calibrated: {sorted(table)}); set r "
                    "explicitly (spec v1.6 E11).")
            walks = table[self.wisp_max_length]
        if self.wisp_max_length < 2:
            raise ValueError(f"data.wisp_max_length must be >= 2, got {self.wisp_max_length}.")
        if walks < (1 if path_source == "wisp" else 0):
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
        if path_source == "wisp":
            if task_type != "node_classification":
                raise ValueError(
                    "data.path_source: wisp is implemented for node classification only; link "
                    "prediction and graph classification keep their schema-free fixed-length sampler."
                )
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
            supported_split_protocols = {
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
            if self.split_protocol != "stratified_holdout":
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
        if self.metapath_templates:
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
            "neg_sampling_mode", self.neg_sampling_mode, "fixed"
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

    def __post_init__(self) -> None:
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
    scheduler: str = "cosine"
    grad_clip: float = 1.0
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
            scheduler=str(raw.get("scheduler", "cosine")),
            grad_clip=float(raw.get("grad_clip", 1.0)),
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

    @property
    def exp_dir(self) -> Path:
        return self.output_dir / self.exp_name

    @property
    def task_type(self) -> str:
        return self.data.task_type

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        _reject_unknown_keys("top-level config", raw, cls)
        config = cls(
            data=DataConfig.from_dict(raw.get("data", {})),
            model=ModelConfig.from_dict(raw.get("model", {})),
            ablation=AblationConfig.from_dict(raw.get("ablation", {})),
            optim=OptimConfig.from_dict(raw.get("optim", {})),
            epochs=int(raw.get("epochs", 200)),
            seed=int(raw.get("seed", 42)),
            device=str(raw.get("device", "cuda")),
            exp_name=str(raw.get("exp_name", "h2snn")),
            output_dir=Path(raw.get("output_dir", "experiments")),
            log_every=int(raw.get("log_every", 10)),
            checkpoint_metric=str(raw.get("checkpoint_metric", "acc")),
            checkpoint_mode=str(raw.get("checkpoint_mode", "max")),
            early_stopping_patience=int(raw.get("early_stopping_patience", 100)),
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
