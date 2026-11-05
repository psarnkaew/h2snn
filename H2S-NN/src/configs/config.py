from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


AMPSS_PROTOCOL_VERSION = "schema_conditioned_v1"


@dataclass(slots=True)
class MetaPathConfig:
    name: str
    node_types: tuple[str, ...]
    sample_per_anchor: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetaPathConfig":
        return cls(
            name=str(raw["name"]),
            node_types=tuple(str(value) for value in raw["node_types"]),
            sample_per_anchor=raw.get("sample_per_anchor"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_types": list(self.node_types),
            "sample_per_anchor": self.sample_per_anchor,
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
# HINPool's evaluation protocol (ref/HINPOOL.pdf, and hinpool/utils/utils.py:18-48):
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
    # Scale on the synthetic closure 1-cell's restriction maps. 1.0 = a full first-class
    # 1-cell (the honest 2-complex). The old 1e-6 default effectively disabled it, back when
    # the closure edge sat outside delta0 and only fed a separate `closure_L0` correction.
    # The face map absorbs a compensating 1/closure_scale, so delta1.delta0 = 0 holds for any
    # positive value; only the closure edge's L0 contribution (quadratic in the scale) varies.
    closure_scale: float = 1.0

    # --- link prediction ---
    target_relation: tuple[str, str, str] = ("", "", "")
    decoder_type: str = "dot"                    # "dot" | "distmult" | "concat_mlp"
    neg_sampling_mode: str = "fixed"              # "fixed" | "resample_each_epoch"
    neg_sampling_ratio: int = 1
    link_val_ratio: float = 0.09
    link_test_ratio: float = 0.10

    # --- graph classification ---
    pooling_type: str = "meanpool"               # "meanpool" | "grampool" | "s2pool" | "sheafpool" | "sheafcov"
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataConfig":
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
            closure_scale=float(raw.get("closure_scale", 1.0)),
            target_relation=tuple(raw.get("target_relation", ("", "", ""))),
            decoder_type=str(raw.get("decoder_type", "dot")),
            neg_sampling_mode=str(raw.get("neg_sampling_mode", "fixed")),
            neg_sampling_ratio=int(raw.get("neg_sampling_ratio", 1)),
            link_val_ratio=float(raw.get("link_val_ratio", 0.09)),
            link_test_ratio=float(raw.get("link_test_ratio", 0.10)),
            pooling_type=str(raw.get("pooling_type", "meanpool")),
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
        if self.closure_scale <= 0.0:
            raise ValueError("data.closure_scale must be positive.")
        if self.face_sample_k <= 0:
            raise ValueError("data.face_sample_k must be positive.")
        task_type = self.task_type or DATASET_TASK_TYPES.get(dataset_name, "")
        if not task_type:
            raise ValueError(
                f"Unknown dataset '{self.dataset_name}'; set data.task_type explicitly."
            )

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
            supported_pools = {
                "meanpool",
                "grampool",
                "s2pool",
                "sheafpool",
                "sheafcov",
                "sheafconv",
                "ths_spectral",
                "ths_galerkin",
                "s2pool_h2",
            }
            if self.pooling_type not in supported_pools:
                raise ValueError(f"Unsupported data.pooling_type '{self.pooling_type}'.")
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
            )
            for template in templates
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
class SchemaDiscoveryConfig:
    enabled: bool = False
    mode: str = "manual"                 # "manual" | "auto"
    protocol_version: str = AMPSS_PROTOCOL_VERSION

    max_schema_length: int = 4
    min_schema_length: int = 2
    max_schema_candidates: int = 128
    selected_schema_k: int = 8

    max_instances_per_schema: int = 20000
    max_instances_per_anchor: int = 50
    max_total_instances: int | None = None  # global cap across the whole selected pool; None = unbounded
    max_total_incidences: int | None = None
    max_schema_identities: int = 4096
    selector_strategy: str = "probe"
    selector_seed: int = 42
    materialization_seed: int = 42
    selector_max_examples: int = 4096
    selector_holdout_ratio: float = 0.10
    selector_num_attempts: int = 50
    selector_probe_l2: float = 1e-3
    selector_probe_max_iter: int = 50
    selector_stop_tolerance: float = 1e-4
    selector_tie_tolerance: float = 1e-8
    selector_min_examples_per_label: int = 1

    require_target_start: bool = True
    require_target_end: bool = True
    allow_repeated_node_types: bool = True
    exclude_identity_metapaths: bool = True

    instance_weight_mode: str = "schema_conditioned"
    warmup_epochs: int = 0               # epochs before learned weights become active
    operator_anneal_epochs: int = 5

    def __post_init__(self) -> None:
        if self.protocol_version != AMPSS_PROTOCOL_VERSION:
            raise ValueError(
                "Unsupported AMPSS protocol version "
                f"{self.protocol_version!r}; expected {AMPSS_PROTOCOL_VERSION!r}."
            )
        if self.enabled and self.mode == "auto" and self.selector_strategy.lower() != "probe":
            raise ValueError(
                "Only discovery.selector_strategy='probe' is supported by "
                f"{AMPSS_PROTOCOL_VERSION}; legacy DCV selectors are diagnostics-only."
            )
        if self.selector_seed < 0 or self.materialization_seed < 0:
            raise ValueError("selector_seed and materialization_seed must be non-negative.")
        if self.instance_weight_mode not in {
            "schema_conditioned",
            "schema_agnostic",
            "uniform",
        }:
            raise ValueError(
                "discovery.instance_weight_mode must be one of: "
                "schema_conditioned, schema_agnostic, uniform."
            )
        if self.min_schema_length < 2 or self.max_schema_length < self.min_schema_length:
            raise ValueError(
                "Discovery face schemas require 2 <= min_schema_length <= max_schema_length."
            )
        if self.max_schema_identities <= 0 or self.max_schema_candidates <= 0:
            raise ValueError("Schema identity and candidate safety guards must be positive.")
        if self.selected_schema_k <= 0:
            raise ValueError("selected_schema_k must be positive.")
        if self.max_instances_per_schema <= 0 or self.max_instances_per_anchor <= 0:
            raise ValueError("Discovery instance quotas must be positive.")
        if self.max_total_instances is not None and self.max_total_instances <= 0:
            raise ValueError("max_total_instances must be positive when provided.")
        if self.max_total_incidences is not None and self.max_total_incidences <= 0:
            raise ValueError("max_total_incidences must be positive when provided.")
        if self.enabled and self.mode == "auto" and self.max_total_instances is None:
            raise ValueError(
                "Auto discovery requires max_total_instances so the selected cell pool is bounded."
            )
        if self.enabled and self.mode == "auto" and self.max_total_incidences is None:
            raise ValueError(
                "Auto discovery requires max_total_incidences; incidence-aware selection "
                "must be active rather than reported as an unused objective."
            )
        if self.warmup_epochs < 0 or self.operator_anneal_epochs < 1:
            raise ValueError(
                "Discovery warmup must be non-negative and operator_anneal_epochs must be >= 1."
            )
        if not 0.0 < self.selector_holdout_ratio < 1.0:
            raise ValueError("selector_holdout_ratio must be in (0, 1).")
        if self.selector_max_examples < 2:
            raise ValueError("selector_max_examples must be at least 2.")
        if self.selector_num_attempts <= 0:
            raise ValueError("selector_num_attempts must be positive.")
        if self.selector_probe_l2 <= 0.0:
            raise ValueError("selector_probe_l2 must be positive.")
        if self.selector_probe_max_iter <= 0:
            raise ValueError("selector_probe_max_iter must be positive.")
        if self.selector_stop_tolerance <= 0.0 or self.selector_tie_tolerance <= 0.0:
            raise ValueError("Selector stop/tie tolerances must be positive.")
        if self.selector_min_examples_per_label <= 0:
            raise ValueError("selector_min_examples_per_label must be positive.")
        if self.selector_max_examples < 2 * self.selector_min_examples_per_label:
            raise ValueError(
                "selector_max_examples must permit nonempty fit and selection splits."
            )
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SchemaDiscoveryConfig":
        legacy_keys = sorted(
            key
            for key in (
                "objective_version",
                "selector_coverage_weight",
                "schema_gate_type",
                "instance_gate_type",
                "partial_schema_probability",
                "schema_active_threshold",
                "lambda_schema_sparse",
                "lambda_instance_sparse",
                "lambda_coverage",
                "lambda_incidence_mass",
                "lambda_gate_entropy",
                "lambda_redundancy",
                "lambda_cost",
                "selector_incidence_budget",
            )
            if key in raw
        )
        if legacy_keys:
            raise ValueError(
                "Unsupported legacy discovery keys: "
                f"{', '.join(legacy_keys)}. AMPSS now uses task-only "
                "schema-conditioned instance weights; remove these keys and start "
                "a fresh run/sweep."
            )
        max_total_instances = raw.get("max_total_instances")
        max_total_incidences = raw.get("max_total_incidences")
        max_instances_per_anchor = int(raw.get("max_instances_per_anchor", 50))
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "manual")),
            protocol_version=str(
                raw.get("protocol_version", AMPSS_PROTOCOL_VERSION)
            ),
            max_schema_length=int(raw.get("max_schema_length", 4)),
            min_schema_length=int(raw.get("min_schema_length", 2)),
            max_schema_candidates=int(raw.get("max_schema_candidates", 128)),
            selected_schema_k=int(raw.get("selected_schema_k", 8)),
            max_instances_per_schema=int(raw.get("max_instances_per_schema", 20000)),
            max_instances_per_anchor=max_instances_per_anchor,
            max_total_instances=(
                None if max_total_instances is None else int(max_total_instances)
            ),
            max_total_incidences=(
                None
                if max_total_incidences is None
                else int(max_total_incidences)
            ),
            max_schema_identities=int(raw.get("max_schema_identities", 4096)),
            selector_strategy=str(raw.get("selector_strategy", "probe")),
            selector_seed=int(raw.get("selector_seed", 42)),
            materialization_seed=int(raw.get("materialization_seed", 42)),
            selector_max_examples=int(raw.get("selector_max_examples", 4096)),
            selector_holdout_ratio=float(raw.get("selector_holdout_ratio", 0.10)),
            selector_num_attempts=int(
                raw.get("selector_num_attempts", max_instances_per_anchor)
            ),
            selector_probe_l2=float(raw.get("selector_probe_l2", 1e-3)),
            selector_probe_max_iter=int(raw.get("selector_probe_max_iter", 50)),
            selector_stop_tolerance=float(raw.get("selector_stop_tolerance", 1e-4)),
            selector_tie_tolerance=float(raw.get("selector_tie_tolerance", 1e-8)),
            selector_min_examples_per_label=int(
                raw.get("selector_min_examples_per_label", 1)
            ),
            require_target_start=bool(raw.get("require_target_start", True)),
            require_target_end=bool(raw.get("require_target_end", True)),
            allow_repeated_node_types=bool(raw.get("allow_repeated_node_types", True)),
            exclude_identity_metapaths=bool(raw.get("exclude_identity_metapaths", True)),
            instance_weight_mode=str(
                raw.get("instance_weight_mode", "schema_conditioned")
            ),
            warmup_epochs=int(raw.get("warmup_epochs", 0)),
            operator_anneal_epochs=int(raw.get("operator_anneal_epochs", 5)),
        )


@dataclass(slots=True)
class ModelConfig:
    d: int = 64
    d_r: int = 16
    d_h: int = 64
    face_chunk_size: int = 4096
    encoder_dropout: float = 0.1
    edge_mlp_hidden_dim: int = 128
    edge_init_hidden_dim: int = 128
    face_mlp_hidden_dim: int = 128
    classifier_hidden_dim: int = 128
    # Width of the learned descriptor after the fixed 3x3 SheafCov Gram matrix.
    # This is a graph-readout capacity control, not the number of Gram views.
    sheafcov_feat_dim: int = 8
    K: int = 5
    eta_0: float = 0.1
    eta_1: float = 0.1
    learnable_eta: bool = True
    layer_dropout: float = 0.0
    use_orthogonal: bool = True
    orthogonal_map: str = "fasthpp"
    node_edge_memory_guard_fraction: float = 0.20
    # Node->edge restriction-map family (v2 decision D1). "dense" is v1: per-edge [M, d, d]
    # FastHPP maps. "factorized" is F = c * R_{relation,side} * H_k(u), applied matrix-free
    # (src/models/map_operators.py); node classification only.
    map_family: str = "dense"
    # Householder reflections k per incidence in the factorized family, 1 <= k <= d.
    householder_k: int = 2
    # Lower bound eps of the factorized conformal scale c = softplus(.) + eps. Bounds the
    # closed-form inverse 1/c used by the face transport.
    map_scale_eps: float = 0.1
    use_delta0_init: bool = True
    classifier_dropout: float = 0.1
    # How the free per-instance face map A_f is produced. "concat" concatenates the path's
    # node states in order with the schema's relation embeddings and applies one MLP per
    # distinct path length; "lstm" is the superseded token-MLP + LSTM + attention-pool stack,
    # retained only so the sequence-encoder ablation can be run.
    face_encoder: str = "concat"
    # Parameterisation of A_f. The model consumes A_f only through A_f^T A_f (it is a pure
    # left factor of every edge-to-face map in a face), so a d x d output carries d(d-1)/2
    # unidentifiable gauge directions. "triangular" emits d(d+1)/2 numbers into the upper
    # triangle instead, which is lossless because every PSD matrix is R^T R for upper
    # triangular R.
    face_map_param: str = "full"
    ablate_no_faces: bool = False
    ablate_no_delta1: bool = False
    ablate_identity_maps: bool = False
    # Superseded historical hierarchy controls; SheafCovPool does not consume them.
    pool_num_levels: int = 2
    pool_ratio: float = 0.5
    pool_inject_maps: bool = False
    pool_clusters_per_type: int = 4
    pool_coarse_steps: int = 4
    pool_use_coarse: bool = True
    pool_use_spectral: bool = True

    def __post_init__(self) -> None:
        if self.orthogonal_map != "fasthpp":
            raise ValueError(
                f"Unsupported model.orthogonal_map '{self.orthogonal_map}'. "
                "The current H2SNN supports only 'fasthpp' for learned orthogonal maps."
            )
        if not 0.0 < self.node_edge_memory_guard_fraction <= 1.0:
            raise ValueError(
                "model.node_edge_memory_guard_fraction must be in (0, 1], "
                f"got {self.node_edge_memory_guard_fraction}."
            )
        if self.sheafcov_feat_dim <= 0:
            raise ValueError(
                "model.sheafcov_feat_dim must be positive, "
                f"got {self.sheafcov_feat_dim}."
            )
        supported_map_families = {"dense", "factorized"}
        if self.map_family not in supported_map_families:
            supported = ", ".join(sorted(supported_map_families))
            raise ValueError(
                f"Unsupported model.map_family '{self.map_family}'. "
                f"Supported values: {supported}."
            )
        if isinstance(self.householder_k, bool) or not isinstance(self.householder_k, int):
            raise ValueError(
                f"model.householder_k must be an integer, got {self.householder_k!r}."
            )
        if self.householder_k < 1:
            raise ValueError(
                f"model.householder_k must be >= 1, got {self.householder_k}."
            )
        if not self.map_scale_eps > 0.0:
            raise ValueError(
                f"model.map_scale_eps must be positive, got {self.map_scale_eps}."
            )
        if self.map_family == "factorized":
            # d reflections already generate O(d), so k > d adds parameters, not reach.
            if self.householder_k > self.d:
                raise ValueError(
                    f"model.householder_k must be <= d={self.d} for map_family='factorized', "
                    f"got {self.householder_k}."
                )
            # Factorized maps are conformal-orthogonal by construction, and the identity
            # ablation would materialize [M, d, d]; neither v1 switch has a D1 meaning.
            if not self.use_orthogonal:
                raise ValueError(
                    "model.use_orthogonal=false selects unconstrained dense maps and is "
                    "incompatible with map_family='factorized' (always conformal-orthogonal)."
                )
            if self.ablate_identity_maps:
                raise ValueError(
                    "model.ablate_identity_maps is a dense-map (v1) ablation and is not "
                    "implemented for map_family='factorized' (D1 scope)."
                )
        supported_encoders = {"concat", "lstm"}
        if self.face_encoder not in supported_encoders:
            supported = ", ".join(sorted(supported_encoders))
            raise ValueError(
                f"Unsupported model.face_encoder '{self.face_encoder}'. "
                f"Supported values: {supported}."
            )
        supported_face_params = {"full", "triangular"}
        if self.face_map_param not in supported_face_params:
            supported = ", ".join(sorted(supported_face_params))
            raise ValueError(
                f"Unsupported model.face_map_param '{self.face_map_param}'. "
                f"Supported values: {supported}."
            )
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        legacy_gate_flags = sorted(
            key
            for key in (
                "ablate_uniform_gates",
                "ablate_schema_gate_only",
                "ablate_instance_gate_only",
            )
            if key in raw
        )
        if legacy_gate_flags:
            raise ValueError(
                "Unsupported legacy AMPSS model keys: "
                f"{', '.join(legacy_gate_flags)}. Use "
                "discovery.instance_weight_mode instead."
            )
        return cls(
            d=int(raw.get("d", 64)),
            d_r=int(raw.get("d_r", 16)),
            d_h=int(raw.get("d_h", 64)),
            face_chunk_size=int(raw.get("face_chunk_size", 4096)),
            encoder_dropout=float(raw.get("encoder_dropout", 0.1)),
            edge_mlp_hidden_dim=int(raw.get("edge_mlp_hidden_dim", 128)),
            edge_init_hidden_dim=int(raw.get("edge_init_hidden_dim", 128)),
            face_mlp_hidden_dim=int(raw.get("face_mlp_hidden_dim", 128)),
            classifier_hidden_dim=int(raw.get("classifier_hidden_dim", 128)),
            sheafcov_feat_dim=int(raw.get("sheafcov_feat_dim", 8)),
            K=int(raw.get("K", 5)),
            eta_0=float(raw.get("eta_0", 0.1)),
            eta_1=float(raw.get("eta_1", 0.1)),
            learnable_eta=bool(raw.get("learnable_eta", True)),
            layer_dropout=float(raw.get("layer_dropout", 0.0)),
            use_orthogonal=bool(raw.get("use_orthogonal", True)),
            orthogonal_map=str(raw.get("orthogonal_map", "fasthpp")),
            node_edge_memory_guard_fraction=float(
                raw.get("node_edge_memory_guard_fraction", 0.20)
            ),
            map_family=str(raw.get("map_family", "dense")),
            # Not cast with int(): 2.9 or `true` must fail validation, not become 2 or 1.
            householder_k=raw.get("householder_k", 2),
            map_scale_eps=float(raw.get("map_scale_eps", 0.1)),
            use_delta0_init=bool(raw.get("use_delta0_init", True)),
            classifier_dropout=float(raw.get("classifier_dropout", 0.1)),
            face_encoder=str(raw.get("face_encoder", "concat")),
            face_map_param=str(raw.get("face_map_param", "full")),
            ablate_no_faces=bool(raw.get("ablate_no_faces", False)),
            ablate_no_delta1=bool(raw.get("ablate_no_delta1", False)),
            ablate_identity_maps=bool(raw.get("ablate_identity_maps", False)),
            pool_num_levels=int(raw.get("pool_num_levels", 2)),
            pool_ratio=float(raw.get("pool_ratio", 0.5)),
            pool_inject_maps=bool(raw.get("pool_inject_maps", False)),
            pool_clusters_per_type=int(raw.get("pool_clusters_per_type", 4)),
            pool_coarse_steps=int(raw.get("pool_coarse_steps", 4)),
            pool_use_coarse=bool(raw.get("pool_use_coarse", True)),
            pool_use_spectral=bool(raw.get("pool_use_spectral", True)),
        )


@dataclass(slots=True)
class OptimConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    grad_clip: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OptimConfig":
        return cls(
            name=str(raw.get("name", "adamw")),
            lr=float(raw.get("lr", 1e-3)),
            weight_decay=float(raw.get("weight_decay", 1e-4)),
            scheduler=str(raw.get("scheduler", "cosine")),
            grad_clip=float(raw.get("grad_clip", 1.0)),
        )


@dataclass(slots=True)
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    discovery: SchemaDiscoveryConfig = field(default_factory=SchemaDiscoveryConfig)
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

    @property
    def exp_dir(self) -> Path:
        return self.output_dir / self.exp_name

    @property
    def task_type(self) -> str:
        return self.data.task_type

    def is_full_operator_checkpoint_eligible(self, epoch: int) -> bool:
        if not (self.discovery.enabled and self.discovery.mode == "auto"):
            return True
        return epoch >= (
            self.discovery.warmup_epochs + self.discovery.operator_anneal_epochs
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        config = cls(
            data=DataConfig.from_dict(raw.get("data", {})),
            model=ModelConfig.from_dict(raw.get("model", {})),
            optim=OptimConfig.from_dict(raw.get("optim", {})),
            discovery=SchemaDiscoveryConfig.from_dict(raw.get("discovery", {})),
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
        config._validate_cross_section_contracts()
        return config

    def _validate_cross_section_contracts(self) -> None:
        if self.early_stopping_patience <= 0:
            raise ValueError(
                "early_stopping_patience must be positive, "
                f"got {self.early_stopping_patience}."
            )
        if not (self.discovery.enabled and self.discovery.mode == "auto"):
            return
        if self.epochs <= (
            self.discovery.warmup_epochs + self.discovery.operator_anneal_epochs
        ):
            raise ValueError(
                "Auto-discovery training must continue beyond warmup_epochs + "
                "operator_anneal_epochs so an eligible full-operator checkpoint exists."
            )
        if self.data.task_type == "graph_classification":
            if self.data.pooling_type == "meanpool":
                raise ValueError(
                    "Auto AMPSS is incompatible with MeanPool: MeanPool ignores the edge "
                    "cochain, so task gradients cannot train upper-Hodge gates. Use sheafcov."
                )
            if self.data.pooling_type in {"ths_spectral", "ths_galerkin", "s2pool_h2"}:
                raise ValueError(
                    "Auto AMPSS does not support graph readouts that consume face-curl "
                    "descriptors directly. Use sheafcov, grampool, s2pool, or sheafpool."
                )
            if self.discovery.require_target_start or self.discovery.require_target_end:
                raise ValueError(
                    "require_target_start/end are node/link selector constraints and are "
                    "not defined for graph classification. Set both to false."
                )
        if not self.discovery.exclude_identity_metapaths:
            raise ValueError(
                "Auto face discovery requires exclude_identity_metapaths=true."
            )
