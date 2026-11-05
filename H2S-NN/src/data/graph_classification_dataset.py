from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from src.configs.config import TrainConfig
from src.models.h2snn_model import ModelDims
from src.data.graph_faces import build_typed_path_face_groups
from src.data.schema_structures import DiscoveryMetadata, make_discovery_metadata
from src.data.stage1_selector import (
    CandidateStatistics,
    SelectedSchemaStatistics,
    SelectorReport,
    TrialSupport,
    derive_seed,
    make_selector_split,
    select_schema_features,
)
from src.data.selector_reporting import (
    build_coverage_diagnostic,
    enrich_report,
    stable_hash,
    tensor_content_hash,
)

_TUD_NAMES: dict[str, str] = {
    "mutag": "MUTAG",
    "proteins": "PROTEINS",
    "enzymes": "ENZYMES",
    "nci1": "NCI1",
}


@dataclass(slots=True)
class GraphClassificationFold:
    dims: ModelDims
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    fold_index: int
    num_folds: int
    run_index: int = 0
    num_runs: int = 1
    repetition_index: int = 0
    num_repetitions: int = 1
    split_seed: int = 42
    discovery_schema_names: tuple[str, ...] = tuple()
    discovery: DiscoveryMetadata | None = None
    selector_report: SelectorReport | None = None


def _read_int_file(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def _read_edge_pairs(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8") as handle:
        rows: list[tuple[int, int]] = []
        for line in handle:
            if not line.strip():
                continue
            left, right = line.replace(",", " ").split()[:2]
            rows.append((int(left), int(right)))
    return rows


def _compact_labels(values: list[int]) -> tuple[list[int], int]:
    mapping = {value: idx for idx, value in enumerate(sorted(set(values)))}
    return [mapping[value] for value in values], len(mapping)


def _load_tu_type_tensors(
    root: Path,
    tud_name: str,
    data_list: list[Data],
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    raw_dir = root / "tud" / tud_name / "raw"
    node_labels = _read_int_file(raw_dir / f"{tud_name}_node_labels.txt")
    graph_indicator = _read_int_file(raw_dir / f"{tud_name}_graph_indicator.txt")

    if node_labels is None or graph_indicator is None:
        node_type_parts = [
            torch.zeros(data.num_nodes, dtype=torch.long) for data in data_list
        ]
        num_node_types = 1
    else:
        compact_node_labels, num_node_types = _compact_labels(node_labels)
        node_type_parts = []
        offset = 0
        for data in data_list:
            next_offset = offset + int(data.num_nodes)
            node_type_parts.append(
                torch.tensor(compact_node_labels[offset:next_offset], dtype=torch.long)
            )
            offset = next_offset
        if offset != len(compact_node_labels):
            raise ValueError(
                f"TU node-label count mismatch for {tud_name}: "
                f"used {offset}, found {len(compact_node_labels)}."
            )

    edge_labels = _read_int_file(raw_dir / f"{tud_name}_edge_labels.txt")
    edge_pairs_path = raw_dir / f"{tud_name}_A.txt"
    if edge_labels is None or graph_indicator is None or not edge_pairs_path.exists():
        edge_type_parts = [
            torch.zeros(data.edge_index.size(1), dtype=torch.long) for data in data_list
        ]
        return node_type_parts, edge_type_parts, num_node_types, 1

    edge_pairs = _read_edge_pairs(edge_pairs_path)
    if len(edge_pairs) != len(edge_labels):
        raise ValueError(
            f"TU edge-label count mismatch for {tud_name}: "
            f"{len(edge_labels)} labels for {len(edge_pairs)} edges."
        )

    compact_edge_labels, num_relations = _compact_labels(edge_labels)
    labels_by_graph: list[list[int]] = [[] for _ in data_list]
    for (src, _), label in zip(edge_pairs, compact_edge_labels, strict=True):
        graph_id = graph_indicator[src - 1] - 1
        labels_by_graph[graph_id].append(label)

    edge_type_parts = []
    for graph_index, (data, labels) in enumerate(zip(data_list, labels_by_graph, strict=True)):
        num_edges = int(data.edge_index.size(1))
        if len(labels) != num_edges:
            raise ValueError(
                f"TU edge-label count mismatch for {tud_name} graph {graph_index}: "
                f"{len(labels)} labels for {num_edges} PyG edges."
            )
        edge_type_parts.append(torch.tensor(labels, dtype=torch.long))
    return node_type_parts, edge_type_parts, num_node_types, num_relations


def _annotate_tu_types(
    data: Data,
    node_type: torch.Tensor,
    edge_type: torch.Tensor,
    graph_id: int,
) -> Data:
    data.node_type = node_type
    data.edge_type = edge_type
    data.graph_id = torch.tensor([graph_id], dtype=torch.long)
    return data


def _annotate_single_type(data: Data) -> Data:
    """TU graphs are already single-node-type/single-edge-type — attach the trivial
    type/relation tensors H2SNN's heterogeneous machinery expects, rather than
    running the full hetero-flattening pipeline that solves a problem (multiple
    node/edge types) these graphs don't have."""
    data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)
    data.edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long)
    return data


def _generate_stratified_kfold_splits(
    labels: list[int],
    num_folds: int,
    seed: int,
) -> list[dict]:
    """Fresh split for datasets with no precomputed fold file (currently just MUTAG).

    Mirrors the JSON shape of the legacy baseline's folds/*.json files (a list of
    num_folds dicts, each with "test" indices and a "model_selection" list whose
    first entry has "train"/"validation" indices) so it's a drop-in-compatible,
    reusable artifact once cached, not a one-off.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    indices = np.arange(len(labels))
    labels_arr = np.array(labels)
    outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    folds: list[dict] = []
    for train_val_idx, test_idx in outer.split(indices, labels_arr):
        inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        train_val_labels = labels_arr[train_val_idx]
        inner_train_rel, inner_val_rel = next(inner.split(train_val_idx, train_val_labels))
        train_idx = train_val_idx[inner_train_rel]
        val_idx = train_val_idx[inner_val_rel]
        folds.append(
            {
                "test": test_idx.tolist(),
                "model_selection": [
                    {"train": train_idx.tolist(), "validation": val_idx.tolist()}
                ],
            }
        )
    return folds


def generate_repeated_kfold_splits(
    num_graphs: int,
    num_folds: int,
    num_repetitions: int,
    split_seed: int,
) -> list[list[dict[str, object]]]:
    """Build deterministic repeated KFold splits matching the TU baseline protocol.

    Each repetition uses an unstratified shuffled outer KFold. Ten percent of the
    outer training indices are then sampled for validation, leaving approximately
    81/9/10 percent of the complete dataset for train/validation/test when
    ``num_folds == 10``.
    """
    import numpy as np
    from sklearn.model_selection import KFold, train_test_split

    if num_folds < 2:
        raise ValueError(f"num_folds must be at least 2, got {num_folds}.")
    if num_repetitions <= 0:
        raise ValueError(
            f"num_repetitions must be positive, got {num_repetitions}."
        )
    if num_graphs < num_folds:
        raise ValueError(
            f"Cannot create {num_folds} folds from only {num_graphs} graphs."
        )
    if split_seed < 0:
        raise ValueError(f"split_seed must be non-negative, got {split_seed}.")

    indices = np.arange(num_graphs)
    repetitions: list[list[dict[str, object]]] = []
    for repetition_index in range(num_repetitions):
        repetition_seed = (int(split_seed) + repetition_index) % (2**32 - 1)
        outer = KFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=repetition_seed,
        )
        folds: list[dict[str, object]] = []
        for fold_index, (train_val_idx, test_idx) in enumerate(outer.split(indices)):
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.1,
                shuffle=True,
                random_state=((repetition_seed * num_folds) + fold_index) % (2**32 - 1),
            )
            folds.append(
                {
                    "test": test_idx.tolist(),
                    "model_selection": [
                        {
                            "train": train_idx.tolist(),
                            "validation": val_idx.tolist(),
                        }
                    ],
                }
            )
        repetitions.append(folds)
    return repetitions


def _load_or_generate_folds(
    dataset_name: str,
    labels: list[int],
    num_folds: int,
    seed: int,
) -> list[dict]:
    fold_path = Path("folds") / f"{_TUD_NAMES[dataset_name]}_splits.json"
    if fold_path.exists():
        with fold_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    folds = _generate_stratified_kfold_splits(labels, num_folds, seed)
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    with fold_path.open("w", encoding="utf-8") as handle:
        json.dump(folds, handle)
    return folds


def generate_stratified_holdout_split(
    labels: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Graph split ratios must sum to 1.0.")

    import numpy as np
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    labels_arr = np.array(labels)
    train_idx, holdout_idx, _, holdout_labels = train_test_split(
        indices,
        labels_arr,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels_arr,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test,
        random_state=seed + 1,
        shuffle=True,
        stratify=holdout_labels,
    )
    return {
        "train": sorted(int(value) for value in train_idx.tolist()),
        "validation": sorted(int(value) for value in val_idx.tolist()),
        "test": sorted(int(value) for value in test_idx.tolist()),
    }


def _build_graph_discovery_template(
    train_list: list[Data],
    config: TrainConfig,
) -> tuple[DiscoveryMetadata | None, tuple[str, ...], tuple[int, ...], SelectorReport | None]:
    """Returns discovery template, selected schemas, path lengths, and report.

    The third element is what the concat face encoder sizes its input projections from, so
    it must match the lengths batches actually materialise. Without discovery,
    `GraphClassificationTrainer` builds faces with `min_path_length=max_path_length=2`, so
    the only length is 2; the encoder raises on an unseen length rather than failing
    obscurely if that ever diverges.
    """
    if not (config.discovery.enabled and config.discovery.mode == "auto"):
        return None, tuple(), (2,), None
    if config.discovery.selector_strategy.lower() != "probe":
        raise ValueError(
            "Unsupported discovery.selector_strategy for graph classification: "
            f"{config.discovery.selector_strategy!r}."
        )

    schema_counts: dict[str, int] = defaultdict(int)
    schema_lengths: dict[str, int] = {}
    schema_graph_counts: list[dict[str, int]] = []
    labels: list[int] = []

    for graph_row, data in enumerate(train_list):
        label = int(data.y.view(-1)[0].item())
        labels.append(label)
        face_groups, _ = build_typed_path_face_groups(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            node_type=data.node_type,
            batch=torch.zeros(data.num_nodes, dtype=torch.long),
            graph_id=getattr(data, "graph_id", None),
            max_paths_per_anchor=config.discovery.max_instances_per_anchor,
            exclude_identity_paths=config.discovery.exclude_identity_metapaths,
            seed=config.discovery.materialization_seed,
            min_path_length=config.discovery.min_schema_length,
            max_path_length=config.discovery.max_schema_length,
            max_paths_per_schema=config.discovery.max_instances_per_schema,
            # Candidate statistics must be independent of the order in which
            # schema names happen to be traversed. Joint caps are applied
            # exactly to each trial subset below and during rematerialization.
            max_total_paths=None,
            max_total_incidences=None,
            allow_repeated_node_types=config.discovery.allow_repeated_node_types,
            max_template_identities=min(
                config.discovery.max_schema_identities,
                config.discovery.max_schema_candidates,
            ),
        )
        graph_counts: dict[str, int] = defaultdict(int)
        for group in face_groups:
            schema_counts[group.template_name] += group.num_faces
            graph_counts[group.template_name] += group.num_faces
            schema_lengths[group.template_name] = group.length
            if len(schema_counts) > config.discovery.max_schema_identities:
                raise ValueError(
                    "AMPSS feasible graph-schema count exceeds "
                    "max_schema_identities guard."
                )
            if len(schema_counts) > config.discovery.max_schema_candidates:
                raise ValueError(
                    "AMPSS viable graph-schema count exceeds "
                    "max_schema_candidates guard."
                )
        schema_graph_counts.append(graph_counts)

    if not schema_counts:
        return None, tuple(), tuple(), None

    candidate_names = sorted(schema_counts)
    max_count_by_name = {
        name: max((counts.get(name, 0) for counts in schema_graph_counts), default=0)
        for name in candidate_names
    }
    num_train_graphs = max(len(train_list), 1)
    def trial_support(indices) -> TrialSupport | None:
        ordered_indices = tuple(
            sorted(
                (int(index) for index in indices),
                key=lambda index: candidate_names[index],
            )
        )
        counts_by_index = {
            index: [0 for _ in train_list] for index in ordered_indices
        }
        total_incidences = 0
        for graph_row, graph_counts in enumerate(schema_graph_counts):
            remaining_faces = config.discovery.max_total_instances
            remaining_incidences = config.discovery.max_total_incidences
            for index in ordered_indices:
                name = candidate_names[index]
                incidence = schema_lengths[name] + 1
                quota = min(
                    graph_counts.get(name, 0),
                    config.discovery.max_instances_per_schema,
                )
                if remaining_faces is not None:
                    quota = min(quota, max(remaining_faces, 0))
                if remaining_incidences is not None:
                    quota = min(quota, max(remaining_incidences, 0) // incidence)
                counts_by_index[index][graph_row] = quota
                total_incidences += quota * incidence
                if remaining_faces is not None:
                    remaining_faces -= quota
                if remaining_incidences is not None:
                    remaining_incidences -= quota * incidence
        if any(sum(counts_by_index[index]) <= 0 for index in ordered_indices):
            return None

        blocks: list[torch.Tensor] = []
        covered_sets: list[frozenset[int]] = []
        quotas: list[int] = []
        support_tokens: list[object] = []
        for index in ordered_indices:
            name = candidate_names[index]
            max_count = max(max_count_by_name[name], 1)
            realized = counts_by_index[index]
            block = torch.zeros((len(train_list), 3), dtype=torch.float32)
            covered: set[int] = set()
            for row, count in enumerate(realized):
                if count > 0:
                    covered.add(row)
                block[row] = torch.tensor(
                    [
                        torch.log1p(torch.tensor(float(count))).item(),
                        1.0 if count > 0 else 0.0,
                        count / max_count,
                    ],
                    dtype=torch.float32,
                )
            blocks.append(block)
            covered_sets.append(frozenset(covered))
            quotas.append(sum(realized))
            support_tokens.append({"identity": name, "counts": realized})
        return TrialSupport(
            candidate_indices=ordered_indices,
            quotas=tuple(quotas),
            feature_blocks=tuple(blocks),
            coverage_sets=tuple(covered_sets),
            realized_incidence=float(total_incidences / num_train_graphs),
            support_hash=stable_hash(support_tokens),
        )

    standalone_supports = [trial_support((index,)) for index in range(len(candidate_names))]
    if any(support is None for support in standalone_supports):
        raise RuntimeError("A viable graph schema has zero standalone post-quota support.")
    feature_blocks = [
        support.feature_blocks[0]  # type: ignore[union-attr]
        for support in standalone_supports
    ]
    coverage_sets = [
        support.coverage_sets[0]  # type: ignore[union-attr]
        for support in standalone_supports
    ]
    incidence_masses = [
        support.realized_incidence  # type: ignore[union-attr]
        for support in standalone_supports
    ]

    label_tensor = torch.tensor(labels, dtype=torch.long)
    graph_unit_ids = torch.tensor(
        [
            int(getattr(data, "graph_id", torch.tensor([index])).view(-1)[0].item())
            for index, data in enumerate(train_list)
        ],
        dtype=torch.long,
    )
    split_seed = derive_seed(
        config.discovery.selector_seed,
        "ampss_graph_selector_split",
        config.data.dataset_name,
        config.data.fold_index,
        tensor_content_hash(graph_unit_ids),
    )
    split = make_selector_split(
        labels=label_tensor,
        task="single_label",
        holdout_ratio=config.discovery.selector_holdout_ratio,
        seed=split_seed,
        min_examples_per_label=config.discovery.selector_min_examples_per_label,
    )
    incidence_budget = config.discovery.max_total_incidences

    result = select_schema_features(
        candidate_identities=tuple(candidate_names),
        feature_blocks=tuple(feature_blocks),
        coverage_sets=tuple(coverage_sets),
        incidence_masses=tuple(incidence_masses),
        labels=label_tensor,
        split=split,
        task="single_label",
        max_schemas=config.discovery.selected_schema_k,
        incidence_budget=incidence_budget,
        stop_tolerance=config.discovery.selector_stop_tolerance,
        tie_tolerance=config.discovery.selector_tie_tolerance,
        probe_l2=config.discovery.selector_probe_l2,
        probe_max_iter=config.discovery.selector_probe_max_iter,
        split_seed=split_seed,
        sample_seed=config.discovery.materialization_seed,
        candidate_identity_count=len(schema_counts),
        coverage_denominator=len(train_list),
        sampling_attempts=0,
        sampling_successes=sum(schema_counts.values()),
        sampling_unique_paths=sum(schema_counts.values()),
        subset_support_fn=trial_support,
    )
    selected_index_set = set(result.selected_indices)
    selection_step_by_index = {
        step.candidate_index: step.step for step in result.report.steps
    }
    candidate_statistics = tuple(
        CandidateStatistics(
            candidate_index=index,
            identity=name,
            path_length=schema_lengths[name],
            viable=True,
            rejection_reason="",
            sampling_attempts=0,
            sampling_successes=schema_counts[name],
            sampling_unique_paths=schema_counts[name],
            pre_quota_instances=schema_counts[name],
            pre_quota_incidences=schema_counts[name] * (schema_lengths[name] + 1),
            probe_feature_dim=int(feature_blocks[index].size(1)),
            coverage_units=len(coverage_sets[index]),
            coverage_fraction=len(coverage_sets[index]) / max(len(train_list), 1),
            selected=index in selected_index_set,
            selection_step=selection_step_by_index.get(index),
        )
        for index, name in enumerate(candidate_names)
    )
    diagnostic_coverage_sets = list(coverage_sets)
    if result.final_support is not None:
        for index, covered in zip(
            result.final_support.candidate_indices,
            result.final_support.coverage_sets,
            strict=True,
        ):
            diagnostic_coverage_sets[index] = covered
    coverage_diagnostic = build_coverage_diagnostic(
        definition="graph_presence_in_training_fold_postquota",
        candidate_identities=candidate_names,
        coverage_sets=diagnostic_coverage_sets,
        selected_indices=result.selected_indices,
        num_units=len(train_list),
    )
    selected_names = tuple(candidate_names[index] for index in result.selected_indices)
    if not selected_names:
        report = enrich_report(
            result.report,
            selector_unit_hash=tensor_content_hash(graph_unit_ids),
            selector_support_hash="",
            candidate_statistics=candidate_statistics,
            selected_statistics=(),
            coverage_diagnostics=(coverage_diagnostic,),
        )
        return None, tuple(), tuple(), report
    # Re-materialize with only the selected templates. This is the support
    # policy used at runtime, so Stage-II coverage, incidence mass, and the
    # frozen manifest describe the operator that training actually sees.
    selected_index_by_name = {
        name: index for index, name in enumerate(selected_names)
    }
    realized_counts = [0 for _ in selected_names]
    realized_coverage: list[set[int]] = [set() for _ in selected_names]
    realized_tokens: list[list[object]] = [[] for _ in selected_names]
    for graph_row, data in enumerate(train_list):
        face_groups, _ = build_typed_path_face_groups(
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            node_type=data.node_type,
            batch=torch.zeros(data.num_nodes, dtype=torch.long),
            graph_id=getattr(data, "graph_id", None),
            max_paths_per_anchor=config.discovery.max_instances_per_anchor,
            exclude_identity_paths=config.discovery.exclude_identity_metapaths,
            seed=config.discovery.materialization_seed,
            min_path_length=config.discovery.min_schema_length,
            max_path_length=config.discovery.max_schema_length,
            selected_template_names=selected_names,
            max_paths_per_schema=config.discovery.max_instances_per_schema,
            max_total_paths=config.discovery.max_total_instances,
            max_total_incidences=config.discovery.max_total_incidences,
            allow_repeated_node_types=config.discovery.allow_repeated_node_types,
        )
        graph_id_value = getattr(data, "graph_id", None)
        graph_identity = (
            int(graph_id_value.view(-1)[0].item())
            if graph_id_value is not None
            else graph_row
        )
        for group in face_groups:
            schema_index = selected_index_by_name[group.template_name]
            realized_counts[schema_index] += group.num_faces
            realized_coverage[schema_index].add(graph_row)
            for nodes, edges in zip(
                group.face_node_index.detach().cpu().tolist(),
                group.face_edge_index.detach().cpu().tolist(),
                strict=True,
            ):
                realized_tokens[schema_index].append(
                    {
                        "graph_id": graph_identity,
                        "nodes": nodes,
                        "edges": edges,
                    }
                )

    if result.final_support is None:
        raise RuntimeError("Selected graph schemas are missing exact trial support.")
    expected_counts = {
        candidate_names[index]: quota
        for index, quota in zip(
            result.final_support.candidate_indices,
            result.final_support.quotas,
            strict=True,
        )
    }
    actual_counts = {
        name: realized_counts[index]
        for index, name in enumerate(selected_names)
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            "Graph AMPSS selector support and runtime materialization quotas disagree."
        )

    fold_mean_incidences = torch.tensor(
        [
            realized_counts[index] * (schema_lengths[name] + 1)
            / num_train_graphs
            for index, name in enumerate(selected_names)
        ],
        dtype=torch.float32,
    )
    discovery = make_discovery_metadata(
        num_schemas=len(selected_names),
        face_groups=tuple(),
        schema_ids=torch.empty(0, dtype=torch.long),
        schema_incidence_mass=fold_mean_incidences,
        coverage_sets=realized_coverage,
        num_coverage_units=len(train_list),
    )
    selected_statistics = tuple(
        SelectedSchemaStatistics(
            schema_index=schema_index,
            identity=name,
            materialized_faces=realized_counts[schema_index],
            path_incidences=(
                realized_counts[schema_index] * schema_lengths[name]
            ),
            closure_incidences=realized_counts[schema_index],
            total_incidences=(
                realized_counts[schema_index] * (schema_lengths[name] + 1)
            ),
            coverage_units=len(realized_coverage[schema_index]),
            coverage_fraction=(
                len(realized_coverage[schema_index]) / num_train_graphs
            ),
            face_set_hash=stable_hash(
                sorted(
                    realized_tokens[schema_index],
                    key=lambda value: json.dumps(value, sort_keys=True),
                )
            ),
        )
        for schema_index, name in enumerate(selected_names)
    )
    realized_coverage_diagnostic = build_coverage_diagnostic(
        definition="graph_selected_materialized_support",
        candidate_identities=selected_names,
        coverage_sets=tuple(frozenset(values) for values in realized_coverage),
        selected_indices=tuple(range(len(selected_names))),
        num_units=len(train_list),
    )
    report = enrich_report(
        result.report,
        selector_unit_hash=tensor_content_hash(graph_unit_ids),
        selector_support_hash=(
            result.final_support.support_hash
            if result.final_support is not None
            else ""
        ),
        candidate_statistics=candidate_statistics,
        selected_statistics=selected_statistics,
        coverage_diagnostics=(
            coverage_diagnostic,
            realized_coverage_diagnostic,
        ),
    )
    # Distinct path lengths of the *selected* schemas. Batches only ever materialise these,
    # so this is exactly the set the concat face encoder must build projections for. A
    # template name fixes its node-type sequence and therefore its length, so the per-graph
    # overwrites into `schema_lengths` above are consistent.
    selected_lengths = tuple(sorted({schema_lengths[name] for name in selected_names}))
    return discovery, selected_names, selected_lengths, report


def build_graph_classification_dataset(config: TrainConfig) -> GraphClassificationFold:
    dataset_name = config.data.dataset_name.lower()
    tud_name = _TUD_NAMES.get(dataset_name)
    if tud_name is None:
        raise ValueError(f"Unsupported graph-classification dataset '{config.data.dataset_name}'.")

    root = Path(config.data.root)
    dataset = TUDataset(root=str(root / "tud"), name=tud_name)
    raw_data_list = [dataset[i].clone() for i in range(len(dataset))]
    node_types, edge_types, num_node_types, num_relations = _load_tu_type_tensors(
        root=root,
        tud_name=tud_name,
        data_list=raw_data_list,
    )
    data_list = [
        _annotate_tu_types(data, node_type, edge_type, graph_id=index)
        for index, (data, node_type, edge_type) in enumerate(
            zip(raw_data_list, node_types, edge_types, strict=True)
        )
    ]

    labels = [int(item.y.item()) for item in data_list]
    num_classes = int(max(labels) + 1)

    repetition_index = config.data.repetition_index
    num_repetitions = config.data.num_repetitions
    split_seed = config.data.split_seed
    if config.data.split_protocol == "stratified_holdout":
        split = generate_stratified_holdout_split(
            labels=labels,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            seed=config.seed,
        )
        train_indices = split["train"]
        val_indices = split["validation"]
        test_indices = split["test"]
        fold_index = config.data.run_index
        num_folds = config.data.num_runs
        repetition_index = 0
        num_repetitions = 1
    elif config.data.split_protocol == "repeated_kfold":
        repetitions = generate_repeated_kfold_splits(
            num_graphs=len(data_list),
            num_folds=config.data.num_folds,
            num_repetitions=config.data.num_repetitions,
            split_seed=config.data.split_seed,
        )
        fold = repetitions[config.data.repetition_index][config.data.fold_index]
        model_selection = fold["model_selection"][0]
        train_indices = model_selection["train"]
        val_indices = model_selection["validation"]
        test_indices = fold["test"]
        fold_index = config.data.fold_index
        num_folds = config.data.num_folds
    else:
        folds = _load_or_generate_folds(
            dataset_name=dataset_name,
            labels=labels,
            num_folds=config.data.num_folds,
            seed=config.seed,
        )
        fold = folds[config.data.fold_index]
        model_selection = fold["model_selection"][0]
        train_indices = model_selection["train"]
        val_indices = model_selection["validation"]
        test_indices = fold["test"]
        fold_index = config.data.fold_index
        num_folds = config.data.num_folds

    train_list = [data_list[i] for i in train_indices]
    val_list = [data_list[i] for i in val_indices]
    test_list = [data_list[i] for i in test_indices]
    discovery, discovery_schema_names, discovery_face_lengths, selector_report = _build_graph_discovery_template(
        train_list=train_list,
        config=config,
    )

    batch_size = 32
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    in_channels = data_list[0].x.size(1) if data_list[0].x is not None else 0
    dims = ModelDims(
        in_channels=in_channels,
        num_node_types=num_node_types,
        num_relations=num_relations,
        num_classes=num_classes,
        discovery=discovery,
        num_nodes=None,
        face_lengths=discovery_face_lengths,
    )

    return GraphClassificationFold(
        dims=dims,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        fold_index=fold_index,
        num_folds=num_folds,
        run_index=config.data.run_index,
        num_runs=config.data.num_runs,
        repetition_index=repetition_index,
        num_repetitions=num_repetitions,
        split_seed=split_seed,
        discovery_schema_names=discovery_schema_names,
        discovery=discovery,
        selector_report=selector_report,
    )
