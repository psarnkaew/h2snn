from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from src.configs.config import TrainConfig
from src.models.h2snn_model import ModelDims

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


def _read_int_file(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


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

    # Edge labels come from PyG's `edge_attr` (one-hot of the TU edge label), which PyG keeps
    # aligned with its own `edge_index`. The raw `*_A.txt` order is NOT that order (PyG
    # coalesces and sorts edges), so labels must never be assigned by raw position.
    edge_labels_path = raw_dir / f"{tud_name}_edge_labels.txt"
    has_attr = [getattr(data, "edge_attr", None) is not None for data in data_list]
    if not any(has_attr):
        if edge_labels_path.exists():
            raise ValueError(
                f"{tud_name} has raw edge labels but the PyG graphs carry no edge_attr; "
                "edge labels cannot be aligned with the PyG edge order."
            )
        edge_type_parts = [
            torch.zeros(data.edge_index.size(1), dtype=torch.long) for data in data_list
        ]
        return node_type_parts, edge_type_parts, num_node_types, 1
    if not all(has_attr):
        raise ValueError(f"{tud_name}: some graphs carry edge_attr and some do not.")

    raw_parts = []
    for graph_index, data in enumerate(data_list):
        attr = data.edge_attr
        if attr.dim() != 2 or attr.size(0) != data.edge_index.size(1):
            raise ValueError(
                f"{tud_name} graph {graph_index}: edge_attr must be [E, C] for "
                f"{data.edge_index.size(1)} edges, got {tuple(attr.shape)}."
            )
        one_hot = bool(((attr == 0) | (attr == 1)).all()) and bool((attr.sum(dim=1) == 1).all())
        if attr.size(0) and not one_hot:
            raise ValueError(
                f"{tud_name} graph {graph_index}: edge_attr is not a one-hot edge label."
            )
        raw_parts.append(attr.argmax(dim=1) if attr.size(0) else attr.new_zeros(0).long())
    values = sorted({int(v) for part in raw_parts for v in part.tolist()})
    mapping = torch.full((max(values, default=0) + 1,), -1, dtype=torch.long)
    mapping[torch.tensor(values, dtype=torch.long)] = torch.arange(len(values))
    edge_type_parts = [mapping[part] for part in raw_parts]
    return node_type_parts, edge_type_parts, num_node_types, max(len(values), 1)


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
        num_nodes=None,
        # GraphClassificationTrainer enumerates two-hop typed paths per batch
        # (build_typed_path_face_groups defaults), so the only path length is 2.
        face_lengths=(2,),
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
    )
