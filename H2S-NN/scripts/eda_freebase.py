"""Download and analyze the PyG HGB Freebase node-classification dataset.

The primary score follows the convention used in
``docs/pyg_dataset_eda_homophily_report.html``: enumerate valid directed
two-hop target-returning schemas, threshold each endpoint adjacency matrix,
restrict comparisons to nodes with released labels, and report the fraction
of endpoint pairs with different labels (MLH).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import HGBDataset

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_layout import REPO_DATA_ROOT, nc_root  # noqa: E402


TARGET = "book"
# The HGB Freebase benchmark defines eight book-label classes.  The supplied
# label files contain examples from only seven of them (labels 0 through 6).
DECLARED_CLASSES = {"book": 8}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _edge_matrix(edge_index: torch.Tensor, shape: tuple[int, int]) -> sp.csr_matrix:
    rows = edge_index[0].cpu().numpy()
    cols = edge_index[1].cpu().numpy()
    matrix = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=shape
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1
    return matrix


def _class_insensitive_homophily(
    labels: np.ndarray, rows: np.ndarray, cols: np.ndarray, same: np.ndarray
) -> float:
    classes = np.unique(labels)
    if classes.size <= 1:
        return 0.0
    proportions = np.asarray([(labels == value).mean() for value in classes])
    values = []
    for value in classes:
        destination = labels[cols] == value
        denominator = int(destination.sum())
        numerator = int((destination & same).sum())
        values.append(numerator / denominator if denominator else 0.0)
    return float(np.maximum(np.asarray(values) - proportions, 0.0).sum() / (classes.size - 1))


def _adjusted_homophily(
    labels: np.ndarray, rows: np.ndarray, cols: np.ndarray, same: np.ndarray
) -> float:
    del rows  # The H2GB convention uses destination incidence for directed pairs.
    edge_count = cols.size
    if edge_count == 0:
        return float("nan")
    classes = np.unique(labels)
    degree_by_class = np.zeros(classes.size, dtype=np.float64)
    for index, value in enumerate(classes):
        degree_by_class[index] = float((labels[cols] == value).sum())
    chance = float((degree_by_class**2).sum() / edge_count**2)
    h_edge = float(same.mean())
    if np.isclose(1.0 - chance, 0.0):
        return 0.0
    return float((h_edge - chance) / (1.0 - chance))


def _schema_metrics(data: Any, target: str) -> list[dict[str, Any]]:
    node_counts = {node_type: int(data[node_type].num_nodes) for node_type in data.node_types}
    labels_tensor = data[target].y
    labels = labels_tensor.cpu().numpy()
    known = labels >= 0
    rows_out: list[dict[str, Any]] = []

    schemas = []
    for first in data.edge_types:
        if first[0] != target:
            continue
        for second in data.edge_types:
            if second[0] == first[2] and second[2] == target:
                schemas.append((first, second))

    for first, second in schemas:
        first_matrix = _edge_matrix(
            data[first].edge_index,
            (node_counts[first[0]], node_counts[first[2]]),
        )
        second_matrix = _edge_matrix(
            data[second].edge_index,
            (node_counts[second[0]], node_counts[second[2]]),
        )
        endpoint = (first_matrix @ second_matrix).tocsr()
        endpoint.data[:] = 1
        endpoint.eliminate_zeros()
        rows, cols = endpoint.nonzero()
        label_mask = known[rows] & known[cols]
        rows = rows[label_mask]
        cols = cols[label_mask]
        same = labels[rows] == labels[cols]
        edge_count = int(cols.size)
        same_count = int(same.sum())
        different_count = edge_count - same_count
        h_edge = same_count / edge_count if edge_count else float("nan")
        mlh = different_count / edge_count if edge_count else float("nan")
        rows_out.append(
            {
                "schema": (
                    f"{first[0]}-{first[1]}-{first[2]}-"
                    f"{second[1]}-{second[2]}"
                ),
                "first_relation": first[1],
                "intermediate_type": first[2],
                "second_relation": second[1],
                "endpoint_pairs_all": int(endpoint.nnz),
                "endpoint_pairs_with_labels": edge_count,
                "same_label_pairs": same_count,
                "different_label_pairs": different_count,
                "edge_homophily": h_edge,
                "mlh": mlh,
                "class_insensitive_homophily": _class_insensitive_homophily(
                    labels[rows], rows, cols, same
                ) if edge_count else float("nan"),
                "adjusted_homophily": _adjusted_homophily(
                    labels[rows], rows, cols, same
                ) if edge_count else float("nan"),
            }
        )
    return rows_out


def analyze(root: Path, output_dir: Path) -> dict[str, Any]:
    dataset = HGBDataset(root=str(root), name="Freebase")
    data = dataset[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    declared_classes = DECLARED_CLASSES.get(TARGET, 0)

    node_rows = []
    for node_type in data.node_types:
        store = data[node_type]
        row: dict[str, Any] = {
            "node_type": node_type,
            "num_nodes": int(store.num_nodes),
            "feature_dim": int(store.x.size(1)) if hasattr(store, "x") else 0,
            "has_features": bool(hasattr(store, "x")),
            "labeled_nodes": 0,
            "train_nodes": 0,
            "test_nodes": 0,
            "num_classes": 0,
            "declared_num_classes": declared_classes if node_type == TARGET else 0,
        }
        if hasattr(store, "y"):
            y = store.y.cpu().numpy()
            known = y >= 0
            row["labeled_nodes"] = int(known.sum())
            row["num_classes"] = int(y[known].max() + 1) if known.any() else 0
            row["train_nodes"] = int(store.train_mask.sum()) if hasattr(store, "train_mask") else 0
            row["test_nodes"] = int(store.test_mask.sum()) if hasattr(store, "test_mask") else 0
        node_rows.append(row)

    edge_rows = []
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        source_count = int(data[edge_type[0]].num_nodes)
        degree = torch.bincount(edge_index[0], minlength=source_count).cpu().numpy()
        edge_rows.append(
            {
                "source_type": edge_type[0],
                "relation": edge_type[1],
                "destination_type": edge_type[2],
                "edge_count": int(edge_index.size(1)),
                "source_mean_degree": float(degree.mean()),
                "source_median_degree": float(np.median(degree)),
                "source_min_degree": int(degree.min()),
                "source_max_degree": int(degree.max()),
                "nonzero_source_nodes": int((degree > 0).sum()),
            }
        )

    target_store = data[TARGET]
    labels = target_store.y.cpu().numpy()
    known = labels >= 0
    label_rows = []
    for value in sorted(np.unique(labels[known]).tolist()):
        label_rows.append(
            {
                "label": int(value),
                "all_labeled_nodes": int((labels == value).sum()),
                "train_nodes": int(((labels == value) & target_store.train_mask.cpu().numpy()).sum()),
                "test_nodes": int(((labels == value) & target_store.test_mask.cpu().numpy()).sum()),
                "fraction_of_labeled_nodes": float((labels == value).sum() / known.sum()),
            }
        )

    schema_rows = _schema_metrics(data, TARGET)
    finite_mlh = [row["mlh"] for row in schema_rows if np.isfinite(row["mlh"])]
    summary = {
        "dataset": "HGB Freebase",
        "loader": "torch_geometric.datasets.HGBDataset",
        "root": str(root),
        "target_node_type": TARGET,
        "node_types": len(data.node_types),
        "edge_types": len(data.edge_types),
        "total_nodes": int(sum(row["num_nodes"] for row in node_rows)),
        "total_edges": int(sum(row["edge_count"] for row in edge_rows)),
        "target_nodes": int(target_store.num_nodes),
        "target_labeled_nodes": int(known.sum()),
        "target_train_nodes": int(target_store.train_mask.sum()),
        "target_test_nodes": int(target_store.test_mask.sum()),
        "target_num_classes": int(labels[known].max() + 1),
        "target_declared_num_classes": declared_classes,
        "valid_two_hop_schemas": len(schema_rows),
        "mlh": float(np.mean(finite_mlh)) if finite_mlh else None,
        "mlh_definition": "Mean fraction of differently labeled endpoint pairs over valid directed two-hop target-returning schemas; endpoint adjacency is thresholded and only released labels are used.",
    }

    _write_csv(output_dir / "node_type_stats.csv", node_rows)
    _write_csv(output_dir / "edge_type_stats.csv", edge_rows)
    _write_csv(output_dir / "target_label_stats.csv", label_rows)
    _write_csv(output_dir / "mlh_schema_metrics.csv", schema_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=nc_root(REPO_DATA_ROOT))
    parser.add_argument("--output-dir", type=Path,
                        default=nc_root(REPO_DATA_ROOT) / "freebase" / "eda")
    args = parser.parse_args()
    summary = analyze(args.root, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
