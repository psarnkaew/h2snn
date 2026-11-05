"""Streaming EDA for the Network Repository actor export.

The local actor export is too large for a convenient pandas workflow.  This
script keeps only node labels, degree arrays, and small counters in memory,
then writes a compact JSON/CSV/Markdown report and a few diagnostic figures.

Usage (from the project root):
    C:\\Users\\Acer\\anaconda3\\envs\\h2snn\\python.exe \\
        scripts\\eda_actor_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def degree_summary(degrees: np.ndarray) -> dict[str, float | int]:
    nonzero = degrees[degrees > 0]
    return {
        "nodes": int(degrees.size),
        "nonzero_nodes": int(nonzero.size),
        "zero_nodes": int(degrees.size - nonzero.size),
        "mean": float(degrees.mean()) if degrees.size else 0.0,
        "median": percentile(degrees, 50),
        "p90": percentile(degrees, 90),
        "p95": percentile(degrees, 95),
        "p99": percentile(degrees, 99),
        "max": int(degrees.max()) if degrees.size else 0,
    }


def load_labels(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[int] = []
    ids: list[int] = []
    malformed = 0
    for line_no, raw in enumerate(path.open("r", encoding="utf-8"), start=1):
        fields = raw.strip().split(",")
        if len(fields) != 2:
            malformed += 1
            continue
        node_id, label = (int(fields[0]), int(fields[1]))
        ids.append(node_id)
        values.append(label)

    if not ids:
        raise ValueError(f"No labels found in {path}")
    max_id = max(ids)
    labels = np.zeros(max_id + 1, dtype=np.int64)
    labels[np.asarray(ids, dtype=np.int64)] = np.asarray(values, dtype=np.int64)
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    id_set = set(ids)
    missing_ids = [node_id for node_id in range(1, max_id + 1) if node_id not in id_set]
    return labels, {
        "records": len(ids),
        "id_min": min(ids),
        "id_max": max_id,
        "unique_ids": len(id_set),
        "missing_ids": len(missing_ids),
        "malformed_records": malformed,
        "label_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "labels_by_id_range": {
            "first_half": sorted(set(labels[1 : max_id // 2 + 1].tolist())),
            "second_half": sorted(set(labels[max_id // 2 + 1 :].tolist())),
        },
    }


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/actor"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/eda/actor"))
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    labels, label_meta = load_labels(data_dir / "actor.types")
    n_nodes = labels.size - 1
    out_degree = np.zeros(n_nodes + 1, dtype=np.int64)
    in_degree = np.zeros(n_nodes + 1, dtype=np.int64)
    source_seen = np.zeros(n_nodes + 1, dtype=np.bool_)
    target_seen = np.zeros(n_nodes + 1, dtype=np.bool_)

    edge_count = 0
    malformed_edges = 0
    invalid_edges = 0
    self_loops = 0
    consecutive_duplicate_edges = 0
    sorted_order_violations = 0
    min_endpoint = math.inf
    max_endpoint = 0
    source_min = math.inf
    source_max = 0
    target_min = math.inf
    target_max = 0
    mixing = Counter()
    self_loops_by_label = Counter()
    previous_key: tuple[int, int] | None = None

    with (data_dir / "actor.edges").open("rb") as handle:
        for raw in handle:
            fields = raw.strip().split(b",")
            if len(fields) != 2:
                malformed_edges += 1
                continue
            source, target = int(fields[0]), int(fields[1])
            edge_count += 1
            min_endpoint = min(min_endpoint, source, target)
            max_endpoint = max(max_endpoint, source, target)
            source_min = min(source_min, source)
            source_max = max(source_max, source)
            target_min = min(target_min, target)
            target_max = max(target_max, target)

            if not (1 <= source <= n_nodes and 1 <= target <= n_nodes):
                invalid_edges += 1
                continue
            out_degree[source] += 1
            in_degree[target] += 1
            source_seen[source] = True
            target_seen[target] = True
            source_label = int(labels[source])
            target_label = int(labels[target])
            mixing[(source_label, target_label)] += 1
            if source == target:
                self_loops += 1
                self_loops_by_label[source_label] += 1

            current_key = (target, source)
            if previous_key is not None:
                if current_key < previous_key:
                    sorted_order_violations += 1
                if current_key == previous_key:
                    consecutive_duplicate_edges += 1
            previous_key = current_key

    source_labels, source_label_counts = np.unique(labels[source_seen], return_counts=True)
    target_labels, target_label_counts = np.unique(labels[target_seen], return_counts=True)

    label_values = sorted(int(x) for x in np.unique(labels[1:]))
    mixing_rows = []
    for source_label in label_values:
        row_total = sum(mixing[(source_label, target_label)] for target_label in label_values)
        for target_label in label_values:
            count = mixing[(source_label, target_label)]
            mixing_rows.append(
                [source_label, target_label, count, count / row_total if row_total else 0.0]
            )

    summary: dict[str, Any] = {
        "dataset": "actor",
        "input_files": {
            "types": str((data_dir / "actor.types").as_posix()),
            "edges": str((data_dir / "actor.edges").as_posix()),
        },
        "file_bytes": {
            "types": (data_dir / "actor.types").stat().st_size,
            "edges": (data_dir / "actor.edges").stat().st_size,
        },
        "node_count": n_nodes,
        "label_metadata": label_meta,
        "edge_count": edge_count,
        "edge_record_quality": {
            "malformed_records": malformed_edges,
            "invalid_endpoint_records": invalid_edges,
            "consecutive_duplicate_records": consecutive_duplicate_edges,
            "sorted_by_target_then_source": sorted_order_violations == 0,
            "sorted_order_violations": sorted_order_violations,
            "self_loops": self_loops,
            "self_loop_fraction": self_loops / edge_count if edge_count else 0.0,
        },
        "endpoint_ranges": {
            "all_min": int(min_endpoint),
            "all_max": int(max_endpoint),
            "source_min": int(source_min),
            "source_max": int(source_max),
            "target_min": int(target_min),
            "target_max": int(target_max),
        },
        "node_coverage": {
            "source_nodes": int(source_seen.sum()),
            "target_nodes": int(target_seen.sum()),
            "source_coverage": float(source_seen[1:].mean()),
            "target_coverage": float(target_seen[1:].mean()),
            "source_label_counts": {
                str(int(k)): int(v) for k, v in zip(source_labels, source_label_counts)
            },
            "target_label_counts": {
                str(int(k)): int(v) for k, v in zip(target_labels, target_label_counts)
            },
        },
        "degree": {
            "out_degree": degree_summary(out_degree[1:]),
            "in_degree": degree_summary(in_degree[1:]),
            "out_degree_by_label": {
                str(label): degree_summary(out_degree[1:][labels[1:] == label])
                for label in label_values
            },
            "in_degree_by_label": {
                str(label): degree_summary(in_degree[1:][labels[1:] == label])
                for label in label_values
            },
        },
        "label_mixing": mixing_rows,
        "self_loops_by_label": {str(k): int(v) for k, v in sorted(self_loops_by_label.items())},
        "analysis_seconds": round(time.time() - started, 3),
    }

    (output_dir / "actor_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(
        output_dir / "label_mixing.csv",
        ["source_label", "target_label", "edge_records", "row_fraction"],
        mixing_rows,
    )

    ranking_rows: list[list[Any]] = []
    for degree_name, degree_values in [("in_degree", in_degree), ("out_degree", out_degree)]:
        top_indices = np.argsort(degree_values[1:])[::-1][:100] + 1
        for rank, node_id in enumerate(top_indices, start=1):
            ranking_rows.append(
                [degree_name, rank, int(node_id), int(labels[node_id]), int(degree_values[node_id])]
            )
    write_csv(
        output_dir / "top_degree_nodes.csv",
        ["degree_type", "rank", "node_id", "label", "degree"],
        ranking_rows,
    )

    # Figure 1: node-label/type composition.
    counts = [int(np.sum(labels[1:] == label)) for label in label_values]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(x) for x in label_values], counts, color=["#4472C4", "#ED7D31"])
    ax.set_title("Actor export: node label/type counts")
    ax.set_xlabel("Value in actor.types")
    ax.set_ylabel("Nodes")
    ax.bar_label(bars, labels=[f"{x:,}" for x in counts], padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "label_counts.png", dpi=180)
    plt.close(fig)

    # Figure 2: directed-record degree distributions.  Zero-degree nodes are
    # shown separately because source and target roles are asymmetric here.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, degree_values, title, color in [
        (axes[0], out_degree[1:], "Outgoing records", "#4472C4"),
        (axes[1], in_degree[1:], "Incoming records", "#70AD47"),
    ]:
        positive = degree_values[degree_values > 0]
        unique, freq = np.unique(positive, return_counts=True)
        ax.loglog(unique, freq, marker=".", linestyle="none", color=color, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("Degree (log scale)")
        ax.set_ylabel("Number of nodes (log scale)")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("Actor export: degree distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "degree_distributions.png", dpi=180)
    plt.close(fig)

    # Figure 3: endpoint label mixing.  Counts are deliberately used instead
    # of a normalized heatmap because one source label may be absent.
    matrix = np.array(
        [[mixing[(s, t)] for t in label_values] for s in label_values], dtype=np.int64
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(label_values)), labels=[str(x) for x in label_values])
    ax.set_yticks(range(len(label_values)), labels=[str(x) for x in label_values])
    ax.set_xlabel("Target label/type")
    ax.set_ylabel("Source label/type")
    ax.set_title("Edge-record label/type mixing")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Edge records")
    fig.tight_layout()
    fig.savefig(output_dir / "label_mixing.png", dpi=180)
    plt.close(fig)

    # Figure 4: source/target role coverage.
    role_counts = [int(source_seen[1:].sum()), int(target_seen[1:].sum())]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Source endpoint", "Target endpoint"], role_counts, color=["#5B9BD5", "#A5A5A5"])
    ax.set_title("Node participation by endpoint role")
    ax.set_ylabel("Distinct nodes")
    ax.set_ylim(0, n_nodes * 1.08)
    ax.bar_label(bars, labels=[f"{x:,}" for x in role_counts], padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_coverage.png", dpi=180)
    plt.close(fig)

    label_text = ", ".join(f"{label}={count:,}" for label, count in zip(label_values, counts))
    same_label = sum(mixing[(label, label)] for label in label_values)
    cross_label = edge_count - same_label
    order_statement = (
        "The records are not strictly sorted by `(target, source)`; the scan "
        f"observed {sorted_order_violations:,} order violations."
        if sorted_order_violations
        else "The records are sorted by `(target, source)`."
    )
    active_out = out_degree[1:][out_degree[1:] > 0]
    report = f"""# Actor dataset EDA

## Scope

This report analyzes the local files `data/actor/actor.types` and
`data/actor/actor.edges` as edge records. The analysis is streaming and does
not assume that the edge file is a simple undirected graph.

## Summary

- Nodes: **{n_nodes:,}**
- Values in `actor.types`: **{label_text}**
- Edge records: **{edge_count:,}**
- Self-loop records: **{self_loops:,}** ({self_loops / edge_count:.2%})
- Invalid endpoints: **{invalid_edges:,}**
- Consecutive duplicate records: **{consecutive_duplicate_edges:,}**
- Distinct source nodes: **{int(source_seen.sum()):,}** ({source_seen[1:].mean():.2%} of nodes)
- Distinct target nodes: **{int(target_seen.sum()):,}** ({target_seen[1:].mean():.2%} of nodes)
- Same-value endpoint records: **{same_label:,}** ({same_label / edge_count:.2%})
- Cross-value endpoint records: **{cross_label:,}** ({cross_label / edge_count:.2%})

## Representation findings

{order_statement} All valid source IDs fall in
`{int(source_min):,}..{int(source_max):,}` while target IDs span
`{int(target_min):,}..{int(target_max):,}`. The source/target roles are therefore
asymmetric in this export. The values in `actor.types` behave like node types
or endpoint categories in this analysis; they should not automatically be
interpreted as supervised class labels.

## Degree findings

- Out-degree mean/median/p99/max over all nodes: **{out_degree[1:].mean():.2f} / {np.median(out_degree[1:]):.0f} / {np.percentile(out_degree[1:], 99):.0f} / {out_degree.max():,}**
- Out-degree mean/median/p99/max over active source nodes: **{active_out.mean():.2f} / {np.median(active_out):.0f} / {np.percentile(active_out, 99):.0f} / {active_out.max():,}**
- In-degree mean/median/p99/max: **{in_degree[1:].mean():.2f} / {np.median(in_degree[1:]):.0f} / {np.percentile(in_degree[1:], 99):.0f} / {in_degree.max():,}**

## Files

- [actor_summary.json](actor_summary.json)
- [label_mixing.csv](label_mixing.csv)
- [top_degree_nodes.csv](top_degree_nodes.csv)
- [label_counts.png](label_counts.png)
- [degree_distributions.png](degree_distributions.png)
- [label_mixing.png](label_mixing.png)
- [endpoint_coverage.png](endpoint_coverage.png)
"""
    (output_dir / "actor_eda_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
