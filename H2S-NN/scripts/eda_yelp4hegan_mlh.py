"""Calculate the manuscript MLH score for the Yelp4HeGAN dataset.

The calculation follows the current manuscript protocol:
  * enumerate every directed two-hop relation schema that starts and ends
    at the labeled target node type;
  * threshold each adjacency product to distinct endpoint pairs;
  * retain diagonal endpoint pairs;
  * compute the different-label fraction for each induced graph; and
  * take the unweighted mean across valid schemas.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse


def load_triples(path: Path) -> np.ndarray:
    triples = np.loadtxt(path, dtype=np.int64)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"Expected three columns (source, destination, relation): {path}")
    return triples


def load_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError(f"Expected two columns (node, label): {path}")
    order = np.argsort(rows[:, 0])
    return rows[order, 0], rows[order, 1]


def calculate_mlh(triples: np.ndarray, target_ids: np.ndarray, labels: np.ndarray) -> dict:
    node_ids = np.unique(triples[:, :2])
    max_node_id = int(node_ids.max())
    target_to_local = {int(node_id): i for i, node_id in enumerate(target_ids)}
    target_set = set(target_to_local)
    num_targets = len(target_ids)

    edges_by_relation: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for source, destination, relation in triples:
        edges_by_relation[int(relation)].append((int(source), int(destination)))

    first_hops: dict[int, sparse.csr_matrix] = {}
    second_hops: dict[int, sparse.csr_matrix] = {}
    relation_statistics = []

    for relation, edges in sorted(edges_by_relation.items()):
        sources = np.fromiter((u for u, _ in edges), dtype=np.int64)
        destinations = np.fromiter((v for _, v in edges), dtype=np.int64)
        relation_statistics.append(
            {
                "relation": relation,
                "edges": len(edges),
                "unique_sources": int(np.unique(sources).size),
                "unique_destinations": int(np.unique(destinations).size),
                "source_id_range": [int(sources.min()), int(sources.max())],
                "destination_id_range": [int(destinations.min()), int(destinations.max())],
            }
        )

        first = [(target_to_local[u], v) for u, v in edges if u in target_set]
        if first:
            row, col = zip(*first)
            first_hops[relation] = sparse.csr_matrix(
                (np.ones(len(first), dtype=np.int64), (row, col)),
                shape=(num_targets, max_node_id + 1),
            )

        second = [(u, target_to_local[v]) for u, v in edges if v in target_set]
        if second:
            row, col = zip(*second)
            second_hops[relation] = sparse.csr_matrix(
                (np.ones(len(second), dtype=np.int64), (row, col)),
                shape=(max_node_id + 1, num_targets),
            )

    schema_results = []
    for first_relation, first_adjacency in sorted(first_hops.items()):
        for second_relation, second_adjacency in sorted(second_hops.items()):
            endpoint_graph = (first_adjacency @ second_adjacency).tocoo()
            endpoint_graph.eliminate_zeros()
            if endpoint_graph.nnz == 0:
                continue

            rows = endpoint_graph.row
            cols = endpoint_graph.col
            different = labels[rows] != labels[cols]
            schema_results.append(
                {
                    "relations": [first_relation, second_relation],
                    "endpoint_pairs": int(endpoint_graph.nnz),
                    "diagonal_pairs": int(np.count_nonzero(rows == cols)),
                    "different_label_pairs": int(np.count_nonzero(different)),
                    "mlh": float(np.mean(different)),
                }
            )

    if not schema_results:
        raise ValueError("No valid two-hop target-returning relation schemas were found")

    return {
        "protocol": {
            "path_length": 2,
            "schema_scope": "all directed target-returning relation schemas",
            "endpoint_pairs": "distinct (thresholded adjacency product)",
            "diagonal_pairs_retained": True,
            "per_schema_metric": "fraction of endpoint pairs with different labels",
            "aggregation": "unweighted mean",
            "interpretation": "near 0 = homophily; near 1 = heterophily",
        },
        "statistics": {
            "nodes": int(node_ids.size),
            "edges": int(triples.shape[0]),
            "relation_types": len(edges_by_relation),
            "labeled_target_nodes": num_targets,
            "classes": int(np.unique(labels).size),
        },
        "relation_statistics": relation_statistics,
        "schemas": schema_results,
        "mlh": float(np.mean([result["mlh"] for result in schema_results])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/yelp4HeGAN")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/yelp4HeGAN/mlh_ours.json")
    )
    args = parser.parse_args()

    triples = load_triples(args.data_dir / "yelp_triple.dat")
    target_ids, labels = load_labels(args.data_dir / "yelp_business_category.txt")
    result = calculate_mlh(triples, target_ids, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
