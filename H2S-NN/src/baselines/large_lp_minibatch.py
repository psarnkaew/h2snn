"""Link-level neighbour sampling for the baselines on the two large LP tasks (card baselines-h2gb C6; author decision
D2; spec v1.22 §6 LP; Part M3b B6 task files).

* The **message graph** of a task is H2GB's loaded graph without every fact whose node pair is a held-out (validation
  or test) pair, in both orientations and in every relation between the task's node type and itself
  (`src/data/large_lp.held_out_fact_mask`, the rule of Part M3b B6); training positives stay in it.
* A step's links are the seeds of PyG's `LinkNeighborLoader` on that message graph (the endpoints of the links seed the
  sampling; fan-out and depth are the method class's, open under C0 stop S1); negatives come from the task's files
  (`negatives.npz`) and from HGB's per-epoch training negatives, never drawn here.
* `assert_no_held_out_facts` checks a sampled subgraph: no edge (either orientation) between the endpoints of a
  held-out pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.large_lp import LARGE_LP_TASKS, held_out_fact_mask


def read_task_files(root: Path | str, task: str) -> dict[str, np.ndarray]:
    """The task's split (`train`/`valid`/`test` pairs `[P, 2]`, type-local, u < v) and negatives (`test_2hop`,
    `test_random`, `valid`), as `scripts/lp/build_large_lp.py:142-144` wrote them."""
    from src.data.data_layout import large_lp_dir

    folder = large_lp_dir(Path(root), task)
    out: dict[str, np.ndarray] = {}
    with np.load(folder / "split.npz") as split:
        out.update({f"split_{k}": split[k] for k in split.files})
    with np.load(folder / "negatives.npz") as negatives:
        out.update({f"negatives_{k}": negatives[k] for k in negatives.files})
    return out


def held_out_pairs(files: dict[str, np.ndarray]) -> torch.Tensor:
    """Validation and test pairs as `[2, H]` type-local ids."""
    return torch.from_numpy(np.concatenate([files["split_valid"], files["split_test"]]).T.astype(np.int64))


def message_graph(data: Any, node_type: str, held_out: torch.Tensor) -> Any:
    """`data` without every held-out fact (both orientations) in every relation from `node_type` to itself."""
    from torch_geometric.data import HeteroData

    graph = HeteroData()
    for t in data.node_types:
        for key, value in data[t].items():
            graph[t][key] = value
    for edge_type in data.edge_types:
        index = data[edge_type].edge_index
        if edge_type[0] == node_type and edge_type[2] == node_type and held_out.numel():
            index = index[:, ~held_out_fact_mask(index, held_out)]
        graph[edge_type].edge_index = index
    return graph


def task_message_graph(data: Any, task: str, files: dict[str, np.ndarray]) -> Any:
    return message_graph(data, LARGE_LP_TASKS[task].node_type, held_out_pairs(files))


def link_loader(graph: Any, relation: tuple[str, str, str], links: torch.Tensor, num_neighbors: Any,
                batch_size: int, shuffle: bool) -> Any:
    """PyG `LinkNeighborLoader` on `graph` seeded by `links` (`[2, L]`, type-local) of `relation`; no negative
    sampling (negatives are given)."""
    from torch_geometric.loader import LinkNeighborLoader

    return LinkNeighborLoader(graph, num_neighbors=num_neighbors, edge_label_index=(relation, links),
                              batch_size=batch_size, shuffle=shuffle)


def held_out_facts_in(batch: Any, node_type: str, held_out: torch.Tensor) -> int:
    """Edges of a sampled subgraph (relations from `node_type` to itself) whose global pair is held out."""
    count = 0
    for edge_type in batch.edge_types:
        if edge_type[0] != node_type or edge_type[2] != node_type:
            continue
        ids = batch[node_type].n_id
        index = batch[edge_type].edge_index
        if index.numel():
            count += int(held_out_fact_mask(torch.stack([ids[index[0]], ids[index[1]]]), held_out).sum())
    return count


def assert_no_held_out_facts(batch: Any, node_type: str, held_out: torch.Tensor) -> None:
    found = held_out_facts_in(batch, node_type, held_out)
    if found:
        raise AssertionError(f"{found} held-out fact(s) present in a sampled subgraph")


__all__ = ["assert_no_held_out_facts", "held_out_facts_in", "held_out_pairs", "link_loader", "message_graph",
           "read_task_files", "task_message_graph"]
