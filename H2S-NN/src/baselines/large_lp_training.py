"""Training on the two large LP tasks for the baselines (part 2 E4; author decision D2; spec v1.22 §6 LP; Part M3b B6).

* Data: H2GB's graph of the task's NC dataset, the task files (`split.npz`, `negatives.npz`), the **message graph**
  (held-out pairs removed in both orientations, `large_lp_minibatch.task_message_graph`).
* A step: `LP_POSITIVES_PER_STEP` training positives from a seeded order continued across epochs (`TrainingOrder`), one
  HGB uniform-tail negative per positive drawn per epoch with seed `run seed + epoch` (`hgb_train_negatives`, as the
  large-LP loop of H2S-NN); the step's links seed PyG's `LinkNeighborLoader` on the message graph (`step_subgraph`);
  the encoder runs on HGB's homogeneous reduction of that subgraph; BCE over the links; 40 epochs, patience 40 (the LP
  budget), best validation AUROC epoch.
* Evaluation: every endpoint of the split's links is computed **once** per evaluation from node-level neighbour
  sampling on the message graph (`endpoint_loader`), then every link is scored; validation = the task's positives and
  fixed validation negatives; test = HGB's two sets (2-hop and random negatives), evaluated when validation improves.
* Fan-out per hop: the H2GB YAML list of the method on the task's NC dataset, one value per hop for every relation;
  a per-node-type list (H-Pokec) has no per-relation form here (part 2 E4 stop item).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.baselines.large_lp_minibatch import link_loader, read_task_files, task_message_graph
from src.configs.config import EVAL_BATCH_TARGETS, LP_POSITIVES_PER_STEP
from src.data.large_lp import LARGE_LP_TASKS


def per_hop_fanout(sample_sizes: dict[str, list[int]]) -> list[int]:
    """One fan-out per hop for every relation, from H2GB's per-type lists when they are all the same list."""
    lists = {tuple(v) for v in sample_sizes.values()}
    if len(lists) != 1:
        raise ValueError(f"per-node-type fan-out {sample_sizes} has no per-relation form (part 2 E4 stop item)")
    return list(next(iter(lists)))


def step_subgraph(graph: Any, relation: tuple[str, str, str], links: torch.Tensor, num_neighbors: list[int]) -> Any:
    """The sampled subgraph of one step: the links (`[2, L]`, type-local) as seeds of `LinkNeighborLoader` on the
    message graph `graph`; `batch[relation].edge_label_index` holds the links in batch-local ids."""
    return next(iter(link_loader(graph, relation, links, num_neighbors, batch_size=int(links.size(1)), shuffle=False)))


def endpoint_loader(graph: Any, node_type: str, nodes: torch.Tensor, num_neighbors: list[int],
                    batch_size: int = EVAL_BATCH_TARGETS) -> Any:
    """Node-level neighbour sampling on the message graph for evaluation endpoints (each computed once)."""
    from torch_geometric.loader import NeighborLoader

    return NeighborLoader(graph, num_neighbors=num_neighbors, input_nodes=(node_type, nodes), batch_size=batch_size,
                          shuffle=False)


@dataclass
class LargeLPData:
    task: str
    graph: Any
    node_type: str
    relation: tuple[str, str, str]
    train_pos: torch.Tensor   # [2, P] type-local
    val_pos: torch.Tensor
    val_neg: torch.Tensor
    test_pos: torch.Tensor
    test_2hop: torch.Tensor
    test_random: torch.Tensor


def load_large_lp(task: str, root: Path | str) -> LargeLPData:
    from src.data.h2gb_nc import load_h2gb_nc

    spec = LARGE_LP_TASKS[task]
    data = load_h2gb_nc(spec.nc_dataset, root)
    files = read_task_files(root, task)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T)).long()  # noqa: E731  - [L, 2] -> [2, L]
    return LargeLPData(task, task_message_graph(data, task, files), spec.node_type, spec.relation,
                       t(files["split_train"]), t(files["split_valid"]), t(files["negatives_valid"]),
                       t(files["split_test"]), t(files["negatives_test_2hop"]), t(files["negatives_test_random"]))


def fit_large_lp_homogeneous(bundle: Any, params: Any, *, fanout_method: str, seed: int, device: torch.device,
                             build_encoder: Callable[[list[int]], torch.nn.Module],
                             encode: Callable[[torch.nn.Module, list[torch.Tensor], torch.Tensor], torch.Tensor],
                             embed_dim: int) -> dict[str, float]:
    """A homogeneous LP encoder (the local-LP one of its adapter) on the large task with the shared decoder."""
    from src.baselines.budget import default_budget
    from src.baselines.heads import SharedLinkHead
    from src.baselines.metrics import prefixed
    from src.baselines.minibatch import batch_type_inputs, h2gb_input_dims, h2gb_sample_sizes, hgb_homogeneous_batch
    from src.baselines.protocol import LINK_PREDICTION
    from src.data.hgb_lp_negatives import hgb_train_negatives
    from src.models.anchored_solve import check_finite_loss
    from src.trainers.minibatch_trainer import TrainingOrder
    from src.utils.misc import compute_link_prediction_metrics, seed_everything

    seed_everything(seed, device)
    d = load_large_lp(bundle.dataset, bundle.metadata["data_root"])
    node_types, counts, dims = h2gb_input_dims(d.graph)
    fanout = per_hop_fanout(h2gb_sample_sizes(fanout_method, LARGE_LP_TASKS[bundle.dataset].nc_dataset, node_types))
    encoder = build_encoder(dims).to(device)
    head = SharedLinkHead(str(params.get("decoder_type", "dot")), num_relations=1, dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                                  lr=float(params.get("lr", 5e-4)), weight_decay=float(params.get("weight_decay", 1e-4)))
    budget = default_budget(LINK_PREDICTION, bundle.dataset)
    num_dst = int(d.graph[d.node_type].num_nodes)

    def embed(batch: Any) -> tuple[torch.Tensor, int]:
        inputs = [x.to(device) for x in batch_type_inputs(batch, node_types, counts)]
        _, edge_index, offsets = hgb_homogeneous_batch(batch, node_types)
        return encode(encoder, inputs, edge_index.to(device)), offsets[d.node_type][0]

    def score_step(links: torch.Tensor) -> torch.Tensor:
        batch = step_subgraph(d.graph, d.relation, links, fanout)
        out, start = embed(batch)
        return head(out, (batch[d.relation].edge_label_index + start).to(device))

    def score_links(links: torch.Tensor) -> torch.Tensor:
        nodes = torch.unique(links.reshape(-1))
        table = torch.empty(int(nodes.numel()), embed_dim, device=device)
        position = 0
        with torch.no_grad():
            for batch in endpoint_loader(d.graph, d.node_type, nodes, fanout):
                out, start = embed(batch)
                size = int(batch[d.node_type].batch_size)
                table[position:position + size] = out[start:start + size]
                position += size
            return head(table, torch.searchsorted(nodes, links).to(device))

    def metrics(pos: torch.Tensor, neg: torch.Tensor) -> dict[str, float]:
        p, n = score_links(pos), score_links(neg)
        return compute_link_prediction_metrics(p.float(), n.float())

    order = TrainingOrder(torch.arange(d.train_pos.size(1)), seed)
    steps = -(-int(d.train_pos.size(1)) // LP_POSITIVES_PER_STEP)
    best, best_epoch, stale, best_val, best_test = float("-inf"), -1, 0, {}, {}
    for epoch in range(budget.epochs):
        encoder.train()
        head.train()
        negatives = hgb_train_negatives(d.train_pos, (0, num_dst), seed + epoch)
        for _ in range(steps):
            index = order.next(LP_POSITIVES_PER_STEP)
            links = torch.cat([d.train_pos[:, index], negatives[:, index]], dim=1)
            labels = torch.cat([torch.ones(index.numel()), torch.zeros(index.numel())]).to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(score_step(links), labels)
            check_finite_loss(loss, epoch)
            loss.backward()
            optimizer.step()
        encoder.eval()
        head.eval()
        val = metrics(d.val_pos, d.val_neg)
        if val["auroc"] > best:
            best, best_epoch, stale, best_val = val["auroc"], epoch, 0, val
            best_test = {**prefixed(metrics(d.test_pos, d.test_2hop), "2hop"),
                         **prefixed(metrics(d.test_pos, d.test_random), "random")}
        else:
            stale += 1
        if stale >= budget.patience:
            break
    record = {**prefixed(best_val, "val"), **prefixed(best_test, "test")}
    record["best_epoch"] = float(best_epoch)
    record["sampler_depth"] = float(len(fanout))
    return record


__all__ = ["LargeLPData", "endpoint_loader", "fit_large_lp_homogeneous", "load_large_lp", "per_hop_fanout",
           "step_subgraph"]
