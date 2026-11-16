"""Training on the two large LP tasks for the baselines (card 1b E4; card 1c F7: decisions D2, G-8, G-10; spec v1.22 §6
LP; Part M3b B6).

* Data: H2GB's graph of the task's NC dataset, the task files (`split.npz`, `negatives.npz`), the **message graph**
  (held-out pairs removed in both orientations, `large_lp_minibatch.task_message_graph`).
* A step: `LP_POSITIVES_PER_STEP` training positives from a seeded order continued across epochs (`TrainingOrder`), one
  HGB uniform-tail negative per positive drawn per epoch with seed `run seed + epoch` (`hgb_train_negatives`); BCE over
  the links (plus a method's own auxiliary term); 40 epochs, patience 40, best validation AUROC epoch.
* Two samplers, both on the message graph:
  - `link`: PyG `LinkNeighborLoader` seeded by the step's links, one fan-out per hop (D2; H2GB's per-hop list, or the
    method's own fan-out); evaluation endpoints by node-level `NeighborLoader`;
  - `hgt`: H2GB's `hgt` sampler (PyG `HGTLoader`) with H2GB's per-type lists unchanged, seeded by the step's unique
    endpoints (decision G-8; and H2GB's own networks, whose head reads the seed rows); evaluation endpoints likewise.
  The endpoints' rows: `link` -> `edge_label_index` of the batch; `hgt` -> the seeds (the first `batch_size` rows of
  the node type, in the order given).
* Evaluation: every endpoint computed once per evaluation, then every link scored; validation = positives and the
  fixed validation negatives; test = HGB's 2-hop and random sets, evaluated when validation improves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        raise ValueError(f"per-node-type fan-out {sample_sizes} has no per-relation form; use the hgt sampler")
    return list(next(iter(lists)))


@dataclass
class Sampler:
    """`mode` "link" (fanout: per-hop list) or "hgt" (fanout: H2GB's per-type lists)."""

    mode: str
    fanout: Any

    def step(self, graph: Any, relation: tuple[str, str, str], node_type: str, links: torch.Tensor
             ) -> tuple[Any, torch.Tensor]:
        """(sampled subgraph of the step, endpoint row positions `[2, L]` in the encoder's node-type rows)."""
        if self.mode == "link":
            batch = next(iter(link_loader(graph, relation, links, self.fanout, batch_size=int(links.size(1)),
                                          shuffle=False)))
            return batch, batch[relation].edge_label_index
        nodes = torch.unique(links.reshape(-1))
        batch = next(iter(_hgt(graph, node_type, nodes, self.fanout, int(nodes.numel()))))
        if not torch.equal(batch[node_type].n_id[: nodes.numel()], nodes):
            raise RuntimeError("hgt sampler did not put the seeds first in the given order")
        return batch, torch.searchsorted(nodes, links)

    def endpoints(self, graph: Any, node_type: str, nodes: torch.Tensor, batch_size: int = EVAL_BATCH_TARGETS):
        """Batches whose first `batch_size` node-type rows are `nodes`, in order, each computed once."""
        if self.mode == "link":
            from torch_geometric.loader import NeighborLoader

            return NeighborLoader(graph, num_neighbors=self.fanout, input_nodes=(node_type, nodes),
                                  batch_size=batch_size, shuffle=False)
        return _hgt(graph, node_type, nodes, self.fanout, batch_size)


def _hgt(graph: Any, node_type: str, nodes: torch.Tensor, sample_sizes: dict[str, list[int]], batch_size: int):
    from torch_geometric.loader import HGTLoader

    return HGTLoader(graph, num_samples={k: list(v) for k, v in sample_sizes.items()}, input_nodes=(node_type, nodes),
                     batch_size=batch_size, shuffle=False)


@dataclass
class LargeLPModel:
    """What an adapter supplies: the trainable modules, `embed(batch, train)` -> rows of the node type (all rows for
    `link`, at least the seeds for `hgt`), the embedding width, and an optional auxiliary loss read after `embed`."""

    modules: torch.nn.Module
    embed: Callable[[Any, bool], torch.Tensor]
    embed_dim: int
    aux_loss: Callable[[], torch.Tensor] | None = None
    extra: dict = field(default_factory=dict)


def step_scores(model: LargeLPModel, head: torch.nn.Module, sampler: Sampler, graph: Any,
                relation: tuple[str, str, str], node_type: str, links: torch.Tensor, device: torch.device,
                train: bool = True) -> torch.Tensor:
    """Decoder scores of one step's links (the training forward; also what the tests run)."""
    batch, positions = sampler.step(graph, relation, node_type, links)
    rows = model.embed(batch, train)
    return head(rows, positions.to(device))


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


def fit_large_lp(bundle: Any, params: Any, *, seed: int, device: torch.device,
                 build: Callable[[LargeLPData], tuple[LargeLPModel, Sampler]]) -> dict[str, float]:
    """The large-LP loop for one adapter (`build` gives its model and sampler on the task's data)."""
    from src.baselines.budget import default_budget
    from src.baselines.heads import SharedLinkHead
    from src.baselines.metrics import prefixed
    from src.baselines.protocol import LINK_PREDICTION
    from src.data.hgb_lp_negatives import hgb_train_negatives
    from src.models.anchored_solve import check_finite_loss
    from src.trainers.minibatch_trainer import TrainingOrder
    from src.utils.misc import compute_link_prediction_metrics, seed_everything

    seed_everything(seed, device)
    d = load_large_lp(bundle.dataset, bundle.metadata["data_root"])
    model, sampler = build(d)
    model.modules.to(device)
    head = SharedLinkHead(str(params.get("decoder_type", "dot")), num_relations=1, dim=model.embed_dim).to(device)
    optimizer = torch.optim.AdamW(list(model.modules.parameters()) + list(head.parameters()),
                                  lr=float(params.get("lr", 5e-4)), weight_decay=float(params.get("weight_decay", 1e-4)))
    budget = default_budget(LINK_PREDICTION, bundle.dataset)
    num_dst = int(d.graph[d.node_type].num_nodes)

    def score_links(links: torch.Tensor) -> torch.Tensor:
        nodes = torch.unique(links.reshape(-1))
        table = torch.empty(int(nodes.numel()), model.embed_dim, device=device)
        position = 0
        with torch.no_grad():
            for batch in sampler.endpoints(d.graph, d.node_type, nodes):
                size = int(batch[d.node_type].batch_size)
                table[position:position + size] = model.embed(batch, False)[:size]
                position += size
            return head(table, torch.searchsorted(nodes, links).to(device))

    def metrics(pos: torch.Tensor, neg: torch.Tensor) -> dict[str, float]:
        return compute_link_prediction_metrics(score_links(pos).float(), score_links(neg).float())

    order = TrainingOrder(torch.arange(d.train_pos.size(1)), seed)
    steps = -(-int(d.train_pos.size(1)) // LP_POSITIVES_PER_STEP)
    best, best_epoch, stale, best_val, best_test = float("-inf"), -1, 0, {}, {}
    for epoch in range(budget.epochs):
        model.modules.train()
        head.train()
        negatives = hgb_train_negatives(d.train_pos, (0, num_dst), seed + epoch)
        for _ in range(steps):
            index = order.next(LP_POSITIVES_PER_STEP)
            links = torch.cat([d.train_pos[:, index], negatives[:, index]], dim=1)
            labels = torch.cat([torch.ones(index.numel()), torch.zeros(index.numel())]).to(device)
            optimizer.zero_grad()
            scores = step_scores(model, head, sampler, d.graph, d.relation, d.node_type, links, device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
            if model.aux_loss is not None:
                loss = loss + model.aux_loss()
            check_finite_loss(loss, epoch)
            loss.backward()
            optimizer.step()
        model.modules.eval()
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
    record["sampler"] = 0.0 if sampler.mode == "link" else 1.0
    return record


def h2gb_sampler(method: str, task: str, node_types: tuple[str, ...], force_hgt: bool = False) -> Sampler:
    """H2GB's YAML fan-out of `method` on the task's NC dataset: `link` with its per-hop list when one list serves every
    type and `force_hgt` is off (D2), else `hgt` with the per-type lists unchanged (G-8)."""
    from src.baselines.minibatch import h2gb_sample_sizes

    sizes = h2gb_sample_sizes(method, LARGE_LP_TASKS[task].nc_dataset, node_types)
    if not force_hgt:
        try:
            return Sampler("link", per_hop_fanout(sizes))
        except ValueError:
            pass
    return Sampler("hgt", sizes)


def homogeneous_embed(encoder: torch.nn.Module, node_types: tuple[str, ...], counts: dict[str, int], node_type: str,
                      encode: Callable[[torch.nn.Module, list[torch.Tensor], torch.Tensor, int], torch.Tensor],
                      device: torch.device) -> Callable[[Any, bool], torch.Tensor]:
    """`embed` for a homogeneous encoder on HGB's reduction of each batch: the node-type rows of its output."""
    from src.baselines.minibatch import batch_type_inputs, hgb_homogeneous_batch

    def embed(batch: Any, train: bool) -> torch.Tensor:
        inputs = [x.to(device) for x in batch_type_inputs(batch, node_types, counts)]
        _, edge_index, offsets = hgb_homogeneous_batch(batch, node_types)
        out = encode(encoder, inputs, edge_index.to(device), sum(x.size(0) for x in inputs))
        start, end = offsets[node_type]
        return out[start:end]
    return embed


__all__ = ["LargeLPData", "LargeLPModel", "Sampler", "fit_large_lp", "h2gb_sampler", "homogeneous_embed",
           "load_large_lp", "per_hop_fanout", "step_scores"]
