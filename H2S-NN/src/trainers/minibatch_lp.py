"""Mini-batch link prediction on the two large tasks (ogbn-mag P-P, H-Pokec U-U; data pending)
(spec v1.21 §6 "Link prediction"; `docs/minibatch_design_v3B.md` §8; decision MB4; card Part MB,
MB5).

* Held-out (validation and test) facts are removed from the message graph in **both orientations**
  before any draw (`remove_held_out_facts`); training positives stay in it (as full batch and HGB).
* Each endpoint of a step's positive and negative links is a target with its own scope (anchors of
  the draw = those endpoints only); a node in several links of the step is computed once; the
  decoder scores `(x*_u, x*_v)` exactly as in full batch (`H2SNN.link_readout` on `[x*, h0]`).
* A step: `LP_POSITIVES_PER_STEP` positives with 1:1 training negatives, BCE; 40 epochs, patience 40
  (the HGB LP budget, `protocol_budget`).
* Evaluation: the scopes of all validation/test endpoints are computed once (one draw per run from
  the base seed) and every link is scored.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from src.configs.config import EVAL_BATCH_TARGETS
from src.data.scopes import build_scope_batch
from src.data.structures import PreparedH2SData
from src.data.wisp_sampler import AnchorCells, build_wisp_anchor_cells

#: Positive links per optimizer step on the large LP tasks (HGB's 8,192; design §6).
LP_POSITIVES_PER_STEP = 8192


def remove_held_out_facts(edge_index: torch.Tensor, edge_type: torch.Tensor,
                          held_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The message graph without every fact whose pair is a held-out pair, in either orientation."""
    pairs = {tuple(p) for p in held_out.t().tolist()}
    pairs |= {(v, u) for u, v in pairs}
    keep = torch.tensor([(u, v) not in pairs for u, v in edge_index.t().tolist()],
                        dtype=torch.bool)
    return edge_index[:, keep], edge_type[keep]


def endpoint_graph(graph: PreparedH2SData) -> PreparedH2SData:
    """`graph` with every node a possible target (link endpoints; no labels are read)."""
    n = graph.num_nodes
    mask = torch.zeros(n, dtype=torch.bool)
    return replace(graph, target_indices=torch.arange(n), target_y=torch.zeros(n, dtype=torch.long),
                   target_train_mask=mask, target_val_mask=mask, target_test_mask=mask,
                   face_groups=())


class LinkScopes:
    """Endpoint representations `[x*, h0]` from per-endpoint scopes, and link scores."""

    def __init__(self, model: nn.Module, graph: PreparedH2SData, walks_per_anchor: int,
                 max_length: int) -> None:
        self.model = model
        self.graph = endpoint_graph(graph)
        self.walks_per_anchor = walks_per_anchor
        self.max_length = max_length

    def cells(self, endpoints: torch.Tensor, seed: int) -> AnchorCells:
        g = self.graph
        return build_wisp_anchor_cells(g.edge_index, g.edge_type, g.node_type,
                                       g.fact_layer.self_inverse if g.fact_layer else (),
                                       torch.unique(endpoints), self.walks_per_anchor,
                                       self.max_length, seed)

    def representations(self, endpoints: torch.Tensor, cells: AnchorCells, epoch: int = -1,
                        batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(unique endpoints, `[U, 2d]` representations): each endpoint solved once, on its scope."""
        unique = torch.unique(endpoints)
        size = unique.numel() if batch_size is None else batch_size
        parts = []
        for start in range(0, unique.numel(), size):
            part = unique[start:start + size]
            batch = build_scope_batch(self.graph, part, [cells.of(int(t)) for t in part.tolist()])
            data = batch.data
            encoded = self.model.encode(data.x, data.node_type, data.edge_index, data.edge_type,
                                        data.face_groups, data.num_nodes, epoch=epoch,
                                        global_node_id=data.global_node_id,
                                        scope_index=data.scope_index)
            parts.append(encoded.representation[batch.target_rows])
        return unique, torch.cat(parts)

    def score(self, links: torch.Tensor, cells: AnchorCells, epoch: int = -1,
              relation: torch.Tensor | None = None, batch_size: int | None = None) -> torch.Tensor:
        """Decoder scores of `links` [2, L] as in full batch, from the endpoints' scopes."""
        unique, representation = self.representations(links.reshape(-1), cells, epoch, batch_size)
        rows = torch.searchsorted(unique, links)
        return self.model.link_readout(representation, rows, relation)  # type: ignore[misc]

    def train_step(self, positives: torch.Tensor, negatives: torch.Tensor, seed: int, epoch: int,
                   optimizer: torch.optim.Optimizer) -> float:
        """One step: BCE over positives and 1:1 negatives; anchors = the step's endpoints."""
        self.model.train()
        links = torch.cat([positives, negatives], dim=1)
        cells = self.cells(links.reshape(-1), seed)
        optimizer.zero_grad()
        scores = self.score(links, cells, epoch)
        labels = torch.cat([torch.ones(positives.size(1)), torch.zeros(negatives.size(1))])
        loss = nn.functional.binary_cross_entropy_with_logits(scores, labels.to(scores.dtype))
        loss.backward()
        optimizer.step()
        return float(loss)

    def evaluate(self, links: torch.Tensor, seed: int) -> torch.Tensor:
        """Scores of every evaluation link; the endpoints' scopes are computed once (one draw)."""
        self.model.eval()
        with torch.no_grad():
            cells = self.cells(links.reshape(-1), seed)
            return self.score(links, cells, -1, batch_size=EVAL_BATCH_TARGETS)


__all__ = ["LP_POSITIVES_PER_STEP", "LinkScopes", "endpoint_graph", "remove_held_out_facts"]
