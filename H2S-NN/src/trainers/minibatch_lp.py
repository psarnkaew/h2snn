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

# LP_POSITIVES_PER_STEP: positive links per step on the large LP tasks (HGB's 8,192; design §6), the budget
# table's value (`src/configs/config.py`, Part M3b B8), re-exported here.
from src.configs.config import EVAL_BATCH_TARGETS, LP_POSITIVES_PER_STEP
from src.data.large_lp import held_out_fact_mask
from src.models.anchored_solve import check_finite_loss
from src.data.scopes import build_scope_batch
from src.data.structures import PreparedH2SData
from src.data.wisp_sampler import AnchorCells, build_wisp_anchor_cells


def remove_held_out_facts(edge_index: torch.Tensor, edge_type: torch.Tensor,
                          held_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The message graph without every fact whose pair is a held-out pair, in either orientation and
    whatever its relation (vectorised in Part M3b B6 for the large graphs; same result)."""
    keep = ~held_out_fact_mask(edge_index, held_out)
    return edge_index[:, keep], edge_type[keep]


def endpoint_graph(graph: PreparedH2SData) -> PreparedH2SData:
    """`graph` with every node a possible target (link endpoints; no labels are read)."""
    n = graph.num_nodes
    device = graph.edge_index.device
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    return replace(graph, target_indices=torch.arange(n, device=device),
                   target_y=torch.zeros(n, dtype=torch.long, device=device),
                   target_train_mask=mask, target_val_mask=mask, target_test_mask=mask,
                   face_groups=())


class LinkScopes:
    """Endpoint representations `[x*, h0]` from per-endpoint scopes, and link scores."""

    def __init__(self, model: nn.Module, graph: PreparedH2SData, walks_per_anchor: int,
                 max_length: int) -> None:
        self.model = model
        self._graph = endpoint_graph(graph)
        self.walks_per_anchor = walks_per_anchor
        self.max_length = max_length

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def graph(self) -> PreparedH2SData:
        """The endpoint graph on the model's device (card 14: moved once, then draws and scopes run
        there)."""
        device = self._device()
        if self._graph.edge_index.device != device:
            self._graph = self._graph.to(device)
        return self._graph

    def cells(self, endpoints: torch.Tensor, seed: int) -> AnchorCells:
        g = self.graph
        return build_wisp_anchor_cells(g.edge_index, g.edge_type, g.node_type,
                                       g.fact_layer.self_inverse if g.fact_layer else (),
                                       torch.unique(endpoints.to(g.edge_index.device)),
                                       self.walks_per_anchor, self.max_length, seed)

    def representations(self, endpoints: torch.Tensor, cells: AnchorCells, epoch: int = -1,
                        batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(unique endpoints, `[U, 2d]` representations): each endpoint solved once, on its scope."""
        graph = self.graph
        unique = torch.unique(endpoints.to(graph.edge_index.device))
        size = unique.numel() if batch_size is None else batch_size
        parts = []
        for start in range(0, unique.numel(), size):  # evaluation chunks of EVAL_BATCH_TARGETS
            part = unique[start:start + size]
            batch = build_scope_batch(graph, part, cells)  # card 14: on the device, no target loop
            device = self._device()
            data = batch.data
            encoded = self.model.encode(data.x, data.node_type, data.edge_index, data.edge_type,
                                        data.face_groups, data.num_nodes, epoch=epoch,
                                        global_node_id=data.global_node_id,
                                        scope_index=data.scope_index)
            parts.append(encoded.representation[batch.target_rows.to(device)])
        return unique, torch.cat(parts)

    def score(self, links: torch.Tensor, cells: AnchorCells, epoch: int = -1,
              relation: torch.Tensor | None = None, batch_size: int | None = None) -> torch.Tensor:
        """Decoder scores of `links` [2, L] as in full batch, from the endpoints' scopes."""
        unique, representation = self.representations(links.reshape(-1), cells, epoch, batch_size)
        rows = torch.searchsorted(unique, links.to(unique.device)).to(representation.device)
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
        loss = nn.functional.binary_cross_entropy_with_logits(scores, labels.to(scores))
        check_finite_loss(loss, epoch)  # spec v1.16 K5, as every other trainer
        loss.backward()
        optimizer.step()
        return float(loss)

    def evaluate(self, links: torch.Tensor, seed: int) -> torch.Tensor:
        """Scores of every evaluation link; the endpoints' scopes are computed once (one draw)."""
        self.model.eval()
        with torch.no_grad():
            cells = self.cells(links.reshape(-1), seed)
            return self.score(links, cells, -1, batch_size=EVAL_BATCH_TARGETS)


class LinkEpochLoop:
    """The large-LP epoch loop (MB4; decisions D-MB3, D-MB4; card Part M3b B6).

    * Training order: a seeded permutation of the training positives (seed `pass_seed(base, pass)`, as the
      node trainer), continued across epochs; a step takes `LP_POSITIVES_PER_STEP` positives and never
      crosses a pass boundary, so one epoch = one pass over the training positives = `ceil(P / 8192)` steps
      (design §6: "pass over training positives").
    * Negatives 1:1: HGB's `get_train_neg` (one uniform tail per positive) drawn per epoch with seed
      `base + epoch` (`hgb_train_negatives`, spec v1.16 K3); a positive's negative shares its index.
    * Training draw per step: seed `training_draw_seed(base, epoch, step)`; anchors = the step's endpoints.
    * Evaluation every epoch on **all** validation links (positives and their fixed negatives): one draw per
      run from the base seed, endpoint scopes computed once per evaluation and shared by the links
      (`LinkScopes.evaluate`); the metric is ROC-AUC.
    * Early stopping: patience counted in epochs (D-MB4); the best validation epoch is kept.
    * Stop conditions (tau, log-weight range per forward, non-finite g, T_max) raise inside the model and end
      the run (`logged_stop_conditions`); evaluation solves start from the initial guess (MB6a).
    """

    def __init__(self, scopes: LinkScopes, optimizer: torch.optim.Optimizer, base_seed: int, epochs: int,
                 patience: int, dst_range: tuple[int, int], logger=None) -> None:
        self.scopes = scopes
        self.optimizer = optimizer
        self.base_seed = int(base_seed)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.dst_range = dst_range
        self.logger = logger
        self.history: list[dict[str, float]] = []
        self.best_epoch = -1

    def epoch_steps(self, num_positives: int) -> int:
        return -(-int(num_positives) // LP_POSITIVES_PER_STEP)

    def train_epoch(self, order, train_pos: torch.Tensor, epoch: int) -> float:
        from src.data.hgb_lp_negatives import hgb_train_negatives
        from src.trainers.minibatch_trainer import training_draw_seed

        negatives = hgb_train_negatives(train_pos, self.dst_range, self.base_seed + epoch)
        losses = []
        for step in range(self.epoch_steps(train_pos.size(1))):
            index = order.next(LP_POSITIVES_PER_STEP)
            losses.append(self.scopes.train_step(train_pos[:, index], negatives[:, index],
                                                 training_draw_seed(self.base_seed, epoch, step), epoch,
                                                 self.optimizer))
        return sum(losses) / len(losses)

    def validate(self, val_pos: torch.Tensor, val_neg: torch.Tensor) -> float:
        from src.utils.misc import compute_link_prediction_metrics

        scores = self.scopes.evaluate(torch.cat([val_pos, val_neg], dim=1), self.base_seed)
        pos, neg = scores.split([val_pos.size(1), val_neg.size(1)])
        return float(compute_link_prediction_metrics(pos.float(), neg.float())["auroc"])

    def fit(self, train_pos: torch.Tensor, val_pos: torch.Tensor, val_neg: torch.Tensor) -> int:
        """Train for at most `epochs`; returns the best validation epoch."""
        from src.trainers.minibatch_trainer import TrainingOrder
        from src.trainers.stop_conditions import logged_stop_conditions

        order = TrainingOrder(torch.arange(train_pos.size(1)), self.base_seed)
        best, bad_epochs = float("-inf"), 0
        with logged_stop_conditions(self.logger):
            for epoch in range(self.epochs):
                loss = self.train_epoch(order, train_pos, epoch)
                auroc = self.validate(val_pos, val_neg)
                self.history.append({"epoch": epoch, "loss": loss, "val_auroc": auroc})
                if auroc > best:
                    best, bad_epochs, self.best_epoch = auroc, 0, epoch
                else:
                    bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        return self.best_epoch


__all__ = ["LP_POSITIVES_PER_STEP", "LinkEpochLoop", "LinkScopes", "endpoint_graph", "remove_held_out_facts"]
