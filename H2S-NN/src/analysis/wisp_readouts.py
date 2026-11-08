"""WISP read-outs (spec v1.3 §3.2; task card Part H, H4). Evaluation only: every function takes a
frozen model (or its cell weights) and changes nothing it does not restore; this module imports
no training code.

    I_pi        share of sum_f w_f m_f by full-walk schema (the cell's canonical signature)
    I_sigma     share of pair weight by sub-path schema: with uniform participation (Prop. K),
                cell f gives each pair (i < j) the weight w_f m_f / (2 |f|) and the energy
                E_f = sum_{i<j} w_f m_f / (2 |f|) ||T_i x_i - T_j x_j||^2
    I_v_sigma   the same, restricted to the pairs with node v at one end, normalised per node
    overlap     share of pair weight whose sub-path instance also occurs in another sampled cell
    delta_L     validation-loss change when sigma's pair terms are removed (mean over T draws)
    kendall_tau rank agreement of two schema scores, with a permutation null

A sub-path schema is the canonical signature (t_i, r, d + 1, ..., t_j) of positions i..j of a cell
(the smaller of the two orientations, lexicographically).
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import torch

from src.data.structures import FaceGroup
from src.models.vertex_path_laplacian import VertexPathSheafLaplacian


# ----------------------------------------------------------------------------- schemas
def _directions(group: FaceGroup, self_inverse: tuple[int, ...]) -> torch.Tensor:
    relations = group.relation_index.cpu()
    shared = torch.zeros(int(relations.max()) + 1 if relations.numel() else 1, dtype=torch.bool)
    for r in self_inverse:
        if r < shared.numel():
            shared[r] = True
    return torch.where(shared[relations], torch.zeros_like(relations),
                       group.face_edge_sign.cpu().sign().long())


def _signature(types: torch.Tensor, relations: torch.Tensor, directions: torch.Tensor
               ) -> torch.Tensor:
    """`[P, 3k + 1]`: t_0, r_1, d_1 + 1, t_1, ..., r_k, d_k + 1, t_k."""
    columns = [types[:, :1]]
    for j in range(relations.size(1)):
        columns += [relations[:, j:j + 1], directions[:, j:j + 1] + 1, types[:, j + 1:j + 2]]
    return torch.cat(columns, dim=1)


def _canonical(types, relations, directions) -> torch.Tensor:
    forward = _signature(types, relations, directions)
    backward = _signature(types.flip(1), relations.flip(1), -directions.flip(1))
    differ = forward != backward
    first = differ.int().argmax(dim=1, keepdim=True)
    take_backward = differ.any(dim=1) & (backward.gather(1, first) < forward.gather(1, first)).squeeze(1)
    return torch.where(take_backward.unsqueeze(1), backward, forward)


class PairIndex(NamedTuple):
    """Every pair (i < j) of every cell of one group: cell id, positions, sub-path schema key."""

    cell: torch.Tensor      # [Q]
    first: torch.Tensor     # [Q] position i
    second: torch.Tensor    # [Q] position j
    schema: list[tuple[int, ...]]  # [Q] canonical sub-path signature


def pair_index(group: FaceGroup, node_type: torch.Tensor, self_inverse: tuple[int, ...]
               ) -> PairIndex:
    nodes = group.face_node_index.cpu()
    types = node_type.cpu()[nodes]
    relations = group.relation_index.cpu()
    directions = _directions(group, self_inverse)
    cells, firsts, seconds, schemas = [], [], [], []
    size = nodes.size(1)
    for i in range(size):
        for j in range(i + 1, size):  # loops over positions (<= 10 pairs per cell), not cells
            key = _canonical(types[:, i:j + 1], relations[:, i:j], directions[:, i:j])
            cells.append(torch.arange(nodes.size(0)))
            firsts.append(torch.full((nodes.size(0),), i))
            seconds.append(torch.full((nodes.size(0),), j))
            schemas.extend(tuple(row) for row in key.tolist())
    return PairIndex(torch.cat(cells), torch.cat(firsts), torch.cat(seconds), schemas)


def full_schema_keys(group: FaceGroup, node_type: torch.Tensor,
                     self_inverse: tuple[int, ...]) -> list[tuple[int, ...]]:
    nodes = group.face_node_index.cpu()
    return [tuple(r) for r in _canonical(node_type.cpu()[nodes], group.relation_index.cpu(),
                                         _directions(group, self_inverse)).tolist()]


def _cell_mass(group: FaceGroup, weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().double().cpu() * group.path_weight.detach().double().cpu()


# ----------------------------------------------------------------------------- shares
def i_pi(groups, weights, node_type, self_inverse=()) -> dict[tuple[int, ...], float]:
    """Share of sum_f w_f m_f per full-walk schema (sums to 1)."""
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for group, weight in zip(groups, weights):
        for key, value in zip(full_schema_keys(group, node_type, self_inverse),
                              _cell_mass(group, weight).tolist()):
            totals[key] += value
    grand = sum(totals.values())
    return {k: v / grand for k, v in totals.items()}


def mean_log_weight(groups, weights, node_type, self_inverse=()) -> dict[tuple[int, ...], float]:
    """Mean learned log-weight g(xi_f) = log w_f per cell, by full-walk schema (frequency-free;
    the primary S2 criterion of the H0 addendum)."""
    sums: dict[tuple[int, ...], float] = defaultdict(float)
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    for group, weight in zip(groups, weights):
        for key, value in zip(full_schema_keys(group, node_type, self_inverse),
                              weight.detach().double().cpu().log().tolist()):
            sums[key] += value
            counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums}


def _pair_weights(group: FaceGroup, weight: torch.Tensor, pairs: PairIndex) -> torch.Tensor:
    size = group.face_node_index.size(1)
    return _cell_mass(group, weight)[pairs.cell] / (2.0 * size)


def i_sigma(groups, weights, node_type, self_inverse=()) -> dict[tuple[int, ...], float]:
    """Share of pair weight per sub-path schema (sums to 1; Prop. K)."""
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    for group, weight in zip(groups, weights):
        pairs = pair_index(group, node_type, self_inverse)
        for key, value in zip(pairs.schema, _pair_weights(group, weight, pairs).tolist()):
            totals[key] += value
    grand = sum(totals.values())
    return {k: v / grand for k, v in totals.items()}


def i_v_sigma(groups, weights, node_type, self_inverse=()) -> dict[int, dict[tuple[int, ...], float]]:
    """Per node v: share of the pair weight of the pairs with v at one end, by sub-path schema."""
    per_node: dict[int, dict[tuple[int, ...], float]] = defaultdict(lambda: defaultdict(float))
    for group, weight in zip(groups, weights):
        pairs = pair_index(group, node_type, self_inverse)
        values = _pair_weights(group, weight, pairs).tolist()
        nodes = group.face_node_index.cpu()
        ends_a = nodes[pairs.cell, pairs.first].tolist()
        ends_b = nodes[pairs.cell, pairs.second].tolist()
        for key, value, a, b in zip(pairs.schema, values, ends_a, ends_b):
            per_node[a][key] += value
            per_node[b][key] += value
    out = {}
    for v, table in per_node.items():
        total = sum(table.values())
        out[v] = {k: x / total for k, x in table.items()}
    return out


def overlap_ratio(groups, weights, node_type, self_inverse=()) -> float:
    """Share of pair weight whose sub-path instance (its fact sequence, either orientation) also
    occurs in another sampled cell."""
    instances: dict[tuple[int, ...], set] = defaultdict(set)
    records = []
    for g, (group, weight) in enumerate(zip(groups, weights)):
        pairs = pair_index(group, node_type, self_inverse)
        facts = group.face_edge_index.cpu()
        values = _pair_weights(group, weight, pairs).tolist()
        for cell, i, j, value in zip(pairs.cell.tolist(), pairs.first.tolist(),
                                     pairs.second.tolist(), values):
            seq = tuple(facts[cell, i:j].tolist())
            key = min(seq, seq[::-1])
            instances[key].add((g, cell))
            records.append((key, value))
    total = sum(v for _, v in records)
    shared = sum(v for key, v in records if len(instances[key]) > 1)
    return shared / total if total > 0 else 0.0


# ----------------------------------------------------------------------------- pair-form term
class PairCellTerm:
    """Evaluation-only pair form of one group's cells (Prop. K, uniform kappa):
    (L_f x)_{v_i} = (w_f / |f|) T_i^T sum_{j kept} (T_i x_i - T_j x_j), and the matching degree
    sum_{j kept} (w_f / |f|) s_i^2. With every pair kept it equals `PathCellTerm`."""

    def __init__(self, transport, weights: torch.Tensor, keep: torch.Tensor) -> None:
        self.transport = transport
        self.weights = weights
        self.keep = keep.to(weights.dtype)  # [P, n, n] symmetric 0/1, zero diagonal
        self.size = transport.face_node_index.size(1)

    def apply(self, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
        y = self.transport.apply(x)                                       # [P, n, d]
        degree = self.keep.sum(dim=2, keepdim=True)
        residual = degree * y - torch.bmm(self.keep, y)
        residual = residual * (self.weights / self.size).view(-1, 1, 1)
        return self.transport.apply_T(residual, num_nodes)

    def diagonal(self, num_nodes: int) -> torch.Tensor:
        per = (self.weights / self.size).unsqueeze(1) * self.keep.sum(dim=2) * \
            self.transport.squared_scales()
        out = per.new_zeros(num_nodes)
        out.index_add_(0, self.transport.face_node_index.reshape(-1), per.reshape(-1))
        return out


def cell_energy_pairs(transport, weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """`[P]` sum_{i<j} w_f / (2 |f|) ||T_i x_i - T_j x_j||^2 (Prop. K's pair decomposition)."""
    y = transport.apply(x)
    size = y.size(1)
    total = y.new_zeros(y.size(0))
    for i in range(size):
        for j in range(i + 1, size):
            total = total + (y[:, i] - y[:, j]).square().sum(dim=-1)
    return weights * total / (2.0 * size)


def schema_keep(group: FaceGroup, node_type: torch.Tensor, self_inverse: tuple[int, ...],
                schema: tuple[int, ...] | None, device=None) -> torch.Tensor:
    """`[P, n, n]` symmetric 0/1 pair mask with a zero diagonal: every pair kept, except (when
    `schema` is given) the pairs whose sub-path schema is `schema`, dropped in both orders."""
    size = group.face_node_index.size(1)
    keep = torch.ones(group.num_faces, size, size, device=device) - torch.eye(size, device=device)
    if schema is not None:
        pairs = pair_index(group, node_type, self_inverse)
        hit = torch.tensor([key == schema for key in pairs.schema], dtype=torch.bool)
        if bool(hit.any()):
            c, i, j = pairs.cell[hit].to(device), pairs.first[hit].to(device), pairs.second[hit].to(device)
            keep[c, i, j] = 0.0
            keep[c, j, i] = 0.0
    return keep


def delta_loss(model, x, node_type, edge_index, edge_type, num_nodes, draws, target_index,
               labels, schema: tuple[int, ...], self_inverse=()) -> float:
    """Validation-loss change when `schema`'s pair terms are removed (mean over the given
    evaluation draws), with the frozen `model` in eval mode. A schema absent from every draw gives
    exactly 0: both losses come from the same pair-form computation."""
    was_training = model.training
    saved_stats = dict(model.last_path_stats)
    model.eval()
    try:
        deltas = []
        with torch.no_grad():
            h0 = model.encode_nodes(x, node_type)
            for groups in draws:
                operator = model.build_operator(h0, edge_index, edge_type, groups, num_nodes)
                nonempty = [g for g in groups if g.num_faces > 0]
                losses = []
                for drop in (False, True):
                    terms = []
                    for group, term in zip(nonempty, operator.laplacian.path_terms):
                        keep = schema_keep(group, node_type, self_inverse,
                                           schema if drop else None, h0.device)
                        terms.append(PairCellTerm(term.incidence, term.weights, keep))
                    laplacian = VertexPathSheafLaplacian(
                        operator.laplacian.edge_maps, edge_index, num_nodes, tuple(terms),
                        operator.laplacian.gamma, checkpoint=False)
                    x_star = model.propagate(h0, operator._replace(laplacian=laplacian))
                    logits = model.readout(torch.cat([x_star, h0], dim=-1)[target_index])
                    losses.append(float(torch.nn.functional.cross_entropy(logits, labels)))
                deltas.append(losses[1] - losses[0])
        return sum(deltas) / len(deltas)
    finally:
        model.train(was_training)
        model.last_path_stats = saved_stats


# ----------------------------------------------------------------------------- stability
def kendall_tau(a: dict, b: dict, permutations: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Kendall tau-b of two score dicts over the union of keys (missing = 0), and the one-sided
    permutation p-value P(tau_perm >= tau)."""
    keys = sorted(set(a) | set(b))
    x = torch.tensor([a.get(k, 0.0) for k in keys], dtype=torch.float64)
    y = torch.tensor([b.get(k, 0.0) for k in keys], dtype=torch.float64)

    def tau(u, v):
        du = torch.sign(u.unsqueeze(0) - u.unsqueeze(1))
        dv = torch.sign(v.unsqueeze(0) - v.unsqueeze(1))
        num = (du * dv).sum()
        den = torch.sqrt((du != 0).sum().double() * (dv != 0).sum().double())
        return float(num / den) if float(den) > 0 else 0.0

    observed = tau(x, y)
    generator = torch.Generator().manual_seed(seed)
    hits = sum(tau(x, y[torch.randperm(y.numel(), generator=generator)]) >= observed
               for _ in range(permutations))
    return observed, (hits + 1) / (permutations + 1)


__all__ = ["PairCellTerm", "cell_energy_pairs", "delta_loss", "full_schema_keys", "i_pi",
           "i_sigma", "i_v_sigma", "kendall_tau", "mean_log_weight", "overlap_ratio",
           "pair_index", "schema_keep"]
