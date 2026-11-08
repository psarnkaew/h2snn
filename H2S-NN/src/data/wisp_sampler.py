"""WISP sampler: Walk-sampled, Instance-weighted Sheaf Paths (spec v1.3 §3.2 (a)-(b); Part H).

Target measure on simple walks from an anchor (a sub-probability: dead ends are discarded):

    l ~ Uniform{2..l_max};  for j = 1..l:
        A_j = {actions (relation, direction) with >= 1 fact from v_{j-1} to a node not on the walk}
        if A_j is empty: discard the walk
        a_j ~ Uniform(A_j);   v_j ~ Uniform(facts of a_j from v_{j-1} to unvisited nodes)
    p(w) = 1/(l_max - 1) * prod_j 1/|A_j| * 1/|N_j|

Optional proposal q (task card H3): q_len(l) * prod_j q_act(type(v_{j-1}), a_j) / sum_{a in A_j}
q_act(type(v_{j-1}), a) * 1/|N_j|. Each walk is drawn whole from p or from q (one fair coin per
walk) and gets the weight (1/r) * p(w) / (p(w)/2 + q(w)/2) <= 2/r (Prop. S). Without a proposal
(q = p) the ratio is exactly 1 and m_f is D8's sum of 1/r.

Cells: D8 identity (canonical fact sequence) and summed merge, one `FaceGroup` per length, with a
per-cell relation sequence and signature id. Actions follow D8's half-edge convention: direction
+1/-1 along/against the stored fact, 0 for a self-inverse relation (a self-loop once).

Counts are exact integers: |N_j| = deg_a(v) - #(facts of a from v to visited nodes), from a
(node, action) segment table and a (node, other) lookup; probabilities are float64. Everything is
vectorized over walkers. The Python loops run over hops (<= l_max) and over at most l_max * M
visited half-edges (M = the largest number of facts between one ordered node pair); never over
anchors, walks or signatures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.data.face_geometry import validate_face_orientation
from src.data.path_sampler import Walks, build_half_edges, canonical_cells
from src.data.structures import FaceGroup

NUM_DIRECTIONS = 3  # direction -1, 0 (self-inverse), +1


def action_of(relation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Dense action id of (relation, direction)."""
    return relation * NUM_DIRECTIONS + (direction + 1)


@dataclass(frozen=True, slots=True)
class ActionGraph:
    """Half-edges in (node, action, other) order with per-(node, action) segments, and a
    (node, other) lookup that gives each half-edge's action and position in that order."""

    num_nodes: int
    num_actions: int
    ptr: torch.Tensor          # [N * A + 1] segment starts of (node, action)
    other: torch.Tensor        # [H] in (node, action, other) order
    fact: torch.Tensor
    sign: torch.Tensor
    relation: torch.Tensor
    direction: torch.Tensor
    pair_key: torch.Tensor     # [H] node * N + other, sorted
    pair_position: torch.Tensor  # [H] position of that half-edge in the (node, action, other) order
    pair_action: torch.Tensor  # [H]
    max_multiplicity: int      # largest number of half-edges between one ordered node pair


def build_action_graph(edge_index: torch.Tensor, edge_type: torch.Tensor, num_nodes: int,
                       self_inverse: tuple[int, ...]) -> ActionGraph:
    half = build_half_edges(edge_index, edge_type, num_nodes, self_inverse, backward=True)
    num_relations = max([int(edge_type.max()) + 1 if edge_type.numel() else 0,
                         *(r + 1 for r in self_inverse)])
    num_actions = NUM_DIRECTIONS * max(num_relations, 1)
    n = int(num_nodes)
    action = action_of(half.relation, half.direction)
    order = torch.argsort((half.node * num_actions + action) * n + half.other, stable=True)
    segment = half.node[order] * num_actions + action[order]
    counts = torch.bincount(segment, minlength=n * num_actions)
    ptr = torch.zeros(n * num_actions + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    other = half.other[order]
    pair = half.node[order] * n + other
    pair_order = torch.argsort(pair, stable=True)
    pair_key = pair[pair_order]
    multiplicity = 0
    if pair_key.numel():
        multiplicity = int(torch.unique_consecutive(pair_key, return_counts=True)[1].max())
    return ActionGraph(n, num_actions, ptr, other, half.fact[order], half.sign[order],
                       half.relation[order], half.direction[order], pair_key, pair_order,
                       action[order][pair_order], multiplicity)


@dataclass(frozen=True, slots=True)
class Proposal:
    """q of task card H3: `length[l]` = q_len(l) for l in 2..l_max (index = length), and
    `action[t, a]` >= 0 per (node type, action); both float64, detached, no floors."""

    length: torch.Tensor
    action: torch.Tensor


@dataclass(frozen=True, slots=True)
class WispDraw:
    """Accepted walks, padded to l_max hops (node -1 / fact -1 beyond `length`)."""

    nodes: torch.Tensor       # [W, l_max + 1]
    facts: torch.Tensor       # [W, l_max]
    signs: torch.Tensor
    relations: torch.Tensor
    directions: torch.Tensor
    length: torch.Tensor      # [W]
    anchor: torch.Tensor      # [W]
    log_p: torch.Tensor       # [W] float64
    log_q: torch.Tensor       # [W] float64 (= log_p without a proposal)
    ratio: torch.Tensor       # [W] float64, p / (p/2 + q/2) (1 without a proposal)
    discarded_anchor: torch.Tensor  # anchors of the discarded walks (dead ends)


def _visited_half_edges(graph: ActionGraph, current: torch.Tensor, visited: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Half-edges from `current` to any node in `visited` [W, j]: their action, position in the
    (node, action, other) order, and a validity mask, each [W, j * M]."""
    width = visited.size(1) * graph.max_multiplicity
    if width == 0 or graph.pair_key.numel() == 0:
        empty = torch.zeros(current.numel(), 0, dtype=torch.long)
        return empty, empty, empty.bool()
    key = current.unsqueeze(1) * graph.num_nodes + visited          # [W, j]
    low = torch.searchsorted(graph.pair_key, key)
    high = torch.searchsorted(graph.pair_key, key, right=True)
    offsets = torch.arange(graph.max_multiplicity)
    index = low.unsqueeze(-1) + offsets                                # [W, j, M]
    valid = index < high.unsqueeze(-1)
    index = index.clamp(max=graph.pair_key.numel() - 1)
    action = graph.pair_action[index].reshape(current.numel(), width)
    position = graph.pair_position[index].reshape(current.numel(), width)
    return action, position, valid.reshape(current.numel(), width)


def sample_wisp_walks(graph: ActionGraph, node_type: torch.Tensor, anchors: torch.Tensor,
                      walks_per_anchor: int, max_length: int, generator: torch.Generator,
                      proposal: Proposal | None = None) -> WispDraw:
    if walks_per_anchor <= 0:
        raise ValueError("walks_per_anchor must be positive.")
    if max_length < 2:
        raise ValueError(f"max_length must be >= 2, got {max_length}.")
    anchor = anchors.long().repeat_interleave(walks_per_anchor)
    num = anchor.numel()
    num_actions = graph.num_actions
    action_ids = torch.arange(num_actions)
    length = 2 + (torch.rand(num, generator=generator, dtype=torch.float64)
                  * (max_length - 1)).long().clamp(max=max_length - 2)
    from_q = torch.zeros(num, dtype=torch.bool)
    if proposal is not None:
        from_q = torch.rand(num, generator=generator, dtype=torch.float64) < 0.5
        q_length = torch.multinomial(proposal.length[2:max_length + 1], num, replacement=True,
                                     generator=generator) + 2
        length = torch.where(from_q, q_length, length)
    nodes = torch.full((num, max_length + 1), -1, dtype=torch.long)
    nodes[:, 0] = anchor
    facts = torch.full((num, max_length), -1, dtype=torch.long)
    signs, relations, directions = facts.clone(), facts.clone(), facts.clone()
    log_p = torch.full((num,), -math.log(max_length - 1), dtype=torch.float64)
    log_q = torch.zeros(num, dtype=torch.float64)
    alive = torch.ones(num, dtype=torch.bool)
    current = anchor.clone()
    for hop in range(1, max_length + 1):
        u_action = torch.rand(num, generator=generator, dtype=torch.float64)
        u_fact = torch.rand(num, generator=generator, dtype=torch.float64)
        active = alive & (hop <= length)
        segment = current.unsqueeze(1) * num_actions + action_ids               # [W, A]
        degree = graph.ptr[segment + 1] - graph.ptr[segment]
        v_action, v_position, v_valid = _visited_half_edges(graph, current, nodes[:, :hop])
        seen = torch.zeros(num, num_actions, dtype=torch.long)
        seen.scatter_add_(1, v_action, v_valid.long())
        open_ = degree - seen                                                    # |N(a)|
        available = open_ > 0
        num_available = available.sum(dim=1)
        dead = active & (num_available == 0)
        # target choice: uniform over the available actions
        k = (u_action * num_available).long().clamp(min=0)
        k = torch.minimum(k, (num_available - 1).clamp(min=0))
        chosen = (available.cumsum(dim=1) > k.unsqueeze(1)).long().argmax(dim=1)
        if proposal is not None:
            weight = proposal.action[node_type[current]] * available               # [W, A] f64
            cumulative = weight.cumsum(dim=1)
            total = cumulative[:, -1]
            q_dead = active & from_q & (total <= 0)
            dead = dead | q_dead
            chosen_q = (cumulative > (u_action * total).unsqueeze(1)).long().argmax(dim=1)
            chosen = torch.where(from_q, chosen_q, chosen)
            picked = weight.gather(1, chosen.unsqueeze(1)).squeeze(1)
            step_q = torch.where(total > 0, torch.log(picked) - torch.log(total),
                                 torch.full_like(total, -math.inf))
        alive = alive & ~dead
        active = active & alive
        unvisited = open_.gather(1, chosen.unsqueeze(1)).squeeze(1)
        safe_unvisited = unvisited.clamp(min=1)
        offset = torch.minimum((u_fact * safe_unvisited).long(), safe_unvisited - 1)
        position = graph.ptr[current * num_actions + chosen] + offset
        # skip the visited half-edges of the chosen action (sorted ascending)
        blocked = torch.where(v_valid & (v_action == chosen.unsqueeze(1)), v_position,
                              torch.full_like(v_position, torch.iinfo(torch.long).max))
        for column in torch.sort(blocked, dim=1).values.unbind(dim=1):
            position = position + (column <= position).long()
        position = torch.where(active, position, torch.zeros_like(position))
        log_step = -torch.log(num_available.clamp(min=1).double()) - torch.log(
            safe_unvisited.double())
        log_p = torch.where(active, log_p + log_step, log_p)
        if proposal is not None:
            log_q = torch.where(active, log_q + step_q - torch.log(safe_unvisited.double()), log_q)
        nxt = graph.other[position] if graph.other.numel() else current
        current = torch.where(active, nxt, current)
        nodes[:, hop] = torch.where(active, nxt, nodes[:, hop])
        for store, column in ((facts, graph.fact), (signs, graph.sign),
                              (relations, graph.relation), (directions, graph.direction)):
            store[:, hop - 1] = torch.where(active, column[position], store[:, hop - 1])
    if proposal is None:
        log_q = log_p.clone()
        ratio = torch.ones(num, dtype=torch.float64)
    else:
        log_q = log_q + torch.log(proposal.length[length])
        ratio = 2.0 / (1.0 + torch.exp(log_q - log_p))
    keep = alive
    return WispDraw(nodes[keep], facts[keep], signs[keep], relations[keep], directions[keep],
                    length[keep], anchor[keep], log_p[keep], log_q[keep], ratio[keep],
                    anchor[~keep])


def merge_wisp_cells(draw: WispDraw, node_type: torch.Tensor, edge_index: torch.Tensor,
                     walks_per_anchor: int, merge: str = "sum") -> tuple[FaceGroup, ...]:
    """D8 identity and merge per length: canonical orientation, key = (signature, facts), weight
    (1/r) * ratio summed over the walks of a cell (`drop_duplicates`: max, the T-H2 control)."""
    if merge not in {"sum", "drop_duplicates"}:
        raise ValueError(f"merge must be 'sum' or 'drop_duplicates', got {merge!r}.")
    groups: list[FaceGroup] = []
    for length in sorted(set(draw.length.tolist())):
        rows = draw.length == length
        walks = Walks(draw.nodes[rows, :length + 1], draw.facts[rows, :length],
                      draw.signs[rows, :length], draw.relations[rows, :length],
                      draw.directions[rows, :length], draw.ratio[rows] / walks_per_anchor)
        oriented, signature = canonical_cells(walks, node_type)
        key = torch.cat([signature, oriented.facts], dim=1)
        cells, cell_of = torch.unique(key, dim=0, return_inverse=True)
        weight = torch.zeros(cells.size(0), dtype=torch.float64)
        if merge == "sum":
            weight.index_add_(0, cell_of, oriented.weight)
        else:
            weight.scatter_reduce_(0, cell_of, oriented.weight, reduce="amax", include_self=False)
        first = torch.full((cells.size(0),), cell_of.numel(), dtype=torch.long)
        first.scatter_reduce_(0, cell_of, torch.arange(cell_of.numel()), reduce="amin")
        table, signature_index = torch.unique(signature[first], dim=0, return_inverse=True)
        name = f"wisp_L{length}"
        face_nodes, face_facts = oriented.nodes[first], oriented.facts[first]
        face_signs = oriented.signs[first].to(torch.float32)
        validate_face_orientation(face_nodes, face_facts, edge_index, name,
                                  face_edge_sign=face_signs)
        groups.append(FaceGroup(
            template_name=name,
            node_type_sequence=(),
            face_node_index=face_nodes,
            face_edge_index=face_facts,
            relation_index=oriented.relations[first],
            face_edge_sign=face_signs,
            path_weight=weight.to(torch.float32),
            signature_index=signature_index,
            signature_table=tuple(tuple(row) for row in table.tolist()),
        ))
    return tuple(groups)


def build_wisp_path_groups(edge_index: torch.Tensor, edge_type: torch.Tensor,
                           node_type: torch.Tensor, self_inverse: tuple[int, ...],
                           anchors: torch.Tensor, walks_per_anchor: int, max_length: int,
                           seed: int, proposal: Proposal | None = None,
                           graph: ActionGraph | None = None,
                           merge: str = "sum") -> tuple[FaceGroup, ...]:
    """One WISP draw (seed `seed`) grouped by length. `graph` may be passed to reuse the index."""
    if graph is None:
        graph = build_action_graph(edge_index, edge_type, int(node_type.numel()), self_inverse)
    generator = torch.Generator().manual_seed(int(seed))
    draw = sample_wisp_walks(graph, node_type, anchors, walks_per_anchor, max_length, generator,
                             proposal)
    return merge_wisp_cells(draw, node_type, edge_index, walks_per_anchor, merge)


__all__ = ["ActionGraph", "Proposal", "WispDraw", "action_of", "build_action_graph",
           "build_wisp_path_groups", "merge_wisp_cells", "sample_wisp_walks"]
