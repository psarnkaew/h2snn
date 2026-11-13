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


def count_actions(edge_type: torch.Tensor, self_inverse: tuple[int, ...]) -> int:
    """Size of the dense action id space (3 directions per relation id)."""
    num_relations = max([int(edge_type.max()) + 1 if edge_type.numel() else 0,
                         *(r + 1 for r in self_inverse)])
    return NUM_DIRECTIONS * max(num_relations, 1)


def build_action_graph(edge_index: torch.Tensor, edge_type: torch.Tensor, num_nodes: int,
                       self_inverse: tuple[int, ...]) -> ActionGraph:
    half = build_half_edges(edge_index, edge_type, num_nodes, self_inverse, backward=True)
    num_actions = count_actions(edge_type, self_inverse)
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


def unvisited_fact_step(graph: ActionGraph, current: torch.Tensor, chosen: torch.Tensor,
                        open_: torch.Tensor, v_action: torch.Tensor, v_position: torch.Tensor,
                        v_valid: torch.Tensor, u_fact: torch.Tensor, active: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """WISP's fact choice of one hop: a uniform fact of action `chosen` from `current` to a node
    not on the walk (`open_` = unvisited counts per action, the visited half-edges from
    `_visited_half_edges`), driven by `u_fact`. Returns the half-edge position (0 where not
    `active`) and max(|N|, 1). Shared by WISP and the fixed-literature-schema row (Part P P0b)."""
    num_actions = graph.num_actions
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
    return position, safe_unvisited


def open_actions(graph: ActionGraph, current: torch.Tensor, visited: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`[W, A]` number of facts of each action from `current` to nodes not in `visited` [W, j],
    and the visited half-edges (`_visited_half_edges`)."""
    num_actions = graph.num_actions
    segment = current.unsqueeze(1) * num_actions + torch.arange(num_actions)
    degree = graph.ptr[segment + 1] - graph.ptr[segment]
    v_action, v_position, v_valid = _visited_half_edges(graph, current, visited)
    seen = torch.zeros(current.numel(), num_actions, dtype=torch.long)
    seen.scatter_add_(1, v_action, v_valid.long())
    return degree - seen, v_action, v_position, v_valid


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
    # Per-step mixture density prod_j (p_j/2 + q_j/2) (with the length term): computed for the
    # T-H3 control only; the estimator uses the walk-level ratio (`_importance_ratio`).
    log_mix_step = torch.zeros(num, dtype=torch.float64)
    if proposal is not None:
        log_mix_step = torch.logaddexp(
            torch.full((num,), -math.log(max_length - 1), dtype=torch.float64),
            torch.log(proposal.length[length])) - math.log(2.0)
    alive = torch.ones(num, dtype=torch.bool)
    current = anchor.clone()
    for hop in range(1, max_length + 1):
        u_action = torch.rand(num, generator=generator, dtype=torch.float64)
        u_fact = torch.rand(num, generator=generator, dtype=torch.float64)
        active = alive & (hop <= length)
        open_, v_action, v_position, v_valid = open_actions(graph, current, nodes[:, :hop])  # |N(a)|
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
        position, safe_unvisited = unvisited_fact_step(graph, current, chosen, open_, v_action,
                                                       v_position, v_valid, u_fact, active)
        log_step = -torch.log(num_available.clamp(min=1).double()) - torch.log(
            safe_unvisited.double())
        log_p = torch.where(active, log_p + log_step, log_p)
        if proposal is not None:
            log_step_q = step_q - torch.log(safe_unvisited.double())
            log_q = torch.where(active, log_q + log_step_q, log_q)
            log_mix = torch.logaddexp(log_step, log_step_q) - math.log(2.0)
            log_mix_step = torch.where(active, log_mix_step + log_mix, log_mix_step)
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
        ratio = _importance_ratio(log_p, log_q, log_mix_step)
    keep = alive
    return WispDraw(nodes[keep], facts[keep], signs[keep], relations[keep], directions[keep],
                    length[keep], anchor[keep], log_p[keep], log_q[keep], ratio[keep],
                    anchor[~keep])


def _importance_ratio(log_p: torch.Tensor, log_q: torch.Tensor,
                      log_mix_step: torch.Tensor) -> torch.Tensor:
    """Walk-level defensive ratio p / (p/2 + q/2) = 2 / (1 + q/p) <= 2 (Prop. S). `log_mix_step`
    (the per-step mixture) is not used: it is the T-H3 control's density."""
    return 2.0 / (1.0 + torch.exp(log_q - log_p))


def proposal_from_mass(groups: tuple[FaceGroup, ...], weights: tuple[torch.Tensor, ...],
                       node_type: torch.Tensor, num_node_types: int, num_actions: int,
                       max_length: int, self_inverse: tuple[int, ...]) -> Proposal:
    """q of task card H3 from the previous epoch's path mass w_f m_f (detached): q_len(l) is the
    share of mass in length l; q_act(t, a) is the mass of the cells that take action a from a node
    of type t, counted in both orientations of each cell (a cell is a walk from either end). No
    floors: an action without mass has q = 0, and the walk-level mixture keeps the ratio <= 2."""
    length = torch.zeros(max_length + 1, dtype=torch.float64)
    action = torch.zeros(num_node_types, num_actions, dtype=torch.float64)
    shared = torch.zeros(max(num_actions // NUM_DIRECTIONS, 1), dtype=torch.bool)
    if self_inverse:
        shared[list(self_inverse)] = True
    nonempty = [g for g in groups if g.num_faces > 0]
    if len(nonempty) != len(weights):
        raise ValueError(f"{len(weights)} weight tensors for {len(nonempty)} non-empty groups.")
    for group, w in zip(nonempty, weights):
        mass = (w.detach().double().cpu() * group.path_weight.detach().double().cpu())
        length[group.length] += mass.sum()
        nodes = group.face_node_index.cpu()
        relation = group.relation_index.cpu()
        direction = torch.where(shared[relation], torch.zeros_like(relation),
                                group.face_edge_sign.cpu().sign().long())
        per_hop = mass.unsqueeze(1).expand(-1, group.length)
        for types, hop_direction in ((node_type[nodes[:, :-1]], direction),
                                     (node_type[nodes[:, 1:]], -direction)):
            act = action_of(relation, hop_direction)
            action.index_put_((types.reshape(-1), act.reshape(-1)), per_hop.reshape(-1),
                              accumulate=True)
    total = length[2:].sum()
    if total > 0:
        length = length / total
    else:
        length[2:] = 1.0 / (max_length - 1)
    return Proposal(length=length, action=action)


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
        groups.append(merge_length_group(walks, node_type, edge_index, f"wisp_L{length}", merge))
    return tuple(groups)


def merge_length_group(walks: Walks, node_type: torch.Tensor, edge_index: torch.Tensor,
                       name: str, merge: str = "sum") -> FaceGroup:
    """One length's walks (each carrying its weight) as one cell group: D8 identity (canonical
    orientation, key = (signature, facts)), weights summed per cell (`drop_duplicates`: max), a
    per-cell signature id. Shared by WISP and the fixed-literature-schema row (Part O)."""
    return _merge_cells(walks, node_type, edge_index, name, merge, None)[0]


def _merge_cells(walks: Walks, node_type: torch.Tensor, edge_index: torch.Tensor, name: str,
                 merge: str, anchor: torch.Tensor | None
                 ) -> tuple[FaceGroup, torch.Tensor | None]:
    """`merge_length_group`; with `anchor` [W] (Part MB, MB2) the anchor joins the merge key, so
    `m_f` sums one anchor's walks only, and the anchor of every cell is returned."""
    oriented, signature = canonical_cells(walks, node_type)
    key = torch.cat([signature, oriented.facts], dim=1)
    if anchor is not None:
        key = torch.cat([anchor.view(-1, 1).to(key.dtype), key], dim=1)
    cells, cell_of = torch.unique(key, dim=0, return_inverse=True)
    weight = torch.zeros(cells.size(0), dtype=torch.float64)
    if merge == "sum":
        weight.index_add_(0, cell_of, oriented.weight)
    else:
        weight.scatter_reduce_(0, cell_of, oriented.weight, reduce="amax", include_self=False)
    first = torch.full((cells.size(0),), cell_of.numel(), dtype=torch.long)
    first.scatter_reduce_(0, cell_of, torch.arange(cell_of.numel()), reduce="amin")
    table, signature_index = torch.unique(signature[first], dim=0, return_inverse=True)
    face_nodes, face_facts = oriented.nodes[first], oriented.facts[first]
    face_signs = oriented.signs[first].to(torch.float32)
    validate_face_orientation(face_nodes, face_facts, edge_index, name,
                              face_edge_sign=face_signs)
    group = FaceGroup(
        template_name=name,
        node_type_sequence=(),
        face_node_index=face_nodes,
        face_edge_index=face_facts,
        relation_index=oriented.relations[first],
        face_edge_sign=face_signs,
        path_weight=weight.to(torch.float32),
        signature_index=signature_index,
        signature_table=tuple(tuple(row) for row in table.tolist()),
    )
    return group, (None if anchor is None else anchor[first])


# ----------------------------------------------------------------------------- mini-batch draws
#: Anchors per chunk of a mini-batch draw (decision N2, author 2026-11-13; a constant, never
#: tuned). Chunking changes which random numbers a walk uses, never `p`.
WISP_CHUNK_ANCHORS = 4096
#: chunk seed = seed * CHUNK_SEED_STRIDE + chunk index: distinct for every (seed, chunk index)
#: with chunk index < CHUNK_SEED_STRIDE (2**20 chunks = 4.3e9 anchors at 4,096 per chunk).
CHUNK_SEED_STRIDE = 1 << 20


def chunk_seed(seed: int, chunk_index: int) -> int:
    """The generator seed of chunk `chunk_index` of a mini-batch draw with seed `seed` (N2)."""
    if not 0 <= int(chunk_index) < CHUNK_SEED_STRIDE:
        raise ValueError(f"chunk index {chunk_index} outside [0, {CHUNK_SEED_STRIDE}).")
    return int(seed) * CHUNK_SEED_STRIDE + int(chunk_index)


def sample_wisp_walks_chunked(graph: ActionGraph, node_type: torch.Tensor, anchors: torch.Tensor,
                              walks_per_anchor: int, max_length: int, seed: int,
                              chunk_size: int = WISP_CHUNK_ANCHORS) -> WispDraw:
    """A mini-batch draw from `p` (`proposal: uniform`): `sample_wisp_walks` on consecutive chunks
    of `chunk_size` anchors, chunk `k` with its own generator seeded by `chunk_seed(seed, k)`. The
    result depends only on (seed, chunk size, anchor order). Full-batch draws do not use this."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    anchors = anchors.long().reshape(-1)
    if anchors.numel() == 0:
        raise ValueError("a draw needs at least one anchor.")
    parts = []
    for k, start in enumerate(range(0, anchors.numel(), chunk_size)):
        generator = torch.Generator().manual_seed(chunk_seed(seed, k))
        parts.append(sample_wisp_walks(graph, node_type, anchors[start:start + chunk_size],
                                       walks_per_anchor, max_length, generator))
    return WispDraw(*(torch.cat([getattr(part, field) for part in parts])
                      for field in WispDraw.__slots__))


@dataclass(frozen=True, slots=True)
class AnchorCells:
    """Cells merged per anchor (one group per length) and the anchor of every cell."""

    groups: tuple[FaceGroup, ...]
    anchors: tuple[torch.Tensor, ...]

    def of(self, anchor: int) -> tuple[FaceGroup, ...]:
        """The cells of one anchor (its scope's cells), one group per length present."""
        out = []
        for group, owner in zip(self.groups, self.anchors):
            rows = (owner == int(anchor)).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            out.append(FaceGroup(
                template_name=group.template_name, node_type_sequence=group.node_type_sequence,
                face_node_index=group.face_node_index[rows],
                face_edge_index=group.face_edge_index[rows],
                relation_index=group.relation_index[rows],
                face_edge_sign=group.face_edge_sign[rows],
                path_weight=group.path_weight[rows],
                signature_index=group.signature_index[rows],
                signature_table=group.signature_table))
        return tuple(out)


def merge_wisp_cells_per_anchor(draw: WispDraw, node_type: torch.Tensor,
                                edge_index: torch.Tensor, walks_per_anchor: int) -> AnchorCells:
    """D8 identity per anchor (MB2): key = (anchor, signature, facts), `m_f` = sum of (1/r) * ratio
    over that anchor's walks only, so one instance drawn from two anchors gives two cells."""
    groups, anchors = [], []
    for length in sorted(set(draw.length.tolist())):
        rows = draw.length == length
        walks = Walks(draw.nodes[rows, :length + 1], draw.facts[rows, :length],
                      draw.signs[rows, :length], draw.relations[rows, :length],
                      draw.directions[rows, :length], draw.ratio[rows] / walks_per_anchor)
        group, owner = _merge_cells(walks, node_type, edge_index, f"wisp_L{length}", "sum",
                                    draw.anchor[rows])
        groups.append(group)
        anchors.append(owner)
    return AnchorCells(groups=tuple(groups), anchors=tuple(anchors))


def build_wisp_anchor_cells(edge_index: torch.Tensor, edge_type: torch.Tensor,
                            node_type: torch.Tensor, self_inverse: tuple[int, ...],
                            anchors: torch.Tensor, walks_per_anchor: int, max_length: int,
                            seed: int, chunk_size: int = WISP_CHUNK_ANCHORS,
                            graph: ActionGraph | None = None) -> AnchorCells:
    """A mini-batch draw (chunked, from `p`) merged per anchor, on the whole given graph (the
    scope is cut afterwards, `src/data/scopes.py`)."""
    if graph is None:
        graph = build_action_graph(edge_index, edge_type, int(node_type.numel()), self_inverse)
    draw = sample_wisp_walks_chunked(graph, node_type, anchors, walks_per_anchor, max_length,
                                     seed, chunk_size)
    return merge_wisp_cells_per_anchor(draw, node_type, edge_index, walks_per_anchor)


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
           "merge_length_group", "open_actions", "unvisited_fact_step",
           "AnchorCells", "CHUNK_SEED_STRIDE", "WISP_CHUNK_ANCHORS", "build_wisp_anchor_cells",
           "chunk_seed", "merge_wisp_cells_per_anchor", "sample_wisp_walks_chunked",
           "build_wisp_path_groups", "count_actions", "merge_wisp_cells", "proposal_from_mass",
           "sample_wisp_walks"]
