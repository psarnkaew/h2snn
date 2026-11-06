"""Walk sampler, cell identity and merged weights (v2 decision D8, plan §9.2).

Stated distribution. For an anchor `a` and a schema `pi` with hops `(psi_j, sigma_j)`, a
walk picks at hop `j` a uniform neighbour among the facts of relation `psi_j` incident to
the current node in direction `sigma_j` (+1: facts stored out of the node, -1: into it,
0 for a self-inverse relation: every incident fact, a self-loop once), so

    p(walk | a, pi) = prod_j 1 / deg_{psi_j, sigma_j}(v_{j-1}),

and walks that revisit a node are rejected (contribute 0). Each of the `r` walks per
(anchor, schema) that is accepted gives its cell weight `1 / r`; cells with the same
identity are merged and their weights SUMMED into `m_f`, so

    E[ L_P_hat ] = sum_pi sum_a E_{walk ~ p(.|a, pi)}[ 1{walk simple} L_{f(walk)} ]

(the target operator of plan §9.2). Without schemas (link prediction, graph
classification; decision 4) a walk picks at each hop a uniform neighbour among ALL facts
incident to the current node (any relation; under `untied` stored out-edges only), and
cells are grouped by the realized (node type, relation, direction) sequence: the same
estimator with `deg` the total incident-fact degree.

Identity. A cell is the tuple of its fact ids in canonical orientation plus its group. The
orientation is the one whose signature (node types, relations, directions) reads smaller;
for a palindromic signature, the one whose fact sequence is lexicographically smaller. A
path and its reversal are therefore one cell, and every merged cell is built once from
its canonical orientation (decision 7). Paths through different reciprocal facts are
different cells. Everything is vectorized over walkers (per-hop CSR, `torch.unique`); there
is no Python loop over nodes, anchors or walks.

`uniform_instance` (ablation, L <= 2): instead of walk probabilities, each draw is uniform
over the schema's walks from the anchor (first hop chosen proportional to the number of
completions), non-simple draws rejected, so accepted draws are uniform over instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from src.data.face_geometry import validate_face_orientation, validate_relation_sequence_constant
from src.data.schemas import ResolvedSchema
from src.data.structures import FaceGroup

PATH_SAMPLERS = ("walk", "dfs", "uniform_instance")


@dataclass(frozen=True, slots=True)
class HalfEdges:
    """All (node -> other) traversals of the facts: one forward entry per fact and, under
    the tied layer, one backward entry (a self-loop of a self-inverse relation once)."""

    node: torch.Tensor
    other: torch.Tensor
    fact: torch.Tensor
    sign: torch.Tensor       # +1 along the stored orientation, -1 against it
    relation: torch.Tensor
    direction: torch.Tensor  # sign, or 0 for a self-inverse relation
    num_nodes: int


def build_half_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    self_inverse: tuple[int, ...],
    backward: bool,
) -> HalfEdges:
    src, dst = edge_index[0].long(), edge_index[1].long()
    facts = torch.arange(src.numel(), device=src.device)
    # Sized by the largest id among the edges AND the self-inverse list: a graph-
    # classification batch need not contain every relation.
    size = max([int(edge_type.max()) if edge_type.numel() else 0, *self_inverse]) + 1
    shared = torch.zeros(size, dtype=torch.bool, device=src.device)
    if self_inverse:
        shared[list(self_inverse)] = True
    is_shared = shared[edge_type] if edge_type.numel() else torch.zeros_like(src, dtype=torch.bool)
    forward_dir = torch.where(is_shared, torch.zeros_like(src), torch.ones_like(src))
    parts = [(src, dst, facts, torch.ones_like(src), edge_type.long(), forward_dir)]
    if backward:
        keep = ~(is_shared & (src == dst))
        parts.append((dst[keep], src[keep], facts[keep], -torch.ones_like(src[keep]),
                      edge_type[keep].long(), -forward_dir[keep]))
    columns = [torch.cat([part[i] for part in parts]) for i in range(6)]
    return HalfEdges(*columns, num_nodes=int(num_nodes))


@dataclass(frozen=True, slots=True)
class CSR:
    ptr: torch.Tensor
    other: torch.Tensor
    fact: torch.Tensor
    sign: torch.Tensor
    relation: torch.Tensor
    direction: torch.Tensor

    def degree(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.ptr[nodes + 1] - self.ptr[nodes]


def build_csr(half: HalfEdges, mask: torch.Tensor) -> CSR:
    """Half-edges selected by `mask`, sorted by node (stable), with row pointers."""
    node = half.node[mask]
    order = torch.argsort(node, stable=True)
    counts = torch.bincount(node, minlength=half.num_nodes)
    ptr = torch.zeros(half.num_nodes + 1, dtype=torch.long, device=node.device)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return CSR(ptr, half.other[mask][order], half.fact[mask][order], half.sign[mask][order],
               half.relation[mask][order], half.direction[mask][order])


def hop_csr(half: HalfEdges, relation: int, direction: int) -> CSR:
    return build_csr(half, (half.relation == relation) & (half.direction == direction))


@dataclass(frozen=True, slots=True)
class Walks:
    nodes: torch.Tensor      # [W, L+1]
    facts: torch.Tensor      # [W, L]
    signs: torch.Tensor      # [W, L]
    relations: torch.Tensor  # [W, L]
    directions: torch.Tensor  # [W, L]
    weight: torch.Tensor     # [W] 1 / r of the walk's (anchor, schema)


def _pick(csr: CSR, current: torch.Tensor, alive: torch.Tensor, u: torch.Tensor,
          counts: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the chosen half-edge per walker (uniform, or proportional to `counts`
    over the node's CSR segment) and the updated alive mask."""
    start = csr.ptr[current]
    degree = csr.ptr[current + 1] - start
    if counts is None:
        alive = alive & (degree > 0)
        offset = torch.minimum((u * degree).long(), (degree - 1).clamp_min(0))
        index = start + offset
    else:
        cumulative = torch.cumsum(counts, dim=0)
        before = torch.where(start > 0, cumulative[(start - 1).clamp_min(0)],
                             torch.zeros_like(start))
        end = (start + degree - 1).clamp_min(0)
        total = torch.where(degree > 0, cumulative[end] - before, torch.zeros_like(start))
        alive = alive & (total > 0)
        target = before + torch.minimum((u * total).long(), (total - 1).clamp_min(0))
        index = torch.searchsorted(cumulative, target, right=True)
    if csr.other.numel() == 0:
        return torch.zeros_like(current), torch.zeros_like(alive)
    index = torch.where(alive, index, torch.zeros_like(index)).clamp(max=csr.other.numel() - 1)
    return index, alive


def sample_walks(
    hop_csrs: list[CSR],
    anchors: torch.Tensor,
    walks_per_anchor: int,
    generator: torch.Generator,
    uniform_instance: bool = False,
) -> Walks:
    """`walks_per_anchor` walks from every anchor along `hop_csrs`; keeps accepted (simple)
    walks only. Vectorized over all walkers; one `torch.rand` call per hop."""
    if walks_per_anchor <= 0:
        raise ValueError("walks_per_anchor must be positive.")
    if uniform_instance and len(hop_csrs) > 2:
        raise ValueError("path_sampler: uniform_instance is implemented for schemas with "
                         f"L <= 2 only, got L = {len(hop_csrs)}.")
    current = anchors.long().repeat_interleave(walks_per_anchor)
    alive = torch.ones_like(current, dtype=torch.bool)
    nodes, facts, signs, relations, directions = [current], [], [], [], []
    for j, csr in enumerate(hop_csrs):
        u = torch.rand(current.numel(), generator=generator, dtype=torch.float64)
        counts = None
        if uniform_instance and j == 0 and len(hop_csrs) == 2:
            counts = hop_csrs[1].degree(csr.other)  # completions through each first hop
        index, alive = _pick(csr, current, alive, u.to(current.device), counts)
        current = csr.other[index] if csr.other.numel() else current
        nodes.append(current)
        facts.append(csr.fact[index] if csr.fact.numel() else current)
        signs.append(csr.sign[index] if csr.sign.numel() else current)
        relations.append(csr.relation[index] if csr.relation.numel() else current)
        directions.append(csr.direction[index] if csr.direction.numel() else current)
    node_path = torch.stack(nodes, dim=1)
    ordered, _ = torch.sort(node_path, dim=1)
    simple = ~(ordered[:, 1:] == ordered[:, :-1]).any(dim=1)
    keep = alive & simple
    weight = torch.full((int(keep.sum()),), 1.0 / walks_per_anchor, dtype=torch.float64,
                        device=node_path.device)
    return Walks(node_path[keep], torch.stack(facts, 1)[keep], torch.stack(signs, 1)[keep],
                 torch.stack(relations, 1)[keep], torch.stack(directions, 1)[keep], weight)


def concat_walks(parts: list[Walks]) -> Walks:
    return Walks(*(torch.cat([getattr(w, f) for w in parts]) for f in Walks.__slots__))


def _lex_less(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise lexicographic `a < b` for integer `[W, K]` tensors."""
    differ = a != b
    first = differ.int().argmax(dim=1, keepdim=True)
    return differ.any(dim=1) & (a.gather(1, first) < b.gather(1, first)).squeeze(1)


def _signature(node_type_path: torch.Tensor, relations: torch.Tensor,
               directions: torch.Tensor) -> torch.Tensor:
    """`[W, 3L+1]`: t_0, r_1, d_1 + 1, t_1, ..., r_L, d_L + 1, t_L."""
    length = relations.size(1)
    columns = [node_type_path[:, :1]]
    for j in range(length):
        columns += [relations[:, j:j + 1], directions[:, j:j + 1] + 1,
                    node_type_path[:, j + 1:j + 2]]
    return torch.cat(columns, dim=1)


def canonical_cells(walks: Walks, node_type: torch.Tensor) -> tuple[Walks, torch.Tensor]:
    """Orient every walk canonically; return the oriented walks and their signatures."""
    types = node_type[walks.nodes]
    signature = _signature(types, walks.relations, walks.directions)
    reverse_signature = _signature(types.flip(1), walks.relations.flip(1),
                                   -walks.directions.flip(1))
    flip = _lex_less(reverse_signature, signature)
    palindromic = (signature == reverse_signature).all(dim=1)
    flip = flip | (palindromic & _lex_less(walks.facts.flip(1), walks.facts))
    f = flip.view(-1, 1)
    oriented = Walks(
        nodes=torch.where(f, walks.nodes.flip(1), walks.nodes),
        facts=torch.where(f, walks.facts.flip(1), walks.facts),
        signs=torch.where(f, -walks.signs.flip(1), walks.signs),
        relations=torch.where(f, walks.relations.flip(1), walks.relations),
        directions=torch.where(f, -walks.directions.flip(1), walks.directions),
        weight=walks.weight,
    )
    return oriented, torch.where(f, reverse_signature, signature)


def merge_cells(
    walks: Walks,
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    name_of: Callable[[tuple[int, ...]], tuple[str, tuple[str, ...]]],
    merge: str = "sum",
) -> tuple[FaceGroup, ...]:
    """Merge walks into cells (identity = group + canonical fact tuple, weights summed) and
    build one `FaceGroup` per signature, groups and cells in sorted order.

    `merge="drop_duplicates"` keeps weight 1/r per distinct cell instead of the sum: the
    biased negative control of T8, never used by the model."""
    if merge not in {"sum", "drop_duplicates"}:
        raise ValueError(f"merge must be 'sum' or 'drop_duplicates', got {merge!r}.")
    if walks.nodes.size(0) == 0:
        return ()
    oriented, signature = canonical_cells(walks, node_type)
    groups, group_of = torch.unique(signature, dim=0, return_inverse=True)
    key = torch.cat([group_of.view(-1, 1), oriented.facts], dim=1)
    cells, cell_of = torch.unique(key, dim=0, return_inverse=True)
    weight = torch.zeros(cells.size(0), dtype=torch.float64, device=key.device)
    if merge == "sum":
        weight.index_add_(0, cell_of, oriented.weight)
    else:
        weight.scatter_reduce_(0, cell_of, oriented.weight, reduce="amax", include_self=False)
    first = torch.full((cells.size(0),), oriented.nodes.size(0), dtype=torch.long,
                       device=key.device)
    first.scatter_reduce_(0, cell_of, torch.arange(cell_of.numel(), device=key.device),
                          reduce="amin")
    cell_group = cells[:, 0]
    result: list[FaceGroup] = []
    for g in range(groups.size(0)):  # loop over groups (schemas), not nodes
        rows = first[cell_group == g]
        name, type_names = name_of(tuple(int(v) for v in groups[g].tolist()))
        nodes = oriented.nodes[rows]
        facts = oriented.facts[rows]
        signs = oriented.signs[rows].to(torch.float32)
        relation_index = oriented.relations[rows]
        validate_face_orientation(nodes, facts, edge_index, name, face_edge_sign=signs)
        validate_relation_sequence_constant(relation_index, name)
        result.append(FaceGroup(
            template_name=name,
            node_type_sequence=type_names,
            face_node_index=nodes,
            face_edge_index=facts,
            relation_index=relation_index,
            face_edge_sign=signs,
            path_weight=weight[cell_group == g].to(torch.float32),
        ))
    return tuple(result)


def _signature_parts(signature: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    types = signature[0::3]
    hops = tuple((signature[3 * j + 1], signature[3 * j + 2] - 1) for j in range(len(types) - 1))
    return types, hops


def build_schema_path_groups(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    node_type: torch.Tensor,
    node_type_names: tuple[str, ...],
    schemas: tuple[ResolvedSchema, ...],
    self_inverse: tuple[int, ...],
    tied: bool,
    anchors_of: Callable[[str], torch.Tensor],
    default_walks: int,
    seed: int,
    sampler: str = "walk",
    merge: str = "sum",
) -> tuple[FaceGroup, ...]:
    """Node classification: walks per resolved schema from every node of its start type."""
    if sampler not in {"walk", "uniform_instance"}:
        raise ValueError(f"build_schema_path_groups samples walks, got sampler {sampler!r}.")
    half = build_half_edges(edge_index, edge_type, node_type.numel(), self_inverse, tied)
    generator = torch.Generator().manual_seed(int(seed))
    parts, names = [], {}
    type_id = {name: i for i, name in enumerate(node_type_names)}
    for schema in schemas:
        csrs = [hop_csr(half, relation, direction) for relation, direction in schema.hops]
        walks = sample_walks(csrs, anchors_of(schema.node_types[0]),
                             schema.sample_per_anchor or default_walks, generator,
                             uniform_instance=(sampler == "uniform_instance"))
        parts.append(walks)
        types = tuple(type_id[t] for t in schema.node_types)
        forward = _signature(torch.tensor([types]), torch.tensor([[r for r, _ in schema.hops]]),
                             torch.tensor([[d for _, d in schema.hops]]))
        backward = _signature(torch.tensor([types[::-1]]),
                              torch.tensor([[r for r, _ in schema.hops[::-1]]]),
                              torch.tensor([[-d for _, d in schema.hops[::-1]]]))
        for signature, type_seq in ((forward, schema.node_types),
                                    (backward, schema.node_types[::-1])):
            names.setdefault(tuple(signature[0].tolist()), (schema.name, tuple(type_seq)))
    groups: list[FaceGroup] = []
    for length in sorted({schema.length for schema in schemas}):  # one merge per path length
        same = [w for w, schema in zip(parts, schemas) if schema.length == length]
        groups.extend(merge_cells(concat_walks(same), node_type, edge_index,
                                  lambda s: names[s], merge))
    return tuple(groups)


def build_free_path_groups(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    node_type: torch.Tensor,
    relation_names: tuple[str, ...] | None,
    self_inverse: tuple[int, ...],
    tied: bool,
    walks_per_anchor: int,
    seed: int,
    length: int = 2,
    anchors: torch.Tensor | None = None,
    merge: str = "sum",
) -> tuple[FaceGroup, ...]:
    """Link prediction and graph classification (decision 4): schema-free walks of `length`
    hops, uniform over all incident facts at each hop (stored out-edges only when untied),
    grouped by the realized (node type, relation, direction) sequence."""
    num_nodes = int(node_type.numel())
    half = build_half_edges(edge_index, edge_type, num_nodes, self_inverse, tied)
    csr = build_csr(half, torch.ones_like(half.node, dtype=torch.bool))
    generator = torch.Generator().manual_seed(int(seed))
    if anchors is None:
        anchors = torch.arange(num_nodes, device=node_type.device)
    walks = sample_walks([csr] * length, anchors, walks_per_anchor, generator)

    def name_of(signature: tuple[int, ...]) -> tuple[str, tuple[str, ...]]:
        types, hops = _signature_parts(signature)
        parts = [f"t{types[0]}"]
        for (relation, direction), t in zip(hops, types[1:]):
            rel = relation_names[relation] if relation_names else f"r{relation}"
            parts.append(f"{rel}{'+' if direction > 0 else '-' if direction < 0 else '~'}")
            parts.append(f"t{t}")
        return "->".join(parts), tuple(f"t{t}" for t in types)

    return merge_cells(walks, node_type, edge_index, name_of, merge)
