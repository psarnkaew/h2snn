"""Three-level sheaf energy operator (spec v1.11 §3 Eq. E; v3-B card U1-U2).

States live on nodes `x` (N), facts `s` (E) and path 2-cells `z` (F); `Y = (x, s, z)`. With the
fact maps `F_{v<|e}` (orthogonal, `FactorizedMapOperator` without its scale), the fact weight
`a_e = c_e^2`, the cell weight `beta_f = w_f m_f`, the forced transports `T_{v_i,f} = Q_1 ... Q_i`
(`PathTransportOperator`) and the fact->cell maps `R_{e_i,f} = T_{v_{i-1},f} F_{v_{i-1}<|e_i}^T`
(Thm 1), the energy is

    E(Y) = 1/2 sum_e a_e sum_{v in e} ||F_{v<|e} x_v - s_e||^2                         (node-fact)
         + 1/2 sum_f beta_f [ sum_{e in f} ||R_{e,f} s_e - z_f||^2                       (fact-cell)
                            + sum_{v in f} ||T_{v,f} x_v - z_f||^2 ]                     (node-cell)

and `L = grad^2 E`. Every map is orthogonal, so each diagonal block of `L` is a scalar times `I`:

    D_v = sum_{e ni v} a_e + sum_{f ni v} beta_f,  D_e = 2 a_e + sum_{f ni e} beta_f,
    D_f = beta_f (2 l_f + 1),            L_hat = D^{-1/2} L D^{-1/2}  (every diagonal block I).

Everything is applied matrix-free: the node-fact part through the map operators, the node-cell
part through `transport_sum` / `adjoint_prefix` (D9), the fact-cell part through the new UP / DOWN
recursions of `FactCellTransport` (card §1). Nothing `d x d` is formed outside the tests.

Arms (spec §6) are flags of the same operator: `covering_only` drops the node-cell term,
`hub_A` drops the fact-cell term (and eliminates every fact state), `closure_only` /
`no_path_cells` / `instance_features` have no cells (optionally an extra node-only term such as
`HolonomyEdgeTerm`), `free_cell_maps` replaces `R_{e,f}` by free orthogonal maps per incidence.

`EliminatedOperator` (Prop. 7, 7') applies the Schur complement of `M = P^T P + tau L_hat` over a
set of eliminated fact states (default: facts on <= 1 cell) exactly and matrix-free: the
eliminated `s_e` are filled with their optimal value given `(x, z)` (the fact block of `L_hat` is
`I`) and the full operator is applied; `x*` and `z*` are unchanged (T7).
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, Sequence

import torch

from src.data.structures import FaceGroup
from src.models.map_operators import (
    DenseMapOperator,
    EdgeMapPair,
    FactorizedMapOperator,
    PathTransportOperator,
)
from src.models.vertex_path_laplacian import PathTerm


def _safe_inverse_sqrt(degree: torch.Tensor) -> torch.Tensor:
    """`D^{-1/2}` with 0 where `D = 0` (no inf/NaN in forward or backward)."""
    positive = degree > 0
    safe = torch.where(positive, degree, torch.ones_like(degree))
    return torch.where(positive, safe.rsqrt(), torch.zeros_like(degree))


# ----------------------------------------------------------------------------- incidences
class IncidenceComplex(NamedTuple):
    """Incidence lists of the three-level complex (card U1), cells numbered globally across the
    face groups (group `g` owns cell ids `cell_offsets[g] .. cell_offsets[g+1] - 1`)."""

    num_nodes: int
    num_facts: int
    num_cells: int
    edge_index: torch.Tensor      # [2, E] node-fact incidences: (source role, target role)
    fact_cell: torch.Tensor       # [I_ef, 5] (cell f, hop i, fact e, near node v_{i-1}, far node v_i)
    node_cell: torch.Tensor       # [I_nc, 2] (cell f, node v)
    cells_per_fact: torch.Tensor  # [E] number of cells containing each fact
    cell_length: torch.Tensor     # [F] l_f
    cell_offsets: tuple[int, ...]

    @classmethod
    def from_face_groups(cls, edge_index: torch.Tensor, num_nodes: int,
                         groups: Sequence[FaceGroup]) -> "IncidenceComplex":
        num_facts = int(edge_index.size(1))
        device = edge_index.device
        fact_cell: list[torch.Tensor] = []
        node_cell: list[torch.Tensor] = []
        lengths: list[torch.Tensor] = []
        offsets = [0]
        for group in groups:
            nodes, facts = group.face_node_index, group.face_edge_index
            num_paths, length = facts.shape
            start = offsets[-1]
            cell_ids = torch.arange(start, start + num_paths, device=device)
            hop = torch.arange(length, device=device)
            fact_cell.append(torch.stack([
                cell_ids.unsqueeze(1).expand(-1, length),
                hop.unsqueeze(0).expand(num_paths, -1),
                facts, nodes[:, :-1], nodes[:, 1:],
            ], dim=-1).reshape(-1, 5))
            node_cell.append(torch.stack([
                cell_ids.unsqueeze(1).expand(-1, length + 1), nodes], dim=-1).reshape(-1, 2))
            lengths.append(torch.full((num_paths,), length, dtype=torch.long, device=device))
            offsets.append(start + num_paths)
        empty5 = torch.zeros(0, 5, dtype=torch.long, device=device)
        empty2 = torch.zeros(0, 2, dtype=torch.long, device=device)
        fc = torch.cat(fact_cell) if fact_cell else empty5
        nc = torch.cat(node_cell) if node_cell else empty2
        cell_length = (torch.cat(lengths) if lengths
                       else torch.zeros(0, dtype=torch.long, device=device))
        cells_per_fact = torch.bincount(fc[:, 2], minlength=num_facts) if fc.numel() else \
            torch.zeros(num_facts, dtype=torch.long, device=device)
        return cls(num_nodes=int(num_nodes), num_facts=num_facts, num_cells=offsets[-1],
                   edge_index=edge_index, fact_cell=fc, node_cell=nc,
                   cells_per_fact=cells_per_fact, cell_length=cell_length,
                   cell_offsets=tuple(offsets))

    @property
    def num_path_facts(self) -> int:
        """`|E_P|`: facts on at least one cell."""
        return int((self.cells_per_fact >= 1).sum())

    @property
    def num_shared_facts(self) -> int:
        """`|E_shared|`: facts on at least two cells (keep a state after Prop. 7')."""
        return int((self.cells_per_fact >= 2).sum())


# ----------------------------------------------------------------------------- fact -> cell maps
class FactCellMaps(Protocol):
    """Matrix-free `R_{e_i,f}` of one face group (`[P, L]` incidences)."""

    face_edge_index: torch.Tensor

    def up(self, s: torch.Tensor) -> torch.Tensor: ...

    def down(self, z: torch.Tensor) -> torch.Tensor: ...

    def apply(self, s: torch.Tensor) -> torch.Tensor: ...


class FactCellTransport:
    """`R_{e_i,f} = T_{v_{i-1},f} F_{v_{i-1}<|e_i}^T`, built on the signed steps of a
    `PathTransportOperator` (`Q_i = F_{v_{i-1}<|e_i}^T F_{v_i<|e_i}`, orthogonal under D1a):

        UP   (facts -> cell):  u <- 0; for i = L..1: u <- F_{v_{i-1}<|e_i}^T s_{e_i} + Q_i u
                               gives sum_i R_{e_i,f} s_{e_i}                      (L steps)
        DOWN (cell -> facts):  t <- z_f; for i = 1..L: r_{e_i} <- F_{v_{i-1}<|e_i} t; t <- Q_i^T t
                               gives r_{e_i} = R_{e_i,f}^T z_f                     (L steps)

    `F_{v_{i-1}<|e_i}` is the map on the near node's role of hop `i`, i.e. the source role on a
    +1 hop and the target role on a -1 hop (`PathTransportOperator._src[i-1]` after row mixing).
    """

    def __init__(self, transport: PathTransportOperator, face_edge_index: torch.Tensor) -> None:
        if not transport.shared_scale:
            raise ValueError("FactCellTransport needs orthogonal fact maps (shared scale, D1a).")
        if face_edge_index.shape != (transport.face_node_index.size(0), transport.length):
            raise ValueError("face_edge_index must be [P, L] for the transport's [P, L+1] nodes.")
        self.transport = transport
        self.face_edge_index = face_edge_index
        self.length = transport.length

    def _near(self, j: int) -> FactorizedMapOperator | DenseMapOperator:
        return self.transport._src[j]

    def up(self, s: torch.Tensor) -> torch.Tensor:
        """`[E, d]` fact states -> `[P, d]` `sum_i R_{e_i,f} s_{e_i}` (Horner, L steps)."""
        values = s[self.face_edge_index]  # [P, L, d]
        total = self._near(self.length - 1).apply_orthogonal_T(values[:, self.length - 1])
        for j in range(self.length - 2, -1, -1):
            total = self._near(j).apply_orthogonal_T(values[:, j]) + self.transport._step(j, total)
        return total

    def down(self, z: torch.Tensor) -> torch.Tensor:
        """`[P, d]` cell states -> `[P, L, d]` `r_{f,i} = R_{e_i,f}^T z_f` (L steps)."""
        t = z
        rows = []
        for j in range(self.length):
            rows.append(self._near(j).apply_orthogonal(t))
            if j < self.length - 1:
                t = self.transport._step_T(j, t)
        return torch.stack(rows, dim=1)

    def apply(self, s: torch.Tensor) -> torch.Tensor:
        """`[E, d]` -> `[P, L, d]` per-incidence values `R_{e_i,f} s_{e_i}` (tests; L(L+1)/2 steps)."""
        values = s[self.face_edge_index]
        columns = [self._near(j).apply_orthogonal_T(values[:, j]) for j in range(self.length)]
        values = torch.stack(columns, dim=1)  # column j needs T_{v_j} = Q_1 ... Q_j
        for j in range(self.length - 1, 0, -1):
            tail = self.transport._step(j - 1, values[:, j:])
            values = torch.cat([values[:, :j], tail], dim=1)
        return values


class FreeFactCellMaps:
    """Arm `free_cell_maps` (Thm 3 control): one free orthogonal map per fact-cell incidence,
    row `p * L + i` of `maps` for hop `i` of cell `p`; not forced by functoriality."""

    def __init__(self, maps: FactorizedMapOperator, face_edge_index: torch.Tensor) -> None:
        if maps.num_edges != face_edge_index.numel():
            raise ValueError("free fact->cell maps need one row per fact-cell incidence.")
        self.maps = maps
        self.face_edge_index = face_edge_index
        self.length = int(face_edge_index.size(1))

    def apply(self, s: torch.Tensor) -> torch.Tensor:
        num_paths = self.face_edge_index.size(0)
        values = s[self.face_edge_index].reshape(num_paths * self.length, s.size(-1))
        return self.maps.apply_orthogonal(values).view(num_paths, self.length, s.size(-1))

    def up(self, s: torch.Tensor) -> torch.Tensor:
        return self.apply(s).sum(dim=1)

    def down(self, z: torch.Tensor) -> torch.Tensor:
        num_paths = z.size(0)
        expanded = z.unsqueeze(1).expand(num_paths, self.length, z.size(-1)).reshape(-1, z.size(-1))
        return self.maps.apply_orthogonal_T(expanded).view(num_paths, self.length, z.size(-1))


class CellGroup(NamedTuple):
    """One face group of the complex: its forced node->cell transports, fact->cell maps and
    weights `beta_f = w_f m_f` (`[P]`)."""

    face_node_index: torch.Tensor
    face_edge_index: torch.Tensor
    transport: PathTransportOperator
    fact_maps: FactCellMaps
    beta: torch.Tensor

    @property
    def num_cells(self) -> int:
        return int(self.face_node_index.size(0))

    @property
    def length(self) -> int:
        return int(self.face_edge_index.size(1))


def build_cell_group(maps: EdgeMapPair, group: FaceGroup, edge_index: torch.Tensor,
                     beta: torch.Tensor, free_maps: FactorizedMapOperator | None = None
                     ) -> CellGroup:
    """Forced transports of `group` from the fact maps (`T`, and `R` unless `free_maps`)."""
    transport = PathTransportOperator(
        maps.src, maps.tgt, group.face_node_index, group.face_edge_index, edge_index,
        shared_scale=True, face_edge_sign=group.face_edge_sign,
    )
    fact_maps: FactCellMaps = (FactCellTransport(transport, group.face_edge_index)
                               if free_maps is None
                               else FreeFactCellMaps(free_maps, group.face_edge_index))
    return CellGroup(group.face_node_index, group.face_edge_index, transport, fact_maps, beta)


# ----------------------------------------------------------------------------- the operator
class StateLayout(NamedTuple):
    """Row offsets of the stacked state `Y = [x; s_kept; z]`."""

    num_nodes: int
    num_fact_states: int
    num_cells: int

    @property
    def rows(self) -> int:
        return self.num_nodes + self.num_fact_states + self.num_cells

    def split(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, e = self.num_nodes, self.num_fact_states
        return y[:n], y[n:n + e], y[n + e:]

    def join(self, x: torch.Tensor, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, s, z], dim=0)


class SheafEnergyOperator:
    """`L`, `D` and `L_hat = D^{-1/2} L D^{-1/2}` of the three-level energy, matrix-free.

    Args:
        maps: fact maps of the observed facts, one shared scale per fact (D1a); the weight is
            `a_e = scale^2` and the maps act through their orthogonal factor.
        edge_index: `[2, E]` facts (source role at row 0, target role at row 1).
        num_nodes: `N`.
        groups: the cells by face group (may be empty: no cell term).
        node_cell: keep the node-cell term (False = arm `covering_only`).
        fact_cell: keep the fact-cell term (False = arm `hub_A`).
        node_terms: extra node-only terms with `apply(x, N)` and `diagonal(N)` (the holonomy edges
            of `closure_only`); their diagonal enters `D_v`.
    """

    def __init__(
        self,
        maps: EdgeMapPair,
        edge_index: torch.Tensor,
        num_nodes: int,
        groups: Sequence[CellGroup],
        node_cell: bool = True,
        fact_cell: bool = True,
        node_terms: Sequence[PathTerm] = (),
    ) -> None:
        if not torch.equal(maps.src.scale, maps.tgt.scale):
            raise ValueError("the energy needs one scale per fact shared by both roles (D1a).")
        self.maps = maps
        self.src_idx = edge_index[0]
        self.tgt_idx = edge_index[1]
        self.num_nodes = int(num_nodes)
        self.num_facts = int(edge_index.size(1))
        self.groups = tuple(groups)
        self.node_cell = node_cell
        self.fact_cell = fact_cell
        self.node_terms = tuple(node_terms)
        self.a = maps.src.scale.square()  # a_e = c_e^2
        self.num_cells = sum(g.num_cells for g in self.groups)
        self.d = maps.src.d
        self.layout = StateLayout(self.num_nodes, self.num_facts, self.num_cells)
        self._degree: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._inverse_sqrt: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------ bookkeeping
    @property
    def beta(self) -> torch.Tensor:
        """`[F]` cell weights in global cell order."""
        if not self.groups:
            return self.a.new_zeros(0)
        return torch.cat([g.beta for g in self.groups])

    @property
    def cell_length(self) -> torch.Tensor:
        if not self.groups:
            return torch.zeros(0, dtype=torch.long, device=self.a.device)
        return torch.cat([torch.full((g.num_cells,), g.length, dtype=torch.long,
                                     device=self.a.device) for g in self.groups])

    def _cell_slices(self) -> list[slice]:
        slices, start = [], 0
        for g in self.groups:
            slices.append(slice(start, start + g.num_cells))
            start += g.num_cells
        return slices

    def cells_per_fact(self) -> torch.Tensor:
        """`[E]` number of cells containing each fact."""
        counts = torch.zeros(self.num_facts, dtype=torch.long, device=self.a.device)
        for g in self.groups:
            counts.index_add_(0, g.face_edge_index.reshape(-1),
                              torch.ones(g.face_edge_index.numel(), dtype=torch.long,
                                         device=self.a.device))
        return counts

    # ------------------------------------------------------------------ degree
    def degree(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(D_v [N], D_e [E], D_f [F])`, the scalar diagonal of `L` (self-loops count both roles)."""
        if self._degree is None:
            a = self.a
            d_v = a.new_zeros(self.num_nodes)
            d_v.index_add_(0, self.src_idx, a)
            d_v.index_add_(0, self.tgt_idx, a)
            d_e = 2.0 * a
            d_f_parts = []
            for g in self.groups:
                beta = g.beta
                if self.node_cell:
                    d_v.index_add_(0, g.face_node_index.reshape(-1),
                                   beta.unsqueeze(1).expand(-1, g.length + 1).reshape(-1))
                if self.fact_cell:
                    d_e = d_e.index_add(0, g.face_edge_index.reshape(-1),
                                        beta.unsqueeze(1).expand(-1, g.length).reshape(-1))
                terms = (g.length if self.fact_cell else 0) + (g.length + 1 if self.node_cell else 0)
                d_f_parts.append(beta * float(terms))
            for term in self.node_terms:
                d_v = d_v + term.diagonal(self.num_nodes)
            d_f = torch.cat(d_f_parts) if d_f_parts else a.new_zeros(0)
            self._degree = (d_v, d_e, d_f)
        return self._degree

    def inverse_sqrt_degree(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._inverse_sqrt is None:
            self._inverse_sqrt = tuple(_safe_inverse_sqrt(d) for d in self.degree())  # type: ignore[assignment]
        return self._inverse_sqrt  # type: ignore[return-value]

    # ------------------------------------------------------------------ pieces of L
    def _facts_from_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """`[E, d]` `a_e (F_u x_u + F_v x_v)`."""
        gathered = (self.maps.src.apply_orthogonal(x[self.src_idx])
                    + self.maps.tgt.apply_orthogonal(x[self.tgt_idx]))
        return self.a.unsqueeze(-1) * gathered

    def _facts_from_cells(self, z: torch.Tensor, out: torch.Tensor) -> None:
        """`out += sum_{f ni e} beta_f R_{e,f}^T z_f` (DOWN), in place."""
        if not self.fact_cell:
            return
        for g, rows in zip(self.groups, self._cell_slices()):
            r = g.fact_maps.down(g.beta.unsqueeze(-1) * z[rows])  # [P, L, d]
            out.index_add_(0, g.face_edge_index.reshape(-1), r.reshape(-1, r.size(-1)))

    def _nodes_from_facts(self, s: torch.Tensor, out: torch.Tensor) -> None:
        """`out += sum_{e ni v} a_e F_{v<|e}^T s_e`, in place."""
        weighted = self.a.unsqueeze(-1) * s
        out.index_add_(0, self.src_idx, self.maps.src.apply_orthogonal_T(weighted))
        out.index_add_(0, self.tgt_idx, self.maps.tgt.apply_orthogonal_T(weighted))

    def _nodes_from_cells(self, z: torch.Tensor, out: torch.Tensor) -> None:
        """`out += sum_{f ni v} beta_f T_{v,f}^T z_f` (adjoint prefix), in place."""
        if not self.node_cell:
            return
        for g, rows in zip(self.groups, self._cell_slices()):
            pulled = g.transport.adjoint_prefix(g.beta.unsqueeze(-1) * z[rows])  # [P, L+1, d]
            out.index_add_(0, g.face_node_index.reshape(-1), pulled.reshape(-1, pulled.size(-1)))

    def _cells_from_nodes_and_facts(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """`[F, d]` `beta_f (sum_e R_{e,f} s_e + sum_v T_{v,f} x_v)` (UP and Horner)."""
        parts = []
        for g in self.groups:
            total = x.new_zeros((g.num_cells, x.size(-1)))
            if self.fact_cell:
                total = total + g.fact_maps.up(s)
            if self.node_cell:
                total = total + g.transport.transport_sum(x)
            parts.append(g.beta.unsqueeze(-1) * total)
        return torch.cat(parts) if parts else x.new_zeros((0, x.size(-1)))

    def off_diagonal(self, x: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
                     o_s: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(O_x, O_s, O_z)` with `L Y = D Y - O(Y) + sum node_terms(x)` (the sheaf couplings).
        `o_s` may pass the precomputed fact gather `a_e (F_u x_u + F_v x_v) + sum_f beta_f R^T z_f`
        (the elimination computes it first)."""
        o_x = torch.zeros_like(x)
        self._nodes_from_facts(s, o_x)
        self._nodes_from_cells(z, o_x)
        if o_s is None:
            o_s = self._facts_from_nodes(x).clone()
            self._facts_from_cells(z, o_s)
        o_z = self._cells_from_nodes_and_facts(x, s)
        return o_x, o_s, o_z

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        """Unnormalized `L Y` on the stacked full state `[x; s; z]`."""
        x, s, z = self.layout.split(y)
        d_v, d_e, d_f = self.degree()
        o_x, o_s, o_z = self.off_diagonal(x, s, z)
        l_x = d_v.unsqueeze(-1) * x - o_x
        for term in self.node_terms:
            l_x = l_x + term.apply(x, self.num_nodes)
        l_s = d_e.unsqueeze(-1) * s - o_s
        l_z = d_f.unsqueeze(-1) * z - o_z
        return self.layout.join(l_x, l_s, l_z)

    def apply_normalized(self, y: torch.Tensor) -> torch.Tensor:
        """`L_hat Y = D^{-1/2} L D^{-1/2} Y`; rows (and columns) with `D = 0` are zero."""
        scale = torch.cat(self.inverse_sqrt_degree()).unsqueeze(-1)
        return scale * self.apply(scale * y)

    # ------------------------------------------------------------------ initial guess
    def initial_guess(self, h0: torch.Tensor) -> torch.Tensor:
        """D-B7: `x = h0`, `s_e = 1/2 (F_u h_u + F_v h_v)`, `z_f` = mean of the cell's incident
        values `R_{e,f} s_e` and `T_{v,f} h_v` over the terms present (one UP)."""
        s0 = 0.5 * (self.maps.src.apply_orthogonal(h0[self.src_idx])
                    + self.maps.tgt.apply_orthogonal(h0[self.tgt_idx]))
        parts = []
        for g in self.groups:
            total = h0.new_zeros((g.num_cells, h0.size(-1)))
            count = 0
            if self.fact_cell:
                total = total + g.fact_maps.up(s0)
                count += g.length
            if self.node_cell:
                total = total + g.transport.transport_sum(h0)
                count += g.length + 1
            parts.append(total / max(count, 1))
        z0 = torch.cat(parts) if parts else h0.new_zeros((0, h0.size(-1)))
        return self.layout.join(h0, s0, z0)


class EliminatedOperator:
    """Reduced `L_hat` on `(x, s_kept, z)` after eliminating the fact states `E_elim` exactly
    (Prop. 7 for facts on no cell, Prop. 7' for facts on one cell; the same Schur complement is
    exact for any set). Default `kept` = facts on >= 2 cells (`E_shared`).

    `M_red = A - B (tau I)^{-1} B^T` applied to `Y_red` equals `M (Y_red, s_elim*)` restricted to
    the kept rows, with `s_elim* = -(L_hat (Y_red, s_elim = 0))_elim` the optimal eliminated
    states (the fact block of `L_hat` is `I`), so the reduced operator is the full one with the
    eliminated states filled in; one extra fact gather per application.
    """

    def __init__(self, operator: SheafEnergyOperator, kept: torch.Tensor | None = None) -> None:
        self.operator = operator
        counts = operator.cells_per_fact()
        if kept is None:
            kept = (counts >= 2).nonzero(as_tuple=True)[0]
        self.kept = kept
        mask = torch.ones(operator.num_facts, dtype=torch.bool, device=kept.device)
        mask[kept] = False
        self.eliminated = mask.nonzero(as_tuple=True)[0]
        self.layout = StateLayout(operator.num_nodes, int(kept.numel()), operator.num_cells)
        self.num_path_facts = int((counts >= 1).sum())
        self.num_shared_facts = int((counts >= 2).sum())

    @property
    def num_nodes(self) -> int:
        return self.operator.num_nodes

    def statistics(self) -> dict[str, float]:
        e = self.operator.num_facts
        return {
            "num_facts": float(e),
            "num_path_facts": float(self.num_path_facts),
            "num_shared_facts": float(self.num_shared_facts),
            "num_fact_states": float(self.layout.num_fact_states),
            "num_cells": float(self.operator.num_cells),
            "eliminated_share": 1.0 - self.layout.num_fact_states / e if e else 0.0,
        }

    def _scales(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.operator.inverse_sqrt_degree()

    def _fact_gather(self, x_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """`[E, d]` `a_e (F_u x~_u + F_v x~_v) + sum_{f ni e} beta_f R_{e,f}^T z~_f` on the scaled
        states `x~ = D^{-1/2} x`, `z~ = D^{-1/2} z`; the optimal eliminated state is
        `s_e* = D_e^{-1/2} gather_e` (and `s~_e* = gather_e / D_e`)."""
        gather = self.operator._facts_from_nodes(x_t).clone()
        self.operator._facts_from_cells(z_t, gather)
        return gather

    def apply_normalized(self, y: torch.Tensor) -> torch.Tensor:
        """Reduced `L_hat Y_red`."""
        op = self.operator
        x, s_kept, z = self.layout.split(y)
        inv_v, inv_e, inv_f = self._scales()
        x_t = inv_v.unsqueeze(-1) * x
        z_t = inv_f.unsqueeze(-1) * z
        gather = self._fact_gather(x_t, z_t)
        s_t = inv_e.square().unsqueeze(-1) * gather                # s~* on the eliminated rows
        if self.kept.numel():
            s_t = s_t.index_copy(0, self.kept, inv_e[self.kept].unsqueeze(-1) * s_kept)
        d_v, d_e, d_f = op.degree()
        o_x, o_s, o_z = op.off_diagonal(x_t, s_t, z_t, o_s=gather)
        l_x = d_v.unsqueeze(-1) * x_t - o_x
        for term in op.node_terms:
            l_x = l_x + term.apply(x_t, op.num_nodes)
        l_s = (d_e.unsqueeze(-1) * s_t - o_s)[self.kept]
        l_z = d_f.unsqueeze(-1) * z_t - o_z
        return self.layout.join(inv_v.unsqueeze(-1) * l_x,
                                inv_e[self.kept].unsqueeze(-1) * l_s,
                                inv_f.unsqueeze(-1) * l_z)

    def diagonal_scalars(self) -> torch.Tensor:
        """`[rows]` scalar diagonal of the reduced `L_hat`: 1 minus the Schur corrections
        `sum_{e in E_elim} w_{p,e}^2`, `w_{v,e} = a_e / sqrt(D_v D_e)`, `w_{f,e} = beta_f / sqrt(D_f D_e)`
        (the cross term of an eliminated self-loop fact, `2 w_u w_v F_u^T F_v`, is not scalar and
        is omitted; it is zero when no eliminated fact is a self-loop)."""
        op = self.operator
        inv_v, inv_e, inv_f = self._scales()
        a, beta = op.a, op.beta
        elim = torch.zeros(op.num_facts, dtype=a.dtype, device=a.device)
        elim[self.eliminated] = 1.0
        w2_e = (a * inv_e).square() * elim                        # a_e^2 / D_e per eliminated fact
        corr_v = torch.zeros(op.num_nodes, dtype=a.dtype, device=a.device)
        corr_v.index_add_(0, op.src_idx, w2_e)
        corr_v.index_add_(0, op.tgt_idx, w2_e)
        corr_v = corr_v * inv_v.square()
        corr_f_parts = []
        for g in op.groups:
            if op.fact_cell:
                e_terms = (inv_e.square() * elim)[g.face_edge_index].sum(dim=1)  # sum_e 1/D_e
                corr_f_parts.append(g.beta.square() * e_terms)
            else:
                corr_f_parts.append(torch.zeros_like(g.beta))
        corr_f = (torch.cat(corr_f_parts) if corr_f_parts else a.new_zeros(0)) * inv_f.square()
        d_v, d_e, d_f = op.degree()
        ones_v = (d_v > 0).to(a.dtype)
        ones_f = (d_f > 0).to(a.dtype)
        return torch.cat([ones_v - corr_v, torch.ones_like(self.kept, dtype=a.dtype), ones_f - corr_f])

    def initial_guess(self, h0: torch.Tensor) -> torch.Tensor:
        x0, s0, z0 = self.operator.layout.split(self.operator.initial_guess(h0))
        return self.layout.join(x0, s0[self.kept], z0)

    def fact_states(self, y: torch.Tensor) -> torch.Tensor:
        """Every fact state `s*` of a reduced solution: the kept ones as given, the eliminated ones
        at their optimum `D_e^{-1/2} gather_e` given `(x, z)`."""
        x, s_kept, z = self.layout.split(y)
        inv_v, inv_e, inv_f = self._scales()
        gather = self._fact_gather(inv_v.unsqueeze(-1) * x, inv_f.unsqueeze(-1) * z)
        s_full = inv_e.unsqueeze(-1) * gather
        if self.kept.numel():
            s_full = s_full.index_copy(0, self.kept, s_kept)
        return s_full


def materialize(apply, rows: int, d: int, dtype: torch.dtype, device: torch.device | None = None
                ) -> torch.Tensor:
    """Dense `[rows*d, rows*d]` matrix of a linear map on `[rows, d]` states, column by column
    (tests only)."""
    columns = []
    for row in range(rows):
        for k in range(d):
            e = torch.zeros(rows, d, dtype=dtype, device=device)
            e[row, k] = 1.0
            columns.append(apply(e).reshape(-1))
    return torch.stack(columns, dim=1)


__all__ = ["CellGroup", "EliminatedOperator", "FactCellTransport", "FreeFactCellMaps",
           "IncidenceComplex", "SheafEnergyOperator", "StateLayout", "build_cell_group",
           "materialize"]
