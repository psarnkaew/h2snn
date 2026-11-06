"""Fact layer (v2 decision D7, plan §9.1): one definition, computed from the data.

    rho(r) = r'  iff  E(r') = {(v, u) : (u, v) in E(r)}      exact set equality; r' = r allowed
    fact   = (u, r, v) == (v, rho(r), u);  one fact = one oriented 1-cell, stored in the
             orientation of its lower relation id (ties, i.e. rho(r) = r: lower (u, v))
    roles  = a source and a target role per relation; they coincide when rho(r) = r

Consequences (not separate rules): a mirror pair (r, r') keeps r's edges and drops r';
a self-inverse relation keeps one cell per unordered pair (`src <= dst`, self-loops once);
reciprocal edges of a relation with rho(r) != r stay two facts; an edge without a partner
is its own fact. There is no threshold and no config key for rho. `fact_layer: untied`
(the ablation control, never tuned) keeps every stored edge and every relation, with
separate roles everywhere: the pre-D7 behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FACT_LAYER_MODES = ("tied", "untied")


@dataclass(frozen=True, slots=True)
class FactLayer:
    """`rho` and the relation layout after tying.

    Attributes:
        mode: "tied" or "untied".
        relation_names: names of the stored (original) relations, by original id.
        rho: per original relation, the original id of its exact inverse, or None.
        kept: original ids of the relations that carry facts, ascending; the compact id of a
            kept relation is its position here (all relations when untied).
        self_inverse: compact ids of kept relations with rho(r) = r (shared roles); empty
            when untied.
        unmatched_edges: per original relation, its distinct stored edges when rho(r) is
            undefined, else 0.
        facts_before: stored edges.
        facts_after: edges kept as facts.
    """

    mode: str
    relation_names: tuple[str, ...]
    rho: tuple[int | None, ...]
    kept: tuple[int, ...]
    self_inverse: tuple[int, ...]
    unmatched_edges: tuple[int, ...]
    facts_before: int
    facts_after: int

    @property
    def kept_names(self) -> tuple[str, ...]:
        return tuple(self.relation_names[r] for r in self.kept)

    def resolve(self, relation_name: str) -> tuple[int, int]:
        """`(compact id, direction)` of an original relation name: +1 when the name is
        kept, -1 when it is the dropped inverse of a kept relation. A self-inverse relation
        resolves to +1 (its two directions are the same facts)."""
        if relation_name not in self.relation_names:
            raise KeyError(f"unknown relation {relation_name!r}; known: {self.relation_names}.")
        original = self.relation_names.index(relation_name)
        if original in self.kept:
            return self.kept.index(original), 1
        partner = self.rho[original]
        if partner is None or partner not in self.kept:  # pragma: no cover - by construction
            raise RuntimeError(f"relation {relation_name!r} is neither kept nor tied.")
        return self.kept.index(partner), -1

    def summary(self) -> dict[str, object]:
        """Loggable record: rho by name, kept relations, unmatched edges, fact counts."""
        return {
            "fact_layer": self.mode,
            "rho": {self.relation_names[r]: (None if p is None else self.relation_names[p])
                    for r, p in enumerate(self.rho)},
            "kept_relations": list(self.kept_names),
            "self_inverse_relations": [self.relation_names[self.kept[c]]
                                       for c in self.self_inverse],
            "unmatched_edges": {self.relation_names[r]: n
                                for r, n in enumerate(self.unmatched_edges)},
            "facts_before": self.facts_before,
            "facts_after": self.facts_after,
        }


def _edge_keys(src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return src.long() * int(num_nodes) + dst.long()


def exact_inverse_relations(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_relations: int,
    num_nodes: int,
) -> tuple[int | None, ...]:
    """`rho` from exact equality of edge sets: `rho(r) = r'` iff the set of stored edges of
    `r'` equals the reversed set of `r` (r' = r allowed). A relation without stored edges
    has no facts and gets None. Two exact partners would make rho ill-defined: error."""
    forward, backward = [], []
    for r in range(num_relations):
        mask = edge_type == r
        src, dst = edge_index[0, mask], edge_index[1, mask]
        forward.append(torch.unique(_edge_keys(src, dst, num_nodes)))
        backward.append(torch.unique(_edge_keys(dst, src, num_nodes)))
    rho: list[int | None] = []
    for r in range(num_relations):
        if forward[r].numel() == 0:
            rho.append(None)
            continue
        partners = [p for p in range(num_relations)
                    if forward[p].numel() == backward[r].numel()
                    and torch.equal(forward[p], backward[r])]
        if len(partners) > 1:
            raise ValueError(
                f"relation {r} has {len(partners)} exact inverses {partners}: two relations "
                "store identical edge sets, so rho is not a function."
            )
        rho.append(partners[0] if partners else None)
    for r, p in enumerate(rho):  # equality of sets is symmetric; checked, not assumed
        if p is not None and rho[p] != r:  # pragma: no cover - mathematical invariant
            raise RuntimeError(f"rho is not an involution at relation {r}.")
    return tuple(rho)


def build_fact_layer(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    relation_names: tuple[str, ...] | list[str],
    num_nodes: int,
    mode: str,
) -> FactLayer:
    """Compute `rho` on the stored edges and the relation layout for `mode`."""
    if mode not in FACT_LAYER_MODES:
        raise ValueError(f"fact_layer must be one of {FACT_LAYER_MODES}, got {mode!r}.")
    names = tuple(relation_names)
    num_relations = len(names)
    rho = exact_inverse_relations(edge_index, edge_type, num_relations, num_nodes)
    unmatched = []
    for r in range(num_relations):
        if rho[r] is None:
            mask = edge_type == r
            keys = _edge_keys(edge_index[0, mask], edge_index[1, mask], num_nodes)
            unmatched.append(int(torch.unique(keys).numel()))
        else:
            unmatched.append(0)
    if mode == "untied":
        kept = tuple(range(num_relations))
        self_inverse: tuple[int, ...] = ()
    else:
        kept = tuple(r for r in range(num_relations) if rho[r] is None or rho[r] >= r)
        self_inverse = tuple(c for c, r in enumerate(kept) if rho[r] == r)
    layer = FactLayer(
        mode=mode, relation_names=names, rho=rho, kept=kept, self_inverse=self_inverse,
        unmatched_edges=tuple(unmatched), facts_before=int(edge_index.size(1)), facts_after=0,
    )
    facts_after = int(fact_mask(layer, edge_index, edge_type).sum())
    return FactLayer(
        mode=mode, relation_names=names, rho=rho, kept=kept, self_inverse=self_inverse,
        unmatched_edges=tuple(unmatched), facts_before=int(edge_index.size(1)),
        facts_after=facts_after,
    )


def fact_mask(layer: FactLayer, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
    """`[E]` bool: which stored edges are kept as facts (vectorized, no loop over edges)."""
    num_relations = len(layer.relation_names)
    kept = torch.zeros(num_relations, dtype=torch.bool, device=edge_type.device)
    kept[list(layer.kept)] = True
    keep = kept[edge_type]
    if layer.self_inverse:
        shared = torch.zeros(num_relations, dtype=torch.bool, device=edge_type.device)
        shared[[layer.kept[c] for c in layer.self_inverse]] = True
        keep = keep & (~shared[edge_type] | (edge_index[0] <= edge_index[1]))
    return keep


def apply_fact_layer(
    layer: FactLayer,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the facts (stored order preserved) and re-index relations to compact ids."""
    keep = fact_mask(layer, edge_index, edge_type)
    compact = torch.full((len(layer.relation_names),), -1, dtype=torch.long,
                         device=edge_type.device)
    compact[list(layer.kept)] = torch.arange(len(layer.kept), device=edge_type.device)
    return edge_index[:, keep], compact[edge_type[keep]]
