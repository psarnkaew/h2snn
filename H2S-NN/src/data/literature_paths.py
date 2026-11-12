"""Cells of the fixed-literature-schema row (spec v1.19 §6; card Part O O3; a comparison row, not
the method).

The main model with one change, the path source: instead of WISP's schema-free walks, cells are
drawn only along the listed meta-paths (`LITERATURE_METAPATHS`, HGB's published table, paths that
start at the target type). Budget (O-2): from every target node, the tuned r walks per anchor split
evenly over the listed paths (`literature_walk_split`, at least one each, summing to r). Within a
meta-path a walk follows D8's stated distribution (`path_sampler.sample_walks`): at each hop a
uniform fact of the hop's (relation, direction) at the current node; walks that revisit a node are
rejected; each accepted walk of path pi carries 1 / r_pi, so a path's cells estimate its sum over
anchors and instances as in D8. Cells use the WISP cell format (one group per length, D8 identity,
per-cell signature id, `wisp_sampler.merge_length_group`), so `g` reads each cell exactly as it
reads a WISP cell (relation embeddings and hop directions per cell).
"""

from __future__ import annotations

import torch

from src.configs.config import literature_walk_split
from src.data.path_sampler import Walks, build_half_edges, concat_walks, hop_csr, sample_walks
from src.data.schemas import ResolvedSchema
from src.data.structures import FaceGroup
from src.data.wisp_sampler import merge_length_group


def literature_walk_plan(schemas: tuple[ResolvedSchema, ...], walks_per_anchor: int
                         ) -> tuple[tuple[ResolvedSchema, int], ...]:
    """(listed path, walks per anchor) in table order; the walks sum to r."""
    return tuple(zip(schemas, literature_walk_split(walks_per_anchor, len(schemas))))


def build_literature_path_groups(edge_index: torch.Tensor, edge_type: torch.Tensor,
                                 node_type: torch.Tensor, node_type_names: tuple[str, ...],
                                 schemas: tuple[ResolvedSchema, ...],
                                 self_inverse: tuple[int, ...], target_node_type: str,
                                 anchors: torch.Tensor, walks_per_anchor: int,
                                 seed: int) -> tuple[FaceGroup, ...]:
    """One draw of the row (seed `seed`): walks of every listed path from every anchor, merged
    per length into WISP-format groups (`literature_L<length>`)."""
    for schema in schemas:
        if schema.node_types[0] != target_node_type:
            raise ValueError(f"listed meta-path {schema.name!r} starts at {schema.node_types[0]!r}, "
                             f"not at the target type {target_node_type!r} (spec v1.19 §6).")
    type_ids = {name: i for i, name in enumerate(node_type_names)}
    if any(int(t) != type_ids[target_node_type] for t in node_type[anchors].unique()):
        raise ValueError("every anchor of the row is a node of the target type.")
    half = build_half_edges(edge_index, edge_type, int(node_type.numel()), self_inverse,
                            backward=True)
    generator = torch.Generator().manual_seed(int(seed))
    by_length: dict[int, list[Walks]] = {}
    for schema, walks in literature_walk_plan(schemas, walks_per_anchor):
        csrs = [hop_csr(half, relation, direction) for relation, direction in schema.hops]
        by_length.setdefault(schema.length, []).append(
            sample_walks(csrs, anchors, walks, generator))
    groups: list[FaceGroup] = []
    for length in sorted(by_length):
        walks = concat_walks(by_length[length])
        if walks.nodes.size(0) == 0:
            continue
        groups.append(merge_length_group(walks, node_type, edge_index, f"literature_L{length}"))
    return tuple(groups)


__all__ = ["build_literature_path_groups", "literature_walk_plan"]
