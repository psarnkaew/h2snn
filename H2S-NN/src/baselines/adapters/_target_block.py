"""Shared structure for the target-centric link-prediction baselines.

HAN, Hetero2Net and H2SGNN are all *target-centric*: they embed one node type and
propagate through meta-paths that return to it. Link prediction needs two adjustments none
of them ships, and both are the same for every one of them, so they live here rather than
being re-derived for each (LatGRL, removed in spec v1.15, used it too).

**The block.** For node classification the target is the classified node type. Here it is
the set of node types the target relation connects -- ``user`` and ``artist`` on LastFM,
``disease`` alone on PubMed. HGB lays node types out in contiguous blocks and on all four
link-prediction datasets that set is a single contiguous range, so a half-open ``[lo, hi)``
describes it exactly. :func:`target_node_range` refuses anything else rather than quietly
dragging in unrelated types.

**Featureless blocks.** LastFM and YouTube carry no node attributes at all. The vendors'
convention is ``torch.eye`` plus a bias-free ``nn.Linear``, which is fine while the
projection accepts a sparse input -- but HAN's ``GATConv`` and the transformer baselines
need a dense matrix, and LastFM's block is 19,524 nodes, so a dense identity is 1.5 GB.
:class:`TargetBlockInput` uses an ``nn.Embedding`` there instead. That is the same function
(identity times a weight matrix is a lookup) at a fraction of the memory, and it is what
H2SNN itself does for featureless inputs.
"""

from __future__ import annotations

import torch
from torch import nn

from src.baselines.bundles import LinkBundle


def target_node_range(bundle: LinkBundle) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` covering every node type the target relation connects."""

    data = bundle.data
    slices = dict(data.node_slices)
    source_type, _, destination_type = data.target_relation
    endpoints = sorted({source_type, destination_type})

    low = min(slices[name][0] for name in endpoints)
    high = max(slices[name][1] for name in endpoints)
    covered = sum(slices[name][1] - slices[name][0] for name in endpoints)
    if high - low != covered:
        raise NotImplementedError(
            f"{bundle.dataset}: target node types {endpoints} are not contiguous in the "
            "flattened layout, so a single-range block crop would include other types"
        )
    return low, high


def target_block_feature_width(bundle: LinkBundle) -> int:
    """Attribute width shared by the block's node types, or 0 if it has none.

    Returns 0 when the types disagree on width as well: a block mixing a 200-dim type
    with a featureless one has no single matrix to project, so it takes the lookup path.
    """

    data = bundle.data
    widths = dict(data.feature_dims)
    source_type, _, destination_type = data.target_relation
    distinct = {widths[name] for name in {source_type, destination_type}}
    if len(distinct) != 1:
        return 0
    return distinct.pop()


class TargetBlockInput(nn.Module):
    """Dense input matrix for a target-centric encoder.

    Every one of these models already owns its first projection -- HAN's ``GATConv``,
    H2SGNN's ``feat_project``, Hetero2Net's encoder -- so this deliberately adds none.
    It hands back the block's real attributes when it has them, and a learned lookup of
    width ``embedding_dim`` when it does not. :attr:`output_dim` is the width the encoder
    should be constructed against, and :attr:`mode` goes on the run record.
    """

    def __init__(
        self, bundle: LinkBundle, node_range: tuple[int, int], embedding_dim: int
    ) -> None:
        super().__init__()
        low, high = node_range
        self.node_range = node_range
        width = target_block_feature_width(bundle)
        self.mode = "attributes" if width else "embedding"

        if width:
            self.register_buffer(
                "features", bundle.data.x[low:high, :width].clone(), persistent=False
            )
            self.lookup: nn.Module | None = None
            self.output_dim = width
        else:
            self.features = None
            self.lookup = nn.Embedding(high - low, embedding_dim)
            self.output_dim = embedding_dim

    def forward(self) -> torch.Tensor:
        if self.lookup is not None:
            return self.lookup.weight
        return self.features


class TypedNodeInput(nn.Module):
    """``x_dict`` for an encoder that consumes the typed graph.

    A typed encoder touches every node type, not just the target one, so every type needs
    a feature matrix -- including the ones HGB leaves attribute-less. Real attributes are
    used where a type has them and a learned lookup elsewhere, the same rule
    :class:`TargetBlockInput` follows. :attr:`in_dims` gives the per-type widths, and
    :attr:`modes` records which branch each type took.
    """

    def __init__(self, bundle: LinkBundle, embedding_dim: int) -> None:
        super().__init__()
        data = bundle.data
        slices = dict(data.node_slices)
        widths = dict(data.feature_dims)

        self.node_type_names = tuple(data.node_type_names)
        self.in_dims: dict[str, int] = {}
        self.modes: dict[str, str] = {}
        self.lookups = nn.ModuleDict()
        self._attributed: list[str] = []

        for name in self.node_type_names:
            low, high = slices[name]
            width = widths[name]
            if width:
                self.register_buffer(
                    f"features_{name}", data.x[low:high, :width].clone(), persistent=False
                )
                self.in_dims[name] = width
                self.modes[name] = "attributes"
                self._attributed.append(name)
            else:
                self.lookups[name] = nn.Embedding(high - low, embedding_dim)
                self.in_dims[name] = embedding_dim
                self.modes[name] = "embedding"

    def forward(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in self.node_type_names:
            if name in self.lookups:
                out[name] = self.lookups[name].weight
            else:
                out[name] = getattr(self, f"features_{name}")
        return out


def build_edge_index_dict(
    bundle: LinkBundle, device: torch.device
) -> dict[tuple[str, str, str], torch.Tensor]:
    """``edge_index_dict`` over the training message-passing graph, in type-local ids.

    The bundle stores one flattened edge list with a relation id per edge; a typed encoder
    wants it split by relation and reindexed so each endpoint counts from zero within its
    own type. Only ``data.edge_index`` is read, so no validation or test positive can
    reach the encoder.
    """

    data = bundle.data
    slices = dict(data.node_slices)
    out: dict[tuple[str, str, str], torch.Tensor] = {}

    for relation_id, relation in enumerate(data.relation_tuples):
        source_type, _, destination_type = relation
        mask = data.edge_type == relation_id
        if not bool(mask.any()):
            continue
        row, column = data.edge_index[:, mask]
        out[tuple(relation)] = torch.stack(
            [row - slices[source_type][0], column - slices[destination_type][0]]
        ).to(device)
    return out


def node_type_metadata(
    bundle: LinkBundle, edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """PyG-style ``(node_types, edge_types)`` metadata for a typed encoder."""

    return list(bundle.data.node_type_names), list(edge_index_dict)


def scatter_to_full_graph(
    block: torch.Tensor, node_range: tuple[int, int], num_nodes: int
) -> torch.Tensor:
    """Place block-local embeddings at their global node ids.

    The shared link decoder indexes by global id, while these encoders emit rows for the
    block alone. Every scored pair lies inside the block by construction, so the rows left
    at zero are never read.
    """

    low, high = node_range
    embeddings = block.new_zeros((num_nodes, block.size(1)))
    embeddings[low:high] = block
    return embeddings


__all__ = [
    "TargetBlockInput",
    "TypedNodeInput",
    "build_edge_index_dict",
    "node_type_metadata",
    "scatter_to_full_graph",
    "target_block_feature_width",
    "target_node_range",
]
