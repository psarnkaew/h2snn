"""H2G-former's encoder on link prediction.

H2GB has link-prediction machinery in graphgym (``cfg.dataset.task = 'link_pred'``,
``GNNEdgeHead``), but its benchmark datasets are all node classification and it ships no
link-prediction config, so this cell is the campaign's construction and all four are tuned.

``GTModel`` is used exactly as for node classification -- same encoder, same transformer
stack -- with one change at the very end. Their ``hetero_node`` head projects
``batch[cfg.dataset.task_entity].x`` alone and then indexes it by a ``{split}_mask``
(``H2GB/head/hetero_node.py:44-52``): a single classified node type, selected by a global
config value fixed at construction. Link prediction needs embeddings for both endpoints of
the target relation, and there is no split mask.

:class:`AllTypesHead` therefore reuses that head's own ``layer_post_mp`` MLP -- the same
module, same weights, same construction -- and applies it to every node type instead of
one, returning the dict rather than a masked ``(pred, label)`` pair. Nothing about the
model before the head changes.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.baselines.adapters._target_block import (
    TypedNodeInput,
    build_edge_index_dict,
)
from src.baselines.bundles import LinkBundle


class AllTypesHead(nn.Module):
    """Their ``hetero_node`` head's MLP, applied to every node type.

    Constructed from an existing head so the MLP is the one ``GTModel.__init__`` built,
    at the width their config logic chose.
    """

    def __init__(self, vendor_head: nn.Module) -> None:
        super().__init__()
        self.layer_post_mp = vendor_head.layer_post_mp

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        return {
            node_type: self.layer_post_mp(batch[node_type].x)
            for node_type in batch.node_types
        }


def build_link_hetero_data(
    bundle: LinkBundle,
    node_input: TypedNodeInput,
    device: torch.device,
) -> Any:
    """A ``HeteroData`` over the training message-passing graph.

    Node features come from :class:`TypedNodeInput` so attribute-less types (LastFM and
    YouTube have no attributes at all) get a learned lookup rather than a dense identity
    the transformer could not afford. The values placed here are only for shape discovery
    at construction time; :class:`H2GformerLinkEncoder` overwrites them with live tensors
    on every forward so the lookups actually receive gradient.
    """

    from torch_geometric.data import HeteroData

    data = bundle.data
    slices = dict(data.node_slices)
    edge_index_dict = build_edge_index_dict(bundle, device)

    hetero = HeteroData()
    with torch.no_grad():
        x_dict = node_input()
    for node_type in data.node_type_names:
        low, high = slices[node_type]
        hetero[node_type].x = x_dict[node_type].detach().to(device)
        hetero[node_type].num_nodes = high - low
    for relation, edge_index in edge_index_dict.items():
        hetero[relation].edge_index = edge_index
    return hetero


class H2GformerLinkEncoder(nn.Module):
    """Vendor ``GTModel`` emitting full-graph node embeddings.

    Owns the typed input module so its learned lookups are optimised alongside the
    transformer, and refreshes ``batch[nt].x`` from it before every forward. That refresh
    is required for a second reason: the ``Hetero_Raw`` encoder and the head both
    overwrite ``batch[nt].x`` in place, so a reused ``HeteroData`` would otherwise feed
    one epoch's output back in as the next epoch's input.
    """

    def __init__(
        self,
        vendor_model: nn.Module,
        node_input: TypedNodeInput,
        hetero: Any,
        slices: dict[str, tuple[int, int]],
        num_nodes: int,
    ) -> None:
        super().__init__()
        self.vendor = vendor_model
        self.input = node_input
        self.hetero = hetero
        self.slices = slices
        self.num_nodes = num_nodes

    def forward(self) -> torch.Tensor:
        x_dict = self.input()
        for node_type, value in x_dict.items():
            self.hetero[node_type].x = value

        out = self.vendor(self.hetero)
        if isinstance(out, tuple):
            out = out[0]

        any_block = next(iter(out.values()))
        embeddings = any_block.new_zeros((self.num_nodes, any_block.size(1)))
        for node_type, block in out.items():
            low, high = self.slices[node_type]
            embeddings[low:high] = block
        return embeddings


__all__ = ["AllTypesHead", "H2GformerLinkEncoder", "build_link_hetero_data"]
