"""HAN's encoder on link prediction.

HGB's link-prediction benchmark ships no HAN, so this cell is the campaign's construction
rather than a protocol of theirs. The model itself is untouched: HGB's
``model_hetero.HAN`` runs as published, and only its two inputs are supplied here.

``HANLayer.forward`` turns each meta-path into a homogeneous graph with
``dgl.metapath_reachable_graph(g, meta_path)`` and runs one ``GATConv`` per meta-path,
then mixes them by semantic attention (HGB/NC/.../HAN/model_hetero.py:75-90). Two
consequences shape what has to be handed to it:

* every meta-path must start and end at the **same** node type, since one feature matrix
  feeds all of them; and
* the meta-path composition is something the layer performs itself, from a heterograph.

So the graph built here has a single node type -- the target block from
:mod:`._target_block` -- and one edge type per meta-path, whose edges are the *already
composed* length-2 relation product from :mod:`._h2sgnn_lp`. Each meta-path is then the
one-hop path ``[mp_i]``, and ``metapath_reachable_graph`` returns exactly that operator's
graph. Precomputing the composition rather than letting DGL do it changes nothing about
what HAN sees; it just avoids rebuilding a heterograph whose typed structure HAN would
immediately collapse anyway.

Meta-path edges are **binarised**: ``metapath_reachable_graph`` returns an unweighted
graph, and HAN's ``GATConv`` learns its own edge weights by attention, so carrying the
transition probabilities through would be a second, silent weighting.
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines.adapters._h2sgnn_lp import build_metapath_operators
from src.baselines.adapters._target_block import (
    TargetBlockInput,
    scatter_to_full_graph,
)
from src.baselines.bundles import LinkBundle
from src.baselines.runtime import _dgl_compat  # noqa: F401  (import for side effect)
from src.baselines.runtime import pushd_sys_path

#: Node type name for the single-type meta-path graph. Arbitrary -- HAN never reads it.
BLOCK_TYPE = "target"


def build_metapath_graph(
    bundle: LinkBundle, node_range: tuple[int, int], device: torch.device
) -> tuple[Any, list[list[str]], int]:
    """A one-node-type DGL heterograph with one edge type per meta-path."""

    import dgl

    low, high = node_range
    size = high - low
    operators, names = build_metapath_operators(bundle, node_range, torch.device("cpu"))

    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    meta_paths: list[list[str]] = []
    for index, operator in enumerate(operators):
        row, column, _ = operator.coo()
        # Binarised: coalesce already merged duplicates, so dropping the values is all
        # that is needed to match metapath_reachable_graph's unweighted output.
        relation = f"mp{index}_{names[index]}"
        edge_dict[(BLOCK_TYPE, relation, BLOCK_TYPE)] = (row, column)
        meta_paths.append([relation])

    graph = dgl.heterograph(edge_dict, num_nodes_dict={BLOCK_TYPE: size}).to(device)
    return graph, meta_paths, len(operators)


class HANLinkEncoder(torch.nn.Module):
    """Vendor HAN emitting full-graph node embeddings."""

    def __init__(
        self,
        bundle: LinkBundle,
        node_range: tuple[int, int],
        *,
        meta_paths: list[list[str]],
        hidden_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        vendors: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.node_range = node_range
        self.num_nodes = int(bundle.data.x.size(0))
        self.input = TargetBlockInput(bundle, node_range, embedding_dim=hidden_dim)

        with pushd_sys_path(*vendors):
            from model_hetero import HAN  # type: ignore[import-not-found]

            self.vendor = HAN(
                meta_paths=meta_paths,
                in_size=self.input.output_dim,
                hidden_size=hidden_dim,
                out_size=embed_dim,
                num_heads=[num_heads],
                dropout=dropout,
            )

    def forward(self, graph: Any) -> torch.Tensor:
        block = self.vendor(graph, self.input())
        return scatter_to_full_graph(block, self.node_range, self.num_nodes)


__all__ = ["BLOCK_TYPE", "HANLinkEncoder", "build_metapath_graph"]
