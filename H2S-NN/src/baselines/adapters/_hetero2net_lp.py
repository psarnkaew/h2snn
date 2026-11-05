"""Hetero2Net's encoder on link prediction.

Their repository has no link-prediction task. The ``EdgeDecoder`` in ``models.py`` is an
*auxiliary* head used inside node-classification training to score homophilic against
heterophilic edge components (``utils.py:193-202``), not a task head. So this cell is the
campaign's construction, and all four are tuned.

What is kept from their method:

* **``HeteroGNN`` unchanged**, including ``DisenConv``'s split of each message into
  homophilic and heterophilic parts.
* **The disentanglement regulariser.** ``alpha * dist_corr(x_homo, x_hetero)`` is added on
  every step of their own ``train_full_batch`` (utils.py:161-167), unconditionally and
  independently of the label. Dropping it would be running a different model, so it is fed
  to the shared trainer through its ``aux_loss`` hook.
* **``metapaths=None``.** Their trainer guards both metapath dropping and the ``beta``
  link loss behind ``if metapaths is not None``, so this is a configuration their own code
  supports rather than something removed here. Their metapath lists are defined per
  node-classification dataset (``hetero2net/dataset.py``) and none exists for the
  link-prediction graphs; inventing them would be inventing their inductive bias.

The one structural adaptation: ``HeteroGNN.forward`` projects and returns rows for
``model.node_type`` alone -- a single attribute set externally in their ``main.py:186``.
Link prediction needs both endpoints of the target relation, so on a cross-type target
(LastFM's ``user``-``artist``) the encoder is run once per endpoint type and the two blocks
are placed at their global ids. Same-type targets need one pass.
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines.adapters._target_block import (
    TypedNodeInput,
    build_edge_index_dict,
)
from src.baselines.bundles import LinkBundle
from src.baselines.runtime import pushd_sys_path

VENDOR = "hetero2net"


class Hetero2NetLinkEncoder(torch.nn.Module):
    """Vendor ``HeteroGNN`` emitting full-graph node embeddings."""

    def __init__(
        self,
        bundle: LinkBundle,
        *,
        hidden_channels: int,
        embed_dim: int,
        dropout: float,
        num_layers: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        data = bundle.data
        self.num_nodes = int(data.x.size(0))
        self.node_slices = dict(data.node_slices)

        source_type, _, destination_type = data.target_relation
        # Ordered so a same-type target runs a single pass.
        self.endpoint_types = sorted({source_type, destination_type})

        self.input = TypedNodeInput(bundle, embedding_dim=hidden_channels)
        self.edge_index_dict = build_edge_index_dict(bundle, device)
        metadata = (list(data.node_type_names), list(self.edge_index_dict))

        with pushd_sys_path(VENDOR):
            from hetero2net.models import HeteroGNN  # type: ignore[import-not-found]

            self.vendor = HeteroGNN(
                metadata,
                # in_channels sizes their label-embedding table only, which is unused
                # here (y_emb is None), but must still be a valid width.
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=embed_dim,
                dropout=dropout,
                num_layers=num_layers,
                project=True,
            )
        # Set externally in their main.py:186 rather than in __init__; reassigned per
        # pass below when the target spans two node types.
        self.vendor.node_type = self.endpoint_types[0]
        self._last_components: list[tuple[dict, dict]] = []

    def forward(self) -> torch.Tensor:
        x_dict = self.input()
        embeddings: torch.Tensor | None = None
        self._last_components = []

        for node_type in self.endpoint_types:
            self.vendor.node_type = node_type
            # A fresh dict per pass: their forward writes into x_dict when y_emb is set,
            # and their hetero_conv rebinds it, so passing the same object twice would
            # hand the second pass the first one's convolved features.
            out = self.vendor(dict(x_dict), self.edge_index_dict)
            if isinstance(out, tuple):
                out, homos, heteros = out
                self._last_components.append((homos, heteros))
            if isinstance(out, dict):
                out = out[node_type]

            if embeddings is None:
                embeddings = out.new_zeros((self.num_nodes, out.size(1)))
            low, high = self.node_slices[node_type]
            embeddings[low:high] = out

        assert embeddings is not None
        return embeddings

    def disentanglement_loss(self, alpha: float) -> torch.Tensor:
        """``alpha * dist_corr(x_homo, x_hetero)``, summed as their utils.py:161-167 does.

        Reads the homophilic/heterophilic components cached by the most recent training
        forward, so it must be called after :meth:`forward` in the same graph.
        """

        total = torch.zeros((), device=next(self.parameters()).device)
        if alpha <= 0 or not self._last_components:
            return total

        with pushd_sys_path(VENDOR):
            from hetero2net.utils import dist_corr  # type: ignore[import-not-found]

            for homos, heteros in self._last_components:
                for homo_dict, hetero_dict in zip(homos, heteros):
                    for edge_type in homo_dict:
                        total = total + alpha * dist_corr(
                            homo_dict[edge_type], hetero_dict[edge_type]
                        )
        return total


__all__ = ["Hetero2NetLinkEncoder"]
