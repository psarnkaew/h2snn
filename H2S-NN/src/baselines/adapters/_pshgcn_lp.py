"""PSHGCN's link-prediction encoder, driven from the canonical bundle.

Their ``hgb-lp/`` tree is a separate codebase from ``hgb/``: a different model file, a
different preprocessing file, and two runner scripts. This module runs that model
unchanged and returns node embeddings, so the campaign's shared decoder scores the pairs.

Two things are worth knowing about the vendor forward
(``PSHGCN/hgb-lp/pshgcn.py:76-135``):

* it ends ``logits = self.lin2(res); return self.decoder(logits[left], logits[right], mid)``
  -- there is no seam that yields the embedding matrix on its own. Rather than duplicate
  the propagation, the adapter swaps ``model.decoder`` for a passthrough and calls the
  model with ``left = right = arange(N)``, which returns exactly ``logits``.
* the polynomial is applied twice, once with ``lis`` and once with ``lis_t``, so the
  activation cost is two full passes over every monomial. Term counts are on the run record
  as ``polynomial_terms``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.baselines.adapters._polynomial_lp import (
    build_relation_adjacencies,
    build_term_lists,
    polynomial_term_count,
)
from src.baselines.bundles import LinkBundle
from src.baselines.runtime import pushd_sys_path

# hgb-lp/pshgcn.py opens with `import dgl` (leftover -- it never uses DGL), so the
# GraphBolt shim has to be in place before the vendor import. The node-classification
# model has no such import, which is why only this module needs it.
from src.baselines.runtime import _dgl_compat  # noqa: F401,E402  (import for side effect)

VENDOR = "pshgcn_lp"


class _ReturnLeftEmbedding(nn.Module):
    """Stand-in decoder returning the left operand untouched.

    With ``left = arange(N)`` this makes the vendor forward return its full ``lin2``
    output, which is the node embedding matrix the shared head expects. The vendor's own
    Dot/DisMult are then unused -- the campaign scores every baseline with the same
    decoder so the tables compare encoders.
    """

    def forward(
        self,
        left_emb: torch.Tensor,
        right_emb: torch.Tensor,  # noqa: ARG002  (vendor calling convention)
        relation_ids: Any,  # noqa: ARG002
    ) -> torch.Tensor:
        return left_emb


class PSHGCNLinkEncoder(nn.Module):
    """Vendor PSHGCN (hgb-lp) wrapped to emit node embeddings."""

    def __init__(
        self,
        in_dims: list[int],
        embed_dim: int,
        lis: dict[int, list[str]],
        lis_t: dict[int, list[str]],
        args: SimpleNamespace,
    ) -> None:
        super().__init__()
        with pushd_sys_path(VENDOR):
            from pshgcn import PSHGCN  # type: ignore[import-not-found]

            self.vendor = PSHGCN(in_dims, embed_dim, lis, lis_t, args)
        # Replaced after construction so the vendor's __init__ runs exactly as shipped
        # (it builds Dot or DisMult from args.decode before we discard it).
        self.vendor.decoder = _ReturnLeftEmbedding()

    def forward(
        self, adjacencies: dict[str, Any], features_list: list[torch.Tensor]
    ) -> torch.Tensor:
        # The vendor concatenates the per-type blocks along dim 0, so the flattened node
        # count is the sum of the block heights.
        num_nodes = sum(int(feature.shape[0]) for feature in features_list)
        index = torch.arange(num_nodes, device=features_list[0].device)
        return self.vendor(adjacencies, features_list, index, index, None)


def vendor_args(params: Any, *, device: torch.device) -> SimpleNamespace:
    """The argparse-shaped namespace hgb-lp's PSHGCN reads.

    Defaults are ``training.py``'s (their LastFM runner), so an un-tuned call still
    lands on published values rather than arbitrary ones.
    """

    return SimpleNamespace(
        device=device,
        K=int(params.get("K", 1)),
        emb_dim=int(params.get("emb_dim", 256)),
        hidden=int(params.get("hidden", 128)),
        input_drop=float(params.get("input_drop", 0.5)),
        dropout=float(params.get("dropout", 0.6)),
        # Consumed by the vendor __init__ to build a decoder we then replace; kept
        # faithful so construction matches theirs parameter-for-parameter.
        decode=str(params.get("decoder_type", "dot")),
    )


def build_polynomial_inputs(
    bundle: LinkBundle, order: int, device: torch.device
) -> tuple[dict[str, Any], dict[int, list[str]], dict[int, list[str]], int]:
    """Relation operators and monomial lists for one link-prediction dataset."""

    adjacencies, enforce_chaining = build_relation_adjacencies(bundle, device)
    lis, lis_t = build_term_lists(
        adjacencies, order, enforce_type_chaining=enforce_chaining
    )
    return adjacencies, lis, lis_t, polynomial_term_count(lis)


def parameter_groups(
    model: PSHGCNLinkEncoder, head: nn.Module, args: SimpleNamespace, params: Any
) -> list[dict[str, Any]]:
    """The four-group AdamW layout of PSHGCN/hgb-lp/training.py:48-51.

    The propagation coefficients ``W`` carry their own learning rate and weight decay,
    which is central to the method -- folding them into one group changes the model.
    The shared decoder joins the ordinary group so it trains end to end with the encoder.
    """

    weight_decay = float(params.get("weight_decay", 0.8))
    learning_rate = float(params.get("lr", 5e-4))
    vendor = model.vendor
    return [
        {"params": vendor.feat_project.parameters(), "weight_decay": weight_decay, "lr": learning_rate},
        {"params": vendor.lin1.parameters(), "weight_decay": weight_decay, "lr": learning_rate},
        {"params": vendor.lin2.parameters(), "weight_decay": weight_decay, "lr": learning_rate},
        {"params": head.parameters(), "weight_decay": weight_decay, "lr": learning_rate},
        {
            "params": vendor.W,
            "weight_decay": float(params.get("prop_wd", 5e-5)),
            "lr": float(params.get("prop_lr", 0.05)),
        },
    ]


__all__ = [
    "PSHGCNLinkEncoder",
    "build_polynomial_inputs",
    "parameter_groups",
    "vendor_args",
]
