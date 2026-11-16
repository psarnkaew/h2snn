"""PSHGCN's ogbn-mag model with the input width of each node type from the dataset (card 1c F2; author decision S3 (b)).

Their `PSHGCN` (`vendor/PSHGCN/ogbn-mag/model.py:48-120`) hard-codes MAG's widths: the target type `P` 128, every other
type 256 (`embedings_x`, `:57-62`) and `lin_residue` 128 (`:68`). `build_model` constructs their class unchanged and
rebuilds only those width-bearing modules with the same constructor (`MLP(in, emb_dim, emb_dim, 2, dropout, bias)`,
`nn.Linear(in, emb_dim, bias=False)` with their xavier init); propagation, coefficients, label part, loss and optimiser
are theirs. The forward reads the target as letter `P` and each type by its letter (`:96-107`), so the adapter names
the target type `P`.

The label part needs `sparse_tools` (self-effect diagonals, `processing.py:133-151`), which could not be installed in
`h2snn-b` (its import needs `sortednp`, whose wheels require numpy < 2; card 1c F2 stop item): the ogbn-mag/mag-year cells
stay pending; this module is the width part only.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_model(widths: dict[str, int], emb_dim: int, hidden_x: int, hidden_l: int, nclass: int,
                label_keys: list[str], layers_x: int, layers_l: int, coe_num: int, expander: list[str],
                dropout: float, input_drop: float, bias: bool = False) -> nn.Module:
    """Their `PSHGCN` with `widths` (letter -> input width; the target is `P`)."""
    from src.baselines.runtime import pushd_sys_path

    with pushd_sys_path("pshgcn_mag"):
        from model import MLP, PSHGCN  # type: ignore[import-not-found]

        model = PSHGCN(emb_dim, hidden_x, hidden_l, nclass, list(widths), label_keys, layers_x, layers_l, coe_num,
                       expander, dropout, input_drop, bias)
        for node, width in widths.items():
            model.embedings_x[node] = MLP(width, emb_dim, emb_dim, model.layers_emb, dropout, bias=bias)
        model.lin_residue = nn.Linear(widths["P"], emb_dim, bias=False)
        nn.init.xavier_uniform_(model.lin_residue.weight, gain=nn.init.calculate_gain("relu"))
    return model


__all__ = ["build_model"]
