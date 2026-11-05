"""H2SGNN's encoder on link prediction.

H2SGNN is target-centric in a way PSHGCN is not. Its forward
(``H2SGNN/model.py:67-88``) reads ``features_list[0]`` -- the target type's feature matrix
alone -- and propagates through ``adjs_list``, a list of length-2 meta-path operators that
*start and end at the target type*, each cropped to the target block
(``H2SGNN/processing.py:145-158``). ``adjs`` and ``lis_t`` are accepted but never used;
``len(lis)`` sizes the per-meta-path mixing weights ``beta`` and ``temp_list``.

Carrying that to link prediction needs three things, all of which follow their code rather
than departing from it:

* **A target block.** For node classification the target is one node type; here it is the
  set of node types the target relation connects (LastFM ``user`` and ``artist``, PubMed
  ``disease`` alone). HGB lays node types out in contiguous blocks, and on all four
  link-prediction datasets that set is a single contiguous range, so their
  ``mask = (row <= target_max) & (col <= target_max)`` generalises to a half-open range.
* **Meta-paths.** Their rule is every ordered key pair ``(i, j)`` with ``i[-1] == j[0]``
  and both outer endpoints at the target type. On a single-node-type graph the key
  characters are relation labels rather than types, so the chaining test is dropped --
  the same branch PSHGCN takes for Amazon (see :mod:`._polynomial_lp`). YouTube's five
  relations therefore give 25 meta-paths, not 5.
* **Full-graph embeddings.** The model emits rows for target-block nodes only, while the
  shared decoder indexes by global node id. The block is scattered back into a zero-filled
  ``[N, d]`` matrix; every scored pair lies inside the block by construction, so the zero
  rows are never read.

Two observations from bringing this up, both left as the authors wrote them:

* ``GPR_GPR`` rebinds ``x`` inside its meta-path loop (model.py:120-124), so meta-path *m*
  filters the output of meta-paths 1..m-1 rather than the shared input. ``GPR_legendre``
  and ``GPR_jaco`` do not -- they start every meta-path from the same ``x``. On YouTube's
  25 chained operators that costs ``GPR_GPR`` about 19 AUROC points against
  ``GPR_legendre`` at equal budget. ``poly`` is in the search space, so validation
  selection decides; do not "fix" the rebinding, it is what their published code does.
* YouTube needs a real epoch budget before it means anything: at 6 epochs every variant
  sits at exactly 0.5 AUROC because the low-order terms have not been upweighted yet, and
  by 60 it is at 0.72 and still climbing. A short smoke run on this dataset says nothing
  about the method.
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines.adapters._polynomial_lp import build_relation_adjacencies
from src.baselines.adapters._target_block import (
    scatter_to_full_graph,
    target_node_range,
)
from src.baselines.bundles import LinkBundle
from src.baselines.runtime import pushd_sys_path

VENDOR = "h2sgnn"


def target_features(
    bundle: LinkBundle, node_range: tuple[int, int], device: torch.device
) -> torch.Tensor:
    """The single dense matrix ``features_list[0]`` the model projects.

    Their loader uses a type's real attributes when it has them and ``torch.eye``
    otherwise (H2SGNN/processing.py mirrors PSHGCN's). The identity is built sparse here:
    LastFM's block is 19,524 nodes, where a dense eye would be 1.5 GB. ``feat_project``
    is ``nn.Linear(..., bias=False)``, so a sparse input multiplies through unchanged.
    """

    low, high = node_range
    data = bundle.data
    slices = dict(data.node_slices)
    widths = dict(data.feature_dims)
    source_type, _, destination_type = data.target_relation
    endpoints = sorted({source_type, destination_type})

    distinct_widths = {widths[name] for name in endpoints}
    if len(distinct_widths) == 1 and distinct_widths != {0}:
        width = distinct_widths.pop()
        return data.x[low:high, :width].to(device)

    # Featureless, or types of differing widths sharing one block: fall back to an
    # identity over the block, which is what their eye() does per type.
    size = high - low
    index = torch.arange(size, device=device)
    return torch.sparse_coo_tensor(
        torch.stack([index, index]),
        torch.ones(size, device=device),
        (size, size),
    ).coalesce()


def build_metapath_operators(
    bundle: LinkBundle, node_range: tuple[int, int], device: torch.device
) -> tuple[list[Any], list[str]]:
    """Their ``adjs_list``/``lis``: length-2 meta-paths cropped to the target block."""

    from torch_sparse import SparseTensor
    from torch_sparse import matmul as torch_sparse_matmul

    low, high = node_range
    size = high - low
    # Compose on CPU, then move each finished operator across. YouTube's block is 2,000
    # nodes with 1.18M edges, so a relation operator is ~91% dense and a sparse-sparse
    # product asks for over 10 GB of intermediate on the way to a 3.6M-nnz result -- it
    # OOMs an 8 GB card while the finished operators fit comfortably.
    adjacencies, enforce_chaining = build_relation_adjacencies(
        bundle, torch.device("cpu")
    )

    data = bundle.data
    type_index = {name: index for index, name in enumerate(data.node_type_names)}
    source_type, _, destination_type = data.target_relation
    target_letters = {
        chr(ord("A") + type_index[name])
        for name in {source_type, destination_type}
    }

    operators: list[Any] = []
    names: list[str] = []
    for first in adjacencies:
        for second in adjacencies:
            if enforce_chaining:
                # Their condition verbatim: the hop types must meet, and both outer
                # endpoints must be the target type.
                if first[-1] != second[0]:
                    continue
                if first[0] not in target_letters or second[1] not in target_letters:
                    continue
            product = torch_sparse_matmul(adjacencies[first], adjacencies[second])
            row, column, value = product.coo()
            mask = (row >= low) & (row < high) & (column >= low) & (column < high)
            operators.append(
                SparseTensor(
                    row=row[mask] - low,
                    col=column[mask] - low,
                    value=value[mask],
                    sparse_sizes=(size, size),
                ).to(device)
            )
            names.append(first + second)

    if not operators:
        raise RuntimeError(
            f"{bundle.dataset}: no length-2 meta-path returns to the target block; "
            "H2SGNN has nothing to propagate through"
        )
    return operators, names


class H2SGNNLinkEncoder(torch.nn.Module):
    """Vendor H2SGNN emitting full-graph node embeddings."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        metapath_names: list[str],
        args: Any,
        *,
        num_nodes: int,
        node_range: tuple[int, int],
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node_range = node_range
        with pushd_sys_path(VENDOR):
            from model import H2SGNN  # type: ignore[import-not-found]

            # lis is a *list* here, not the dict PSHGCN uses: its length sizes beta and
            # temp_list. lis_t is accepted and ignored (H2SGNN/processing.py:161).
            self.vendor = H2SGNN(in_dim, embed_dim, metapath_names, {}, args)

    def forward(
        self, features: torch.Tensor, metapath_operators: list[Any]
    ) -> torch.Tensor:
        # The first argument (`adjs`) is unused by their forward; pass the list it does
        # read for both so the call site cannot drift.
        block = self.vendor(metapath_operators, [features], metapath_operators)
        return scatter_to_full_graph(block, self.node_range, self.num_nodes)


def parameter_groups(
    model: H2SGNNLinkEncoder, head: torch.nn.Module, args: Any, params: Any
) -> list[dict[str, Any]]:
    """The six-group AdamW layout of H2SGNN/main.py:22-28.

    The global filter coefficients (``temp``), the per-meta-path coefficients
    (``temp_list``) and the meta-path mixing weights (``beta``) each carry their own
    learning rate and weight decay; collapsing them into one group changes the method.
    """

    vendor = model.vendor
    base = {"weight_decay": float(params.get("weight_decay", 0.0)), "lr": float(params.get("lr", 0.01))}
    return [
        {"params": vendor.feat_project.parameters(), **base},
        {"params": vendor.lin1.parameters(), **base},
        {"params": vendor.lin2.parameters(), **base},
        {"params": head.parameters(), **base},
        {"params": vendor.temp, "weight_decay": args.temp_wd, "lr": args.temp_lr},
        {"params": vendor.temp_list.parameters(), "weight_decay": args.temp_list_wd, "lr": args.temp_list_lr},
        {"params": vendor.beta, "weight_decay": args.beta_wd, "lr": args.beta_lr},
    ]


__all__ = [
    "H2SGNNLinkEncoder",
    "build_metapath_operators",
    "parameter_groups",
    "target_features",
]
