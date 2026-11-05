"""LatGRL's unsupervised encoder on link prediction.

LatGRL trains with no labels and is then probed, so its link-prediction cell follows the
same two-stage shape as its node-classification one: run their objective to convergence,
freeze, and train only the shared decoder on the frozen embeddings. That keeps the
comparison honest -- a supervised decoder over frozen features is what their own protocol
evaluates (``LatGRL/utils/evaluate.py``).

**Why only two of the four datasets.** Their latent-graph construction is built on feature
similarity (``module/graph_generating.py:30-64``)::

    fea_sim = cosine(x, x)
    sim_l   = adj_sim * fea_sim ;  sim_l -= diag(sim_l)
    kg_pos  = top_k(sim_l)

With the HGB convention for an attribute-less node type -- an identity feature matrix --
``fea_sim`` *is* the identity, so ``sim_l`` is diagonal and the next line zeroes it. The
low-pass latent graph, which is the method's central construction, then has no signal and
its top-k is arbitrary. LastFM and YouTube carry no node attributes at all, so those two
cells are declared unsupported with that reason rather than reported as a number. Amazon
(1,156-dim) and PubMed (200-dim) have real attributes and run normally.

**Cost.** ``graph_construction`` is dense throughout: an ``N x N`` cosine matrix, a dense
copy of every meta-path adjacency, and an ``N x N`` identity, several live at once. On
PubMed's 20,163-node disease block that is ~1.6 GB per matrix, so this cell needs a
large-memory GPU; Amazon's 10,099-node block is ~0.4 GB per matrix and fits comfortably.
Their sampler variant (``main_sampler.py``) does not help here -- it batches the *training
loop*, while ``graph_sampler_generating.graph_construction`` is dense in exactly the same
way.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from src.baselines.adapters._h2sgnn_lp import build_metapath_operators
from src.baselines.adapters._target_block import (
    target_block_feature_width,
    target_node_range,
)
from src.baselines.bundles import LinkBundle
from src.baselines.runtime import pushd_sys_path

VENDOR = "latgrl"

#: Datasets whose target block carries real node attributes. See the module docstring.
SUPPORTED_LP_DATASETS = frozenset({"amazon", "pubmed"})

FEATURELESS_REASON = (
    "LatGRL builds its latent graphs from feature cosine similarity "
    "(module/graph_generating.py:30-64); with no node attributes that similarity is the "
    "identity, so the low-pass graph is zeroed by the diagonal removal on the next line "
    "and its top-k selection carries no signal"
)


def block_metapath_adjacencies(
    bundle: LinkBundle, node_range: tuple[int, int], device: torch.device
) -> list[torch.Tensor]:
    """Binarised meta-path adjacencies over the target block, as sparse COO tensors.

    LatGRL treats meta-path graphs as unweighted (their cached ``.npz`` inputs are 0/1),
    and ``graph_construction`` row-normalises them itself, so the transition
    probabilities carried by the composed operators are dropped here.
    """

    operators, _ = build_metapath_operators(bundle, node_range, torch.device("cpu"))
    size = node_range[1] - node_range[0]

    adjacencies: list[torch.Tensor] = []
    for operator in operators:
        row, column, _ = operator.coo()
        adjacencies.append(
            torch.sparse_coo_tensor(
                torch.stack([row, column]),
                torch.ones(row.numel()),
                (size, size),
            ).coalesce().to(device)
        )
    return adjacencies


def block_features(
    bundle: LinkBundle, node_range: tuple[int, int], device: torch.device
) -> torch.Tensor:
    """The block's real attributes. Callers must gate on :data:`SUPPORTED_LP_DATASETS`."""

    width = target_block_feature_width(bundle)
    if not width:
        raise RuntimeError(f"{bundle.dataset}: {FEATURELESS_REASON}")
    low, high = node_range
    return bundle.data.x[low:high, :width].to(device)


def train_unsupervised(
    bundle: LinkBundle,
    params: Any,
    device: torch.device,
) -> tuple[torch.Tensor, float, int]:
    """Run their objective to convergence and return frozen block embeddings.

    Returns ``(embeddings, best_loss, num_metapaths)``.
    """

    node_range = target_node_range(bundle)
    feat = block_features(bundle, node_range, device)
    adjs = block_metapath_adjacencies(bundle, node_range, device)

    args = SimpleNamespace(
        graph_k=int(params.get("graph_k", 5)),
        k_pos=int(params.get("k_pos", 0)),
        dataset=bundle.dataset,
    )

    with pushd_sys_path(VENDOR):
        from module.graph_generating import graph_process  # type: ignore[import-not-found]
        from module.LatGRL import LatGRL  # type: ignore[import-not-found]

        try:
            adjs_l, adjs_h, adjs_o, pos = graph_process(adjs, feat, args)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            size = int(feat.size(0))
            per_matrix = size * size * 4 / 1024**3
            raise RuntimeError(
                f"LatGRL's latent-graph construction ran out of memory on "
                f"{bundle.dataset}: the target block is {size:,} nodes, so each dense "
                f"N x N intermediate is {per_matrix:.1f} GB and graph_construction holds "
                f"several at once (cosine similarity, one per meta-path, and an "
                f"identity). This cell needs a large-memory GPU. Their sampler variant "
                f"does not help -- it batches the training loop, not this construction."
            ) from exc
        model = LatGRL(
            int(feat.size(1)),
            len(adjs),
            int(params.get("hidden_dim", 128)),
            int(params.get("embed_dim", 64)),
            float(params.get("tau", 1.0)),
            float(params.get("dropout", 0.2)),
            torch.nn.ELU(),
            float(params.get("drop_feat", 0.1)),
            int(feat.size(0)),
            bundle.dataset,
            float(params.get("alpha", 2.0)),
            int(params.get("nlayer", 2)),
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get("lr", 5e-4)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )

    # Early stopping on the training objective, as their main.py does. No labels here,
    # and no validation edge either, so nothing about the split can leak in.
    best_loss = float("inf")
    stale = 0
    patience = int(params.get("patience", 10))
    for _ in range(int(params.get("nb_epochs", 200))):
        model.train()
        optimizer.zero_grad()
        loss = model(feat, adjs_l, adjs_h, adjs_o, pos[0])
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.eval()
    with torch.no_grad():
        block = model.get_embeds(feat, adjs_o).detach()

    embeddings = block.new_zeros((int(bundle.data.x.size(0)), block.size(1)))
    low, high = node_range
    embeddings[low:high] = block
    return embeddings, best_loss, len(adjs)


__all__ = [
    "FEATURELESS_REASON",
    "SUPPORTED_LP_DATASETS",
    "block_features",
    "block_metapath_adjacencies",
    "train_unsupervised",
]
