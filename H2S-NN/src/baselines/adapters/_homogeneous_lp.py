"""Link prediction for the homogeneous baselines (NSD, CSNN, and friends).

These models were published as node classifiers on homogeneous graphs, so link prediction
needs two things they do not supply: a reduction of the heterogeneous graph, and a pair
decoder. Both come from the campaign's shared components, so every link-prediction
baseline is scored the same way and only the encoder differs.

Protocol, following HGB's link-prediction benchmark:

* **Message-passing graph** is ``HomogeneousView.edge_index``: HGB's own reduction for
  link prediction -- sum the relation adjacencies, append the reverse of every edge, add
  a self-loop per node (``gen_edge_list(dl, reverse=True)``,
  HGB/LP/benchmark/methods/GNN/utils.py:15). It is built from ``data.edge_index``, which
  holds training positives only, so no validation or test edge enters it directly. Every
  homogeneous link baseline reads this same graph, GCN and GAT included.
* **Test negatives** are HGB's 1:1 two-hop hard negatives, fixed in the bundle and shared
  by every model (verified to match ``dl.get_test_neigh()``'s 18,567+18,567 on LastFM).
* **Decoder** is ``dot`` or ``distmult``, validation-selected, mirroring HGB's own
  link-prediction models which ship both and let each method choose.
* **Metrics** are AUROC (the selection monitor) and AUPR, with HGB's source-grouped MRR
  reported alongside as a diagnostic.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from src.baselines.bundles import LinkBundle
from src.baselines.heads import SharedLinkHead
from src.baselines.budget import budget_kwargs
from src.baselines.protocol import LINK_PREDICTION, BaselineParams
from src.baselines.training import train_link_prediction
from src.utils.misc import seed_everything

#: Encoder factory: (homogeneous view, message-passing edge_index, embedding width)
#: -> module whose forward(features_list) returns [num_nodes, embedding width].
EncoderFactory = Callable[[Any, torch.Tensor, int], torch.nn.Module]


def fit_homogeneous_link_prediction(
    name: str,
    bundle: LinkBundle,
    params: BaselineParams,
    *,
    seed: int,
    device: torch.device,
    build_encoder: EncoderFactory,
    self_loops: bool = False,
) -> dict[str, float]:
    """Train a homogeneous encoder plus the shared link decoder on one LP dataset.

    ``self_loops`` selects which form of the reduced graph the encoder expects: sheaf
    models build their own Laplacian and require exact symmetric pairs with no diagonal,
    while message-passing models want HGB's self-looped graph.
    """

    seed_everything(seed, device)
    data = bundle.data.to(device)

    # feats_type 0 keeps real attributes where a type has them (PubMed, Amazon) and falls
    # back to an identity encoding where it does not (LastFM, YouTube).
    view = bundle.homogeneous(feats_type=int(params.get("feats_type", 0)))
    features = [
        feature.coalesce().to(device) if feature.is_sparse else feature.to(device)
        for feature in view.features_list
    ]
    edge_index = (
        view.edge_index if self_loops else view.edge_index_without_self_loops
    ).to(device)

    embed_dim = int(params.get("embed_dim", params.get("hidden_channels", 64)))
    model = build_encoder(view, edge_index, embed_dim).to(device)
    head = SharedLinkHead(
        str(params.get("decoder_type", "dot")),
        num_relations=max(int(data.num_target_relations), 1),
        dim=embed_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )

    record = train_link_prediction(
        model=model,
        head=head,
        embed=lambda: model(features),
        data=data,
        optimizer=optimizer,
        device=device,
        **budget_kwargs(LINK_PREDICTION, params),
    )
    record["params_source"] = params.source
    record["decoder_type"] = str(params.get("decoder_type", "dot"))
    record["input_projection"] = "per_type_linear"
    return record


__all__ = ["EncoderFactory", "fit_homogeneous_link_prediction"]
