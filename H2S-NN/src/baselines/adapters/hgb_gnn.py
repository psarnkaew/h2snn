"""GCN and GAT, as run by the HGB benchmark's own homogeneous baselines.

Serves two registry rows from one module because HGB implements them in one file
(``HGB/NC/benchmark/methods/GNN/GNN.py``) behind a ``--model-type`` switch.

The graph reduction is HGB's: sum the relation adjacencies, symmetrise, drop self-loops
then add them back, and discard edge direction and relation type
(``run.py:81-84``). Each node type keeps its own feature width and gets its own
``nn.Linear`` into the shared hidden size (``GNN.py:29``); attribute-less types get an
identity encoding. All of that is reproduced by
:meth:`src.baselines.bundles.NodeBundle.homogeneous`, which is also what the other
homogeneous baselines consume, so GCN, GAT, ChebNetII, NSD and CSNN all see the same graph.

Deviations from ``run.py``, both recorded on the run record:

1. **The split.** HGB carves 20% of the training mask off with an unseeded
   ``np.random.shuffle`` (``utils/data.py:19-25``). We use the project's deterministic
   carve-out of the same HGB training mask, so the ratio matches HGB exactly and every
   model in the table sees identical indices.
2. **Model selection** is on validation macro-F1 rather than validation loss, uniformly.

``feats_type`` follows HGB's per-dataset recommendations by default and is searchable.
"""

from __future__ import annotations

from typing import Any

import torch

# Must precede any dgl import.
from src.baselines.runtime._dgl_compat import resolve_dgl_device
from src.baselines.adapters._pyg_compat import legacy_conv_attrs
from src.baselines.bundles import HGB_DEFAULT_FEATS_TYPE, LinkBundle, NodeBundle
from src.baselines.heads import SharedGraphHead, SharedLinkHead
from src.baselines.budget import budget_kwargs
from src.baselines import search_grids as grids
from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    BaselineParams,
    UnsupportedCell,
)
from src.baselines.published import load_published_params
from src.baselines.runtime import pushd_sys_path
from src.baselines.training import (
    node_classification_criterion,
    train_full_batch_node_classification,
    train_graph_classification,
    train_link_prediction,
)
from src.utils.misc import seed_everything

VENDOR = "hgb_nc_gnn"
# HGB ships a separate, PyG-based implementation for link prediction; no DGL there.
VENDOR_LP = "hgb_lp_gnn"
SUPPORTED_DATASETS = frozenset(HGB_DEFAULT_FEATS_TYPE)
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "youtube", "amazon", "pubmed"})
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})


def _build_dgl_graph(view: Any, device: torch.device) -> Any:
    """HGB's graph: symmetrised, self-looped, untyped.

    ``HomogeneousView.edge_index`` already carries the symmetrisation and self-loops, so
    this only wraps it in the DGLGraph the vendored modules expect.
    """

    import dgl

    source, destination = view.edge_index
    graph = dgl.graph(
        (source, destination), num_nodes=view.num_nodes, device=torch.device("cpu")
    )
    return graph.to(device)


def _to_device(features: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    """Move the per-type feature blocks, keeping identity blocks sparse."""

    moved: list[torch.Tensor] = []
    for feature in features:
        if feature.is_sparse:
            moved.append(feature.coalesce().to(device))
        else:
            moved.append(feature.to(device))
    return moved


class HGBHomogeneousAdapter:
    """Shared adapter for HGB's GCN and GAT."""

    def __init__(self, name: str, model_type: str) -> None:
        self.name = name
        self.model_type = model_type

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return dataset in SUPPORTED_DATASETS
        if task == LINK_PREDICTION:
            return dataset in SUPPORTED_LP_DATASETS
        if task == GRAPH_CLASSIFICATION:
            return dataset in SUPPORTED_GC_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task == GRAPH_CLASSIFICATION:
            space = {
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,
                "hidden_dim": {"values": [64, 128, 256]},
                "dropout": grids.DROPOUT,
                "num_layers": {"values": [2, 3]},
            }
            if self.model_type == "gat":
                space["num_heads"] = {"values": [4, 8]}
            return space
        if task == LINK_PREDICTION:
            space = {
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,
                "hidden_dim": {"values": [64, 128, 256]},
                "dropout": grids.DROPOUT,
                "num_layers": {"values": [2, 3]},
                # HGB itself offers both decoders and lets each method choose
                # (HGB/LP/benchmark/methods/GNN/GNN.py), so it is validation-selected.
                "decoder_type": {"values": ["dot", "distmult"]},
            }
            if self.model_type == "gat":
                space["num_heads"] = {"values": [4, 8]}
            return space
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(self.name, task, dataset, "search space not defined yet")
        space: dict[str, Any] = {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden_dim": {"values": [64, 128, 256]},
            "dropout": grids.DROPOUT,
            "num_layers": {"values": [2, 3]},
            # HGB treats this as a tuned knob ("To be tuned" in their README); 1 is
            # omitted because the 10-dim zero block it substitutes is strictly less
            # informative than the identity encoding of 2.
            "feats_type": {"values": [0, 2, 3]},
        }
        if self.model_type == "gat":
            space["num_heads"] = {"values": [4, 8]}
        return space

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        return load_published_params(self.name, task, dataset)

    def fit_task(
        self,
        bundle: Any,
        params: BaselineParams,
        *,
        task: str,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        if task == LINK_PREDICTION:
            return self._fit_link_prediction(bundle, params, seed=seed, device=device)
        if task == GRAPH_CLASSIFICATION:
            return self._fit_graph_classification(
                bundle, params, seed=seed, device=device
            )
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(
                self.name, task, bundle.dataset, "unsupported task"
            )

        # DGL ships separate CPU and CUDA builds; fall back rather than crash if this
        # environment has the CPU one. Recorded on the run record so a slow cell is
        # explainable after the fact.
        device = resolve_dgl_device(device)
        seed_everything(seed, device)
        feats_type = int(
            params.get("feats_type", HGB_DEFAULT_FEATS_TYPE[bundle.dataset])
        )
        view = bundle.homogeneous(feats_type=feats_type)
        graph = _build_dgl_graph(view, device)
        features = _to_device(view.features_list, device)

        data = bundle.data
        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()
        train_idx = data.target_train_mask.nonzero(as_tuple=False).view(-1).to(device)
        val_idx = data.target_val_mask.nonzero(as_tuple=False).view(-1).to(device)
        test_idx = data.target_test_mask.nonzero(as_tuple=False).view(-1).to(device)

        hidden_dim = int(params.get("hidden_dim", 64))
        num_layers = int(params.get("num_layers", 2))
        dropout = float(params.get("dropout", 0.5))

        with pushd_sys_path(VENDOR):
            import torch.nn.functional as F
            from GNN import GAT, GCN  # type: ignore[import-not-found]

            if self.model_type == "gat":
                num_heads = int(params.get("num_heads", 8))
                heads = [num_heads] * num_layers + [1]
                model = GAT(
                    graph,
                    view.in_dims,
                    hidden_dim,
                    bundle.num_classes,
                    num_layers,
                    heads,
                    F.elu,
                    dropout,
                    dropout,
                    float(params.get("slope", 0.05)),
                    False,
                )
            else:
                model = GCN(
                    graph,
                    view.in_dims,
                    hidden_dim,
                    bundle.num_classes,
                    num_layers,
                    F.elu,
                    dropout,
                )
            model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 5e-4)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )

        record = train_full_batch_node_classification(
            model=model,
            forward=lambda: model(features),
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            optimizer=optimizer,
            criterion=node_classification_criterion(bundle.is_multilabel),
            is_multilabel=bundle.is_multilabel,
            **budget_kwargs(NODE_CLASSIFICATION, params),
            # HGB's models emit logits for every node in the flattened space; the target
            # type occupies the leading block.
            num_target=int(labels.shape[0]),
        )
        record["params_source"] = params.source
        record["feats_type"] = float(feats_type)
        record["device"] = device.type
        return record


    def _fit_link_prediction(
        self,
        bundle: LinkBundle,
        params: BaselineParams,
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        """Run HGB's own link-prediction encoder plus the campaign's shared decoder.

        HGB implements LP separately from NC and, unlike the NC models, that code is
        already PyG-based (HGB/LP/benchmark/methods/GNN/GNN.py), so no DGL is involved and
        this runs on the GPU everywhere. Their ``encode`` is used verbatim; their bundled
        Dot/DisMult decoder is replaced by the identical shared head so every LP baseline
        scores pairs the same way and the decoder choice is validation-selected.
        """

        seed_everything(seed, device)
        data = bundle.data.to(device)
        # feats_type 0 keeps real attributes where a type has them (PubMed, Amazon) and
        # falls back to identity where it does not (LastFM, YouTube).
        view = bundle.homogeneous(feats_type=int(params.get("feats_type", 0)))
        features = _to_device(view.features_list, device)
        # HGB's own link-prediction runner reduces the graph with
        # `gen_edge_list(dl, reverse=True)` (HGB/LP/benchmark/methods/GNN/utils.py:15):
        # sum the relation adjacencies, append the reverse of every edge, then add a
        # self-loop per node. `HomogeneousView.edge_index` is exactly that construction,
        # and it is what the node-classification path in this same file already uses.
        #
        # Reading the directed `data.edge_index` here instead gave GCN/GAT a different
        # message-passing graph from NSD/CSNN, which take the symmetrised view via
        # `_homogeneous_lp`. On Amazon, where a held-out edge can have a training-set
        # counterpart in the opposite direction under the same relation, that is not a
        # cosmetic difference: one reduction can reach the reverse of a validation edge
        # and the other cannot, so the two families were not being scored on the same
        # evidence.
        edge_index = view.edge_index.to(device)

        hidden_dim = int(params.get("hidden_dim", 64))
        num_layers = int(params.get("num_layers", 2))
        dropout = float(params.get("dropout", 0.5))
        decoder_type = str(params.get("decoder_type", "dot"))

        with pushd_sys_path(VENDOR_LP), legacy_conv_attrs():
            from GNN import GAT, GCN  # type: ignore[import-not-found]

            if self.model_type == "gat":
                num_heads = int(params.get("num_heads", 8))
                # Their GAT indexes heads[-2], so it needs at least two entries.
                heads = [num_heads] * max(num_layers - 1, 1) + [1]
                model = GAT(
                    view.in_dims,
                    hidden_dim,
                    n_layers=num_layers,
                    dropout=dropout,
                    heads=heads,
                    decoder=decoder_type,
                    rel_num=data.num_target_relations,
                )
            else:
                model = GCN(
                    view.in_dims,
                    hidden_dim,
                    n_layers=num_layers,
                    dropout=dropout,
                    decoder=decoder_type,
                    rel_num=data.num_target_relations,
                )
            model = model.to(device)

        # GAT concatenates heads, so the embedding width is hidden_dim * heads[-1].
        embed_dim = hidden_dim
        head = SharedLinkHead(
            decoder_type,
            num_relations=max(int(data.num_target_relations), 1),
            dim=embed_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 5e-4)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )

        record = train_link_prediction(
            model=model,
            head=head,
            embed=lambda: model.encode((features, edge_index)),
            data=data,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(LINK_PREDICTION, params),
        )
        record["params_source"] = params.source
        record["decoder_type"] = decoder_type
        record["device"] = device.type
        return record


    def _fit_graph_classification(
        self,
        bundle: Any,
        params: BaselineParams,
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        """Minibatched GCN/GAT over TU graphs with the shared mean-pool head.

        Unlike node classification and link prediction, there is **no vendor
        implementation to reproduce here**: HGB has no graph-classification benchmark and
        neither GCN nor GAT was published on TUDataset by these authors. The encoder is
        therefore a plain stack of PyG ``GCNConv``/``GATConv`` -- the same operators HGB's
        own link-prediction code uses -- and the readout is the campaign's shared
        mean-pool head, identical for every graph-classification baseline. This is our
        construction, not the authors', and the results table should say so.
        """

        from torch_geometric.nn import GATConv, GCNConv

        seed_everything(seed, device)
        hidden_dim = int(params.get("hidden_dim", 64))
        num_layers = int(params.get("num_layers", 2))
        dropout = float(params.get("dropout", 0.5))
        num_heads = int(params.get("num_heads", 8)) if self.model_type == "gat" else 1

        class Encoder(torch.nn.Module):
            def __init__(self, in_dim: int, model_type: str) -> None:
                super().__init__()
                self.convs = torch.nn.ModuleList()
                self.dropout = torch.nn.Dropout(dropout)
                width = in_dim
                for layer in range(num_layers):
                    if model_type == "gat":
                        # Concatenate heads except on the last layer, so the output width
                        # is hidden_dim regardless of depth.
                        concat = layer < num_layers - 1
                        self.convs.append(
                            GATConv(width, hidden_dim, heads=num_heads, concat=concat)
                        )
                        width = hidden_dim * num_heads if concat else hidden_dim
                    else:
                        self.convs.append(GCNConv(width, hidden_dim))
                        width = hidden_dim
                self.out_dim = width

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                for index, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if index < len(self.convs) - 1:
                        x = self.dropout(torch.nn.functional.elu(x))
                return x

        model = Encoder(bundle.in_channels, self.model_type).to(device)
        head = SharedGraphHead(model.out_dim, bundle.num_classes, dropout=dropout).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 0.0)),
        )

        record = train_graph_classification(
            model=model,
            head=head,
            embed_batch=lambda batch: model(batch.x, batch.edge_index),
            loaders=bundle.loaders,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(GRAPH_CLASSIFICATION, params),
        )
        record["params_source"] = params.source
        record["readout"] = "shared_meanpool"
        return record


ADAPTER = {
    "gcn": HGBHomogeneousAdapter("gcn", "gcn"),
    "gat": HGBHomogeneousAdapter("gat", "gat"),
}
