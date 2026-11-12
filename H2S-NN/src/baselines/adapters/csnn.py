"""CSNN -- Cooperative Sheaf Neural Network (Ribeiro et al., 2026).

Unlike the other baselines, CSNN is part of this repository rather than a clone: it is the
legacy homogeneous model under ``vendor/csnn/`` (``models/`` and ``exp/``), kept for
comparison runs (see CLAUDE.md, "Legacy baseline"). The adapter imports it from there so there is exactly one
copy of the model.

Same two adaptations as ChebNetII and NSD, both recorded on the run record:

1. **Graph reduction** via the shared :class:`~src.baselines.bundles.HomogeneousView`.
   Like NSD, CSNN builds its own sheaf Laplacian, so it takes the self-loop-free form.
2. **Input projection** via :class:`~src.baselines.encoders.TypeProjectedInput`, since
   CSNN's ``lin1`` expects one dense matrix.

Published settings cover the heterophilic node benchmarks and TUDataset
(``vendor/csnn/exp/scripts/``), not ACM/DBLP/IMDB, so node-classification cells here are tuned.
ENZYMES *is* covered by ``exp/scripts/run_enzymes.sh`` and is the campaign's only
published graph-classification cell.

Graph classification is the one task CSNN implements natively: ``graph_level=True``
makes its forward pool with ``global_mean_pool`` and apply its own MLP
(models/coopshv_model.py:61-64). The adapter still runs it with ``graph_level`` off and
attaches the campaign's shared mean-pool + linear head, so the graph-classification
table compares encoders rather than readouts. Since their native readout is also a
mean pool, the deviation amounts to a linear head in place of their MLP.

``orth='householder'`` needs ``torch_householder`` -- installed, and listed in
requirements.txt under the baselines section. The current H2S-NN does not use it; that
package exists in this environment for CSNN alone.
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines.adapters._homogeneous_lp import fit_homogeneous_link_prediction
from src.baselines.bundles import HGB_DEFAULT_FEATS_TYPE, GraphBundle, NodeBundle
from src.baselines.encoders import TypeProjectedInput
from src.baselines.heads import SharedGraphHead
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
)
from src.utils.misc import seed_everything

NAME = "csnn"
VENDOR = "csnn"
SUPPORTED_DATASETS = frozenset(HGB_DEFAULT_FEATS_TYPE)
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "pubmed"})  # YouTube, Amazon removed (spec 1.5)
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})

#: Sheaf-learner backbones CSNN supports (models/sheaf_learner.py).
GNN_TYPES = ("GCN", "SAGE", "GAT", "SGC")
#: Orthogonal parameterisations. `matrix_exp` is rejected by the current H2S-NN but is
#: CSNN's own option and stays available to it.
ORTH_TYPES = ("householder", "cayley", "euler", "matrix_exp")


def _vendor_args(
    params: BaselineParams,
    *,
    input_dim: int,
    output_dim: int,
    graph_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """The dict CSNN's SheafDiffusion base reads (models/sheaf_base.py)."""

    return {
        "d": int(params.get("d", 3)),
        "layers": int(params.get("layers", 2)),
        "hidden_channels": int(params.get("hidden_channels", 32)),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "graph_size": graph_size,
        "device": device,
        "dropout": float(params.get("dropout", 0.0)),
        "input_dropout": float(params.get("input_dropout", 0.0)),
        "left_weights": bool(params.get("left_weights", True)),
        "right_weights": bool(params.get("right_weights", True)),
        "use_act": bool(params.get("use_act", True)),
        "layer_norm": bool(params.get("layer_norm", True)),
        "batch_norm": bool(params.get("batch_norm", False)),
        "edge_weights": bool(params.get("edge_weights", True)),
        "sheaf_act": str(params.get("sheaf_act", "tanh")),
        "orth": str(params.get("orth", "householder")),
        "conformal": bool(params.get("conformal", False)),
        "num_heads": int(params.get("num_heads", 1)),
        "linear_emb": bool(params.get("linear_emb", False)),
        "pe_size": int(params.get("pe_size", 0)),
        "gnn_type": str(params.get("gnn_type", "GCN")),
        "gnn_hidden": int(params.get("gnn_hidden", 32)),
        "gnn_layers": int(params.get("gnn_layers", 2)),
        "gnn_residual": bool(params.get("gnn_residual", False)),
        "gnn_default": bool(params.get("gnn_default", True)),
        "batch_size": int(params.get("batch_size", 1)),
        # Node classification here; the graph-level branch adds a mean pool.
        "task": "node_level",
    }


class _CSNNEncoder(torch.nn.Module):
    """Per-type projection feeding CSNN, returning target-node logits."""

    def __init__(
        self,
        in_dims: list[int],
        edge_index: torch.Tensor,
        args: dict[str, Any],
    ) -> None:
        super().__init__()
        self.input_projection = TypeProjectedInput(in_dims, args["input_dim"])
        self.edge_index = edge_index

        # Constructed inside the vendor context: CSNNConv subclasses PyG's MessagePassing,
        # whose __init__ resolves annotations via sys.modules. See pushd_sys_path.
        with pushd_sys_path(VENDOR):
            from models.coopshv_model import CSNN  # type: ignore[import-not-found]

            self.csnn = CSNN(args)

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        from torch_geometric.data import Data

        x = self.input_projection(features_list)
        # CSNN.forward reassigns `batch.x` in place at every stage, so a Data object
        # reused across epochs would feed the previous epoch's *output* back in as input.
        # Building a fresh Data per forward keeps each pass independent.
        batch = Data(x=x, edge_index=self.edge_index)
        batch.edge_attr = None
        return self.csnn(batch)


class CSNNAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return dataset in SUPPORTED_DATASETS
        if task == LINK_PREDICTION:
            return dataset in SUPPORTED_LP_DATASETS
        if task == GRAPH_CLASSIFICATION:
            return dataset in SUPPORTED_GC_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task not in (NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION):
            raise UnsupportedCell(NAME, task, dataset, "unsupported task")
        space = {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden_channels": {"values": [32, 64]},
            "dropout": grids.DROPOUT,
            "d": {"values": [2, 3, 4]},
            "layers": grids.SHEAF_LAYERS,
            "gnn_type": {"values": ["GCN", "SAGE", "GAT"]},
            "orth": {"values": ["householder", "cayley", "euler"]},
            "conformal": {"values": [True, False]},
            "feats_type": {"values": [0, 2, 3]},
        }
        if task == LINK_PREDICTION:
            space["decoder_type"] = {"values": ["dot", "distmult"]}
        if task == GRAPH_CLASSIFICATION:
            # feats_type is a node-classification notion (HGB's per-type feature
            # fill-in); TU graphs carry one uniform feature matrix.
            space.pop("feats_type", None)
            space["embed_dim"] = {"values": [32, 64, 128]}
            # Centred on their published ENZYMES command.
            space["batch_norm"] = {"values": [True, False]}
            space["layer_norm"] = {"values": [True, False]}
            # Their ENZYMES command runs a 1-layer, 64-wide sheaf-learner backbone; the
            # adapter defaulted to 2 x 32 and never searched either.
            space["gnn_layers"] = {"values": [1, 2, 3]}
            space["gnn_hidden"] = {"values": [32, 64, 128]}
        return space

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        # exp/scripts/ covers heterophilic node benchmarks and TUDataset only.
        return load_published_params(NAME, task, dataset)

    def fit_task(
        self,
        bundle: NodeBundle,
        params: BaselineParams,
        *,
        task: str,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        if task == GRAPH_CLASSIFICATION:
            return self._fit_graph_classification(
                bundle, params, seed=seed, device=device
            )
        if task == LINK_PREDICTION:
            def build(view: Any, edge_index: torch.Tensor, embed_dim: int) -> torch.nn.Module:
                # output_dim becomes the embedding width the link decoder scores in,
                # rather than a class count.
                args = _vendor_args(
                    params,
                    input_dim=int(params.get("hidden_channels", 32)),
                    output_dim=embed_dim,
                    graph_size=view.num_nodes,
                    device=device,
                )
                return _CSNNEncoder(view.in_dims, edge_index, args)

            record = fit_homogeneous_link_prediction(
                NAME, bundle, params, seed=seed, device=device, build_encoder=build
            )
            record["gnn_type"] = str(params.get("gnn_type", "GCN"))
            return record
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "unsupported task")

        seed_everything(seed, device)
        feats_type = int(
            params.get("feats_type", HGB_DEFAULT_FEATS_TYPE[bundle.dataset])
        )
        view = bundle.homogeneous(feats_type=feats_type)
        features = [
            feature.coalesce().to(device) if feature.is_sparse else feature.to(device)
            for feature in view.features_list
        ]
        # CSNN builds its own sheaf Laplacian, as NSD does, so no HGB self-loops.
        edge_index = view.edge_index_without_self_loops.to(device)

        args = _vendor_args(
            params,
            input_dim=int(params.get("hidden_channels", 32)),
            output_dim=bundle.num_classes,
            graph_size=view.num_nodes,
            device=device,
        )
        model = _CSNNEncoder(view.in_dims, edge_index, args).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 1e-2)),
            weight_decay=float(params.get("weight_decay", 5e-4)),
        )

        data = bundle.data
        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()

        record = train_full_batch_node_classification(
            model=model,
            forward=lambda: model(features),
            labels=labels,
            train_idx=data.target_train_mask.nonzero(as_tuple=False).view(-1).to(device),
            val_idx=data.target_val_mask.nonzero(as_tuple=False).view(-1).to(device),
            test_idx=data.target_test_mask.nonzero(as_tuple=False).view(-1).to(device),
            optimizer=optimizer,
            criterion=node_classification_criterion(bundle.is_multilabel),
            is_multilabel=bundle.is_multilabel,
            **budget_kwargs(NODE_CLASSIFICATION, params),
            num_target=int(labels.shape[0]),
        )
        record["params_source"] = params.source
        record["feats_type"] = float(feats_type)
        record["gnn_type"] = args["gnn_type"]
        record["input_projection"] = "per_type_linear"
        return record

    def _fit_graph_classification(
        self,
        bundle: GraphBundle,
        params: BaselineParams,
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        seed_everything(seed, device)

        embed_dim = int(params.get("embed_dim", 64))
        args = _vendor_args(
            params,
            # TU node features are one uniform matrix, so the per-type projection the
            # node-classification path needs does not apply here.
            input_dim=bundle.in_channels,
            output_dim=embed_dim,
            # Placeholder: CSNNConv re-reads the node count from every batch
            # (models/CSNNconv.py:96) before using it, so minibatching is fine.
            graph_size=1,
            device=device,
        )
        # graph_level off: emit node embeddings and let the shared head pool them,
        # so this row shares a readout with every other graph-classification cell.
        args["task"] = "node_level"
        args["batch_size"] = int(params.get("batch_size", 128))

        with pushd_sys_path(VENDOR):
            from models.coopshv_model import CSNN  # type: ignore[import-not-found]

            model = CSNN(args).to(device)

        head = SharedGraphHead(
            embed_dim,
            bundle.num_classes,
            dropout=float(params.get("dropout", 0.0)),
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 2e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )

        def embed_batch(batch: Any) -> torch.Tensor:
            # CSNN casts edge_attr to float but never reads it (nothing in CSNNconv or
            # sheaf_learner touches it), so clearing it keeps this encoder's inputs
            # identical to GCN's and GAT's rather than quietly richer.
            batch.edge_attr = None
            return model(batch)

        record = train_graph_classification(
            model=model,
            head=head,
            embed_batch=embed_batch,
            loaders=bundle.loaders,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(GRAPH_CLASSIFICATION, params),
        )
        record["params_source"] = params.source
        record["gnn_type"] = args["gnn_type"]
        record["readout"] = "shared_mean_pool_linear"
        return record


ADAPTER = CSNNAdapter()
