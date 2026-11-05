"""ChebNetII (He et al., NeurIPS 2022) -- Chebyshev approximation, revisited.

A homogeneous spectral model, so it needs two adaptations to appear in a heterogeneous
table. Both are recorded on the run record:

1. **Graph reduction.** It consumes the shared
   :class:`~src.baselines.bundles.HomogeneousView` -- HGB's summed, symmetrised,
   self-looped, untyped graph -- the identical graph GCN, GAT, NSD and CSNN see.
2. **Input projection.** ChebNetII takes one dense feature matrix, so the per-type
   features are projected by :class:`~src.baselines.encoders.TypeProjectedInput`, which is
   HGB's own ``fc_list`` construction.

The published hyperparameters cover Cora/Chameleon/WebKB, not ACM/DBLP/IMDB
(``ChebNetII/main/full.sh``), so every cell here is tuned rather than inherited.

Their ``forward`` ends in ``log_softmax`` (``main/models.py:163``). The campaign's shared
loss and metric path expects raw logits -- and log-probabilities are meaningless under the
BCE loss that multi-label IMDB requires -- so the propagation is driven through their own
layers in their own order, stopping one step short of the final normalisation. Nothing
about the model changes; ``argmax`` is identical either way.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.baselines.bundles import HGB_DEFAULT_FEATS_TYPE, NodeBundle
from src.baselines.encoders import TypeProjectedInput
from src.baselines.budget import budget_kwargs
from src.baselines import search_grids as grids
from src.baselines.protocol import (
    NODE_CLASSIFICATION,
    BaselineParams,
    UnsupportedCell,
)
from src.baselines.published import load_published_params
from src.baselines.runtime import pushd_sys_path
from src.baselines.training import (
    node_classification_criterion,
    train_full_batch_node_classification,
)
from src.utils.misc import seed_everything

NAME = "chebnetii"
VENDOR = "chebnetii"
SUPPORTED_DATASETS = frozenset(HGB_DEFAULT_FEATS_TYPE)


class _ChebNetIIEncoder(torch.nn.Module):
    """Per-type projection followed by ChebNetII's own layers, returning logits."""

    def __init__(
        self,
        in_dims: list[int],
        hidden_dim: int,
        num_classes: int,
        *,
        K: int,
        dropout: float,
        dprate: float,
    ) -> None:
        super().__init__()
        self.input_projection = TypeProjectedInput(in_dims, hidden_dim)
        self.dropout = dropout
        self.dprate = dprate

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, num_classes)

        # Constructed *inside* the vendor context, not merely imported there. PyG's
        # MessagePassing.__init__ inspects the subclass's `message` signature and resolves
        # its annotations through `sys.modules[cls.__module__]`; once pushd_sys_path
        # evicts the vendor module that lookup raises KeyError('ChebnetII_pro').
        with pushd_sys_path(VENDOR):
            from ChebnetII_pro import ChebnetII_prop  # type: ignore[import-not-found]

            self.prop1 = ChebnetII_prop(K)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(
        self, features_list: list[torch.Tensor], edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_projection(features_list)
        # ChebNetII/main/models.py:147-163, minus the trailing log_softmax.
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate > 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        return self.prop1(x, edge_index)


class ChebNetIIAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        return task == NODE_CLASSIFICATION and dataset in SUPPORTED_DATASETS

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, dataset, "node classification only")
        return {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden_dim": {"values": [64, 128, 256]},
            "dropout": grids.DROPOUT,
            "K": {"values": [5, 10]},
            "prop_lr": grids.PROP_LR,
            "prop_wd": {"values": [0.0, 5e-4]},
            "dprate": {"values": [0.0, 0.5, 0.9]},
            "feats_type": {"values": [0, 2, 3]},
        }

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        # ChebNetII/main/full.sh covers Cora/Chameleon/WebKB only.
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
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "node classification only")

        seed_everything(seed, device)
        feats_type = int(
            params.get("feats_type", HGB_DEFAULT_FEATS_TYPE[bundle.dataset])
        )
        view = bundle.homogeneous(feats_type=feats_type)
        features = [
            feature.coalesce().to(device) if feature.is_sparse else feature.to(device)
            for feature in view.features_list
        ]
        edge_index = view.edge_index.to(device)

        model = _ChebNetIIEncoder(
            view.in_dims,
            int(params.get("hidden_dim", 64)),
            bundle.num_classes,
            K=int(params.get("K", 10)),
            dropout=float(params.get("dropout", 0.5)),
            dprate=float(params.get("dprate", 0.0)),
        ).to(device)

        # ChebNetII/main/training.py gives the propagation coefficients their own
        # learning rate and weight decay, separate from the linear layers.
        learning_rate = float(params.get("lr", 1e-2))
        weight_decay = float(params.get("weight_decay", 0.0))
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(model.input_projection.parameters())
                    + list(model.lin1.parameters())
                    + list(model.lin2.parameters()),
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                },
                {
                    "params": model.prop1.parameters(),
                    "lr": float(params.get("prop_lr", learning_rate)),
                    "weight_decay": float(params.get("prop_wd", 0.0)),
                },
            ]
        )

        data = bundle.data
        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()

        record = train_full_batch_node_classification(
            model=model,
            forward=lambda: model(features, edge_index),
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
        record["input_projection"] = "per_type_linear"
        return record


ADAPTER = ChebNetIIAdapter()
