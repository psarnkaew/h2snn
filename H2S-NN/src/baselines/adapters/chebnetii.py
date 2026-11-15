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
#: Card baselines-h2gb C4: H2GB NC datasets with one feature width on every type, where their ogb scheme
#: (`_chebnetii_h2gb.py`: Chebyshev bases once, batches of target rows) runs unchanged.
#: Card baselines-h2gb C4: the H2GB path reads the budget table's batch plan (`minibatch.minibatch_plan`).
READS_BATCH_PLAN = True
H2GB_DATASETS = frozenset({"ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "pdns"})
#: Part 2 E2 (decision C4-c): datasets with several feature widths take HGB's per-type projection, applied after the
#: per-block Chebyshev bases (`_chebnetii_h2gb.block_bases`, `ProjectedBases`). IEEE-CIS-G and H-Pokec have featureless
#: types (identity blocks: a dense N x N_type matrix per k) and are a stop item.
H2GB_PROJECTED = frozenset({"pdns"})


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
        return task == NODE_CLASSIFICATION and (dataset in SUPPORTED_DATASETS or dataset in H2GB_DATASETS)

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, dataset, "node classification only")
        if dataset in H2GB_DATASETS:
            from src.configs.config import MINIBATCH_BATCH_SIZES

            space = {  # their ogb model: K, hidden, dropout (ogb/large_train.py:25-31); B from the shared range
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,
                "hidden": {"values": [256, 512, 1024, 2048]},
                "dropout": grids.DROPOUT,
                "K": {"values": [5, 10]},
                "batch_size": {"values": list(MINIBATCH_BATCH_SIZES)},
            }
            if dataset in H2GB_PROJECTED:
                space["hidden_dim"] = {"values": [64, 128, 256]}  # the projection width, as its HGB adapter
            return space
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
        if bundle.dataset in H2GB_DATASETS:
            return self._fit_h2gb(bundle, params, seed=seed, device=device)

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
        optimizer = torch.optim.AdamW(
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


    def _fit_h2gb(self, bundle: NodeBundle, params: BaselineParams, *, seed: int,
                  device: torch.device) -> dict[str, float]:
        """Their ogb scheme on H2GB's graph (`_chebnetii_h2gb.py`): bases once, batches of B target rows, the shared
        H2GB NC loop."""
        from src.baselines.adapters import _chebnetii_h2gb as h2gb
        from src.baselines.minibatch import h2gb_graph, minibatch_plan, train_index_minibatch
        from src.data.h2gb_nc import ALL_H2GB_NC_SPECS

        seed_everything(seed, device)
        data = h2gb_graph(bundle)
        target = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
        plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
        labels = data[target].y.long()
        num_classes = int(labels.max()) + 1
        K = int(params.get("K", 10))
        if bundle.dataset in H2GB_PROJECTED:
            features, edge_index, slices = h2gb.block_inputs(data)
            start, end = slices[target]
            blocks = h2gb.block_bases(features, edge_index, slices, K, torch.arange(start, end))
            width = int(params.get("hidden_dim", 64))
            projection = h2gb.ProjectedBases([int(f.size(1)) for f in features], width).to(device)
            model = torch.nn.ModuleDict({"projection": projection,
                                         "chebnet": h2gb.build_model(width, num_classes, params)}).to(device)

            def forward_rows(index: torch.Tensor) -> torch.Tensor:
                rows = [[basis[index.cpu()].to(device) for basis in per_type] for per_type in blocks]
                return model["chebnet"](model["projection"](rows), 0, index.numel())
        else:
            x, edge_index, slices = h2gb.homogeneous_inputs(data)
            start, end = slices[target]
            bases = h2gb.chebyshev_bases(x, edge_index, K, torch.arange(start, end))
            model = h2gb.build_model(int(x.size(1)), num_classes, params).to(device)

            def forward_rows(index: torch.Tensor) -> torch.Tensor:
                rows = [basis[index.cpu()].to(device) for basis in bases]
                return model(rows, 0, index.numel())

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 1e-3)),
                                      weight_decay=float(params.get("weight_decay", 0.0)))

        record = train_index_minibatch(
            model=model, forward_rows=forward_rows, labels=labels.to(device),
            train_idx=data[target].train_mask.nonzero().view(-1), val_idx=data[target].val_mask.nonzero().view(-1),
            test_idx=data[target].test_mask.nonzero().view(-1), optimizer=optimizer,
            criterion=torch.nn.functional.nll_loss, plan=plan, base_seed=seed, report=params.report)
        record["params_source"] = params.source
        record["device"] = device.type
        record["batch_size"] = float(plan.batch_size)
        return record


ADAPTER = ChebNetIIAdapter()
