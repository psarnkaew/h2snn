"""Hetero^2Net (Li et al., 2025) -- heterophily-aware heterogeneous representation learning.

This one is PyG-native and reads ``HGBDataset`` directly, so the model, the metapath
dropping, and the auxiliary homophily/heterophily losses are all reused unchanged: the
adapter calls their ``train_full_batch`` and their ``HeteroGNN``.

Two deviations, both recorded on the run record:

1. **The split.** ``hetero2net/dataset.py`` ends with
   ``RandomNodeSplit(num_val=0.2, num_test=0.2)``, which *discards* HGB's official
   train/test masks in favour of a random 60/20/20 partition. On ACM that is roughly
   double the training data HGB allows (60% versus 907 of 3025 labelled papers). Their
   published numbers and hyperparameters therefore come from an easier protocol than the
   rest of this campaign, so the masks are replaced with the project's canonical ones and
   their published hyperparameters are treated as a starting point rather than a
   like-for-like transfer.
2. **Model selection** is on validation macro-F1, uniformly with every other baseline.

Their multi-label rule is already ``(y_pred > 0)`` (hetero2net/utils.py:240), matching the
campaign's ``threshold`` protocol, so IMDB needs no reconciliation here.

Link prediction has no counterpart in their repository -- their ``EdgeDecoder`` is an
auxiliary node-classification regulariser, not a task head -- so all four cells are
tuned. See :mod:`src.baselines.adapters._hetero2net_lp` for what is kept from the
method and the one structural adaptation it needs.

Graph classification presents each minibatch as a single-node-type multiplex
``HeteroData`` keyed by edge type (:mod:`._gc_typed`) and pools with the shared head.
The disentanglement regulariser is kept here too, for the same reason as on link
prediction: their own loop adds it on every step.
"""

from __future__ import annotations

from copy import copy as shallow_copy
from typing import Any

import torch

from src.baselines.adapters._gc_typed import batch_hetero_data, relation_key
from src.baselines.adapters._hetero2net_lp import Hetero2NetLinkEncoder
from src.baselines.bundles import GraphBundle, LinkBundle, NodeBundle
from src.baselines.heads import SharedGraphHead, SharedLinkHead
from src.baselines.metrics import classification_metrics, node_classification_record
from src.baselines.budget import budget_kwargs, resolve_budget
from src.baselines.progress import report_epoch
from src.baselines import search_grids as grids
from src.baselines.protocol import (
    GRAPH_CLASSIFICATION,
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    BaselineParams,
    UnsupportedCell,
)
from src.baselines.published import load_published_params
from src.baselines.runtime import PROJECT_ROOT, pushd_sys_path
from src.data.data_layout import REPO_DATA_ROOT, nc_root
from src.baselines.training import train_graph_classification, train_link_prediction
from src.utils.misc import seed_everything

NAME = "hetero2net"
VENDOR = "hetero2net"

#: Their loader keys datasets by upper-case name.
VENDOR_DATASET = {"acm": "ACM", "dblp": "DBLP", "imdb": "IMDB"}
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "pubmed"})  # YouTube, Amazon removed (spec 1.5)
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})


def _load_vendor_graph(dataset: str) -> tuple[Any, Any]:
    """Run their ``load_dataset`` against our on-disk HGBDataset copy.

    Their call is ``HGBDataset(root=f'{root}/HGBDataset', name=...)``, which does not match
    this project's ``data/<name>/<name>/raw`` layout, so the symbol their module resolves
    is patched to read from our root. Everything else in their loader -- the per-dataset
    feature fill-ins for attribute-less types, the metapath definitions, the
    NormalizeFeatures transform on DBLP -- runs unchanged.
    """

    from torch_geometric.datasets import HGBDataset

    key = dataset.lower()
    root = nc_root(REPO_DATA_ROOT) / key

    with pushd_sys_path(VENDOR):
        from hetero2net import dataset as vendor_dataset  # type: ignore[import-not-found]

        original = vendor_dataset.HGBDataset
        vendor_dataset.HGBDataset = lambda root=None, name=None, transform=None: HGBDataset(
            root=str(nc_root(REPO_DATA_ROOT)),
            name=str(name).upper(),
            transform=transform,
        )
        try:
            return vendor_dataset.load_dataset(VENDOR_DATASET[key])
        finally:
            vendor_dataset.HGBDataset = original


def _apply_canonical_split(data: Any, bundle: NodeBundle, node_type: str) -> None:
    """Replace their RandomNodeSplit masks with the project's canonical ones."""

    target = bundle.data
    expected = int(target.target_y.shape[0])
    actual = int(data[node_type].num_nodes)
    if expected != actual:
        raise RuntimeError(
            f"Hetero2Net loaded {actual} {node_type} nodes for {bundle.dataset}, the "
            f"project bundle has {expected}; the two are not the same graph."
        )
    data[node_type].train_mask = target.target_train_mask.clone()
    data[node_type].val_mask = target.target_val_mask.clone()
    data[node_type].test_mask = target.target_test_mask.clone()


def _build_label_embedding_targets(
    data: Any, node_type: str, num_classes: int, *, mask_test: torch.Tensor,
    mask_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce main.py:129-146 -- the masked label-propagation inputs for --mask_lp.

    Returns ``(train_y_emb, eval_y_emb)``: the training copy additionally hides validation
    labels, so validation never sees its own answers through the label channel.
    """

    labels = data[node_type].y.clone()
    if labels.ndim > 1:
        labels = labels.long()
        rows, cols = labels.nonzero().T
        labels[labels == 0] = num_classes
        labels[rows, cols] = cols
    labels[labels == -1] = num_classes
    labels[mask_test] = num_classes
    eval_y_emb = labels.clone()
    labels[mask_val] = num_classes
    return labels, eval_y_emb


class Hetero2NetAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return dataset in VENDOR_DATASET
        if task == LINK_PREDICTION:
            return dataset in SUPPORTED_LP_DATASETS
        if task == GRAPH_CLASSIFICATION:
            return dataset in SUPPORTED_GC_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task not in (NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION):
            raise UnsupportedCell(NAME, task, dataset, "unsupported task")
        if task == GRAPH_CLASSIFICATION:
            return {
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,
                "hidden_channels": {"values": [64, 128, 256]},
                "embed_dim": {"values": [32, 64, 128]},
                "dropout": grids.DROPOUT,
                "num_layers": {"values": [2, 3]},
                "alpha": {"values": [0.0, 0.1, 0.2]},
            }
        if task == LINK_PREDICTION:
            return {
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,
                "hidden_channels": {"values": [64, 128, 256]},
                "embed_dim": {"values": [16, 64, 128]},
                "dropout": grids.DROPOUT,
                "num_layers": {"values": [2, 3]},
                # Their disentanglement weight; 0 turns the regulariser off, which
                # their own trainer allows.
                "alpha": {"values": [0.0, 0.1, 0.2]},
                "decoder_type": {"values": ["dot", "distmult"]},
            }
        return {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden": grids.HIDDEN,
            "dropout": grids.DROPOUT,
            "alpha": {"values": [0.0, 0.1, 0.2]},
            "beta": {"values": [0.2, 0.4, 0.5]},
            "p": {"values": [0.6, 0.7, 1.0]},
            "r": {"values": [0.3, 0.5]},
            "mask_lp": {"values": [True, False]},
        }

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
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
            return self._fit_link_prediction(bundle, params, seed=seed, device=device)
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "unsupported task")

        seed_everything(seed, device)
        data, metapaths = _load_vendor_graph(bundle.dataset)

        node_types = [t for t in data.node_types if data[t].get("y") is not None]
        if len(node_types) != 1:
            raise RuntimeError(
                f"expected exactly one labelled node type, found {node_types}"
            )
        node_type = node_types[0]
        _apply_canonical_split(data, bundle, node_type)

        num_classes = bundle.num_classes
        data = data.to(device, "x", "y")
        train_mask = data[node_type].train_mask.to(device)
        val_mask = data[node_type].val_mask.to(device)
        test_mask = data[node_type].test_mask.to(device)

        mask_lp = bool(params.get("mask_lp", True))
        train_data = shallow_copy(data)
        eval_data = shallow_copy(data)
        if mask_lp:
            train_y_emb, eval_y_emb = _build_label_embedding_targets(
                data, node_type, num_classes, mask_test=test_mask, mask_val=val_mask
            )
            train_data[node_type].y_emb = train_y_emb
            eval_data[node_type].y_emb = eval_y_emb

        with pushd_sys_path(VENDOR):
            from hetero2net.models import HeteroGNN  # type: ignore[import-not-found]
            from hetero2net.utils import train_full_batch  # type: ignore[import-not-found]

            model = HeteroGNN(
                data.metadata(),
                in_channels=int(data[node_type].x.size(1)),
                hidden_channels=int(params.get("hidden", 128)),
                out_channels=num_classes,
                dropout=float(params.get("dropout", 0.0)),
                project=True,
            ).to(device)
            model.node_type = node_type

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(params.get("lr", 5e-3)),
                weight_decay=float(params.get("weight_decay", 0.0)),
            )

            budget = resolve_budget(NODE_CLASSIFICATION, params)
            epochs, patience = budget.epochs, budget.patience
            best_monitor = float("-inf")
            best_epoch = 0
            best_val: dict[str, float] = {}
            best_test: dict[str, float] = {}
            stale = 0

            for epoch in range(epochs):
                train_full_batch(
                    model,
                    optimizer,
                    train_data,
                    train_mask,
                    device,
                    p=float(params.get("p", 0.7)),
                    metapaths=metapaths,
                    r=float(params.get("r", 0.5)),
                    alpha=float(params.get("alpha", 0.2)),
                    beta=float(params.get("beta", 0.4)),
                )

                model.eval()
                with torch.no_grad():
                    # Their evaluate_full_batch moves edge_index onto the device before
                    # the forward (hetero2net/utils.py:217). Skipping it leaves the
                    # adjacency on CPU while the features are on CUDA, and DisenConv's
                    # fused sparse path then dispatches to a torch.sparse reduction that
                    # has no CUDA kernel.
                    eval_data = eval_data.to(device, "edge_index")
                    logits = model(
                        eval_data.x_dict,
                        eval_data.edge_index_dict,
                        eval_data[node_type].get("y_emb"),
                    )
                    if isinstance(logits, dict):
                        logits = logits[node_type]
                    labels = eval_data[node_type].y
                    val_metrics = classification_metrics(
                        logits[val_mask],
                        labels[val_mask],
                        is_multilabel=bundle.is_multilabel,
                    )
                    test_metrics = classification_metrics(
                        logits[test_mask],
                        labels[test_mask],
                        is_multilabel=bundle.is_multilabel,
                    )

                monitor = val_metrics["macro_f1"]
                if monitor > best_monitor:
                    best_monitor = monitor
                    best_epoch = epoch
                    best_val = val_metrics
                    best_test = test_metrics
                    stale = 0
                else:
                    stale += 1

                report_epoch(
                    params.report,
                    epoch=epoch,
                    monitor=monitor,
                    running_best=best_monitor,
                    metrics=val_metrics,
                )
                if stale >= patience:
                    break

        record = node_classification_record(
            val_metrics=best_val, test_metrics=best_test, best_epoch=best_epoch
        )
        record["params_source"] = params.source
        record["split_note"] = "canonical HGB split replaces vendor RandomNodeSplit"
        return record

    def _fit_link_prediction(
        self,
        bundle: LinkBundle,
        params: BaselineParams,
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        seed_everything(seed, device)

        embed_dim = int(params.get("embed_dim", 64))
        model = Hetero2NetLinkEncoder(
            bundle,
            hidden_channels=int(params.get("hidden_channels", 128)),
            embed_dim=embed_dim,
            dropout=float(params.get("dropout", 0.5)),
            num_layers=int(params.get("num_layers", 2)),
            device=device,
        ).to(device)

        data = bundle.data.to(device)
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

        alpha = float(params.get("alpha", 0.2))
        record = train_link_prediction(
            model=model,
            head=head,
            embed=model,
            data=data,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(LINK_PREDICTION, params),
            aux_loss=lambda: model.disentanglement_loss(alpha),
        )
        record["params_source"] = params.source
        record["alpha"] = alpha
        record["num_endpoint_types"] = float(len(model.endpoint_types))
        record["decoder_type"] = str(params.get("decoder_type", "dot"))
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

        num_relations = int(bundle.fold.dims.num_relations)
        node_type = "node"
        metadata = (
            [node_type],
            [
                (node_type, relation_key(index), node_type)
                for index in range(num_relations)
            ],
        )

        hidden_channels = int(params.get("hidden_channels", 128))
        embed_dim = int(params.get("embed_dim", 64))
        with pushd_sys_path(VENDOR):
            from hetero2net.models import HeteroGNN  # type: ignore[import-not-found]

            model = HeteroGNN(
                metadata,
                in_channels=bundle.in_channels,
                hidden_channels=hidden_channels,
                out_channels=embed_dim,
                dropout=float(params.get("dropout", 0.5)),
                num_layers=int(params.get("num_layers", 2)),
                project=True,
            ).to(device)
        # Set externally in their main.py:186 rather than in __init__.
        model.node_type = node_type

        head = SharedGraphHead(
            embed_dim,
            bundle.num_classes,
            dropout=float(params.get("dropout", 0.0)),
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 0.0)),
        )

        alpha = float(params.get("alpha", 0.2))
        components: list[tuple[dict, dict]] = []

        def embed_batch(batch: Any) -> torch.Tensor:
            hetero = batch_hetero_data(batch, num_relations).to(device)
            out = model(hetero.x_dict, hetero.edge_index_dict)
            components.clear()
            if isinstance(out, tuple):
                out, homos, heteros = out
                components.append((homos, heteros))
            if isinstance(out, dict):
                out = out[node_type]
            return out

        def aux_loss() -> torch.Tensor:
            """alpha * dist_corr(x_homo, x_hetero), as their utils.py:161-167 adds it."""

            total = torch.zeros((), device=device)
            if alpha <= 0 or not components:
                return total
            with pushd_sys_path(VENDOR):
                from hetero2net.utils import dist_corr  # type: ignore[import-not-found]

                for homos, heteros in components:
                    for homo_dict, hetero_dict in zip(homos, heteros):
                        for edge_type in homo_dict:
                            total = total + alpha * dist_corr(
                                homo_dict[edge_type], hetero_dict[edge_type]
                            )
            return total

        record = train_graph_classification(
            model=model,
            head=head,
            embed_batch=embed_batch,
            loaders=bundle.loaders,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(GRAPH_CLASSIFICATION, params),
            aux_loss=aux_loss,
        )
        record["params_source"] = params.source
        record["alpha"] = alpha
        record["num_relation_keys"] = float(num_relations)
        record["readout"] = "shared_mean_pool_linear"
        return record


ADAPTER = Hetero2NetAdapter()
