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
from typing import Any, Callable

from pathlib import Path

import torch

from src.baselines.adapters._gc_typed import batch_hetero_data, relation_key
from src.baselines.adapters._hetero2net_lp import Hetero2NetLinkEncoder
from src.baselines.bundles import GraphBundle, LinkBundle, NodeBundle, bundle_data_root
from src.baselines.heads import SharedGraphHead, SharedLinkHead
from src.baselines.metrics import classification_metrics, node_classification_record
from src.baselines.budget import budget_kwargs, resolve_budget
from src.baselines.progress import prefixed_metrics, report_epoch
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
from src.data.data_layout import nc_root
from src.baselines.training import train_graph_classification, train_link_prediction
from src.utils.misc import seed_everything

NAME = "hetero2net"
VENDOR = "hetero2net"

#: Their loader keys datasets by upper-case name.
VENDOR_DATASET = {"acm": "ACM", "dblp": "DBLP", "imdb": "IMDB", "freebase": "FreeBase"}  # dataset.py:48
# YouTube, Amazon removed (spec 1.5); the derived tasks wired in Part P P2.
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "pubmed", "acm_pa", "acm_pp", "dblp_ap", "imdb_ma", "freebase_bb"})
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})
#: Card 1c F4 (G-5 amended): the H2GB NC datasets with their own NeighborLoader scheme (`_hetero2net_h2gb.py`);
#: ogbn-mag / mag-year wait for the author's `mag_metapath2vec_emb.zip`.
H2GB_DATASETS = frozenset({"ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"})
H2GB_RUNNABLE = H2GB_DATASETS  # card 1d H3: the author's mag_metapath2vec_emb.zip is in place
READS_BATCH_PLAN = True
#: Card 1c F7 (G-10): large LP with its HeteroGNN (as its local-LP encoder: y_emb none, disentanglement term) on
#: link-level batches of the message graph with its own fan-out [10, 10] (D2); inputs as its NC path (FreeBase
#: branch; MAG waits for the author's embedding file).
LARGE_LP_DATASETS = frozenset({"ogbn-mag_pp", "h-pokec_uu"})


def _load_vendor_graph(dataset: str, data_root: Path) -> tuple[Any, Any]:
    """Run their ``load_dataset`` against our on-disk HGBDataset copy.

    Their call is ``HGBDataset(root=f'{root}/HGBDataset', name=...)``, which does not match
    this project's ``data/<name>/<name>/raw`` layout, so the symbol their module resolves
    is patched to read from our root. Everything else in their loader -- the per-dataset
    feature fill-ins for attribute-less types, the metapath definitions, the
    NormalizeFeatures transform on DBLP -- runs unchanged.
    """

    from torch_geometric.datasets import HGBDataset

    key = dataset.lower()
    root = nc_root(data_root) / key

    with pushd_sys_path(VENDOR):
        from hetero2net import dataset as vendor_dataset  # type: ignore[import-not-found]

        original = vendor_dataset.HGBDataset
        vendor_dataset.HGBDataset = lambda root=None, name=None, transform=None: HGBDataset(
            root=str(nc_root(data_root)),
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


def evaluation_logits(model: Any, train_data: Any, eval_data: Any, node_type: str,
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(validation logits, test logits) as their full-batch evaluation (`Hetero2Net/main.py:215-219`,
    `hetero2net/utils.py:212-224`): validation from `train_data` (validation and test labels hidden in `y_emb`), test
    from `eval_data` (their `test_data`: test labels hidden). Each copy's `edge_index` is moved to the device first
    (`utils.py:217`)."""

    def run(data: Any) -> torch.Tensor:
        data = data.to(device, "edge_index")
        out = model(data.x_dict, data.edge_index_dict, data[node_type].get("y_emb"))
        out = out[0] if isinstance(out, tuple) else out
        return out[node_type] if isinstance(out, dict) else out

    return run(train_data), run(eval_data)


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
            return dataset in VENDOR_DATASET or dataset in H2GB_RUNNABLE
        if task == LINK_PREDICTION:
            return dataset in SUPPORTED_LP_DATASETS or dataset in LARGE_LP_DATASETS
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
        space = {
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
        if dataset in H2GB_DATASETS:  # card 1c F4: batch size from the shared range
            from src.configs.config import MINIBATCH_BATCH_SIZES

            space["batch_size"] = {"values": list(MINIBATCH_BATCH_SIZES)}
        return space

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
        if task == LINK_PREDICTION and bundle.dataset in LARGE_LP_DATASETS:
            from src.baselines.large_lp_training import fit_large_lp

            record = fit_large_lp(bundle, params, seed=seed, device=device,
                                  build=lambda d: build_large_lp(d, params, device))
            record["params_source"] = params.source
            return record
        if task == LINK_PREDICTION:
            return self._fit_link_prediction(bundle, params, seed=seed, device=device)
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "unsupported task")
        if bundle.dataset in H2GB_DATASETS:
            return self._fit_h2gb(bundle, params, seed=seed, device=device)

        seed_everything(seed, device)
        data, metapaths = _load_vendor_graph(bundle.dataset, bundle_data_root(bundle))

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
                train_loss = train_full_batch(
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
                    # Card 1d H2 (author decision D2): validation on the copy hiding validation and test
                    # labels (their train_data), test on the copy hiding test labels (their test_data),
                    # as Hetero2Net/main.py:217-219 evaluates; before, validation read eval_data, whose
                    # y_emb showed each validation target its own label (card 1c finding 2).
                    val_logits, logits = evaluation_logits(model, train_data, eval_data, node_type, device)
                    labels = eval_data[node_type].y
                    val_metrics = classification_metrics(
                        val_logits[val_mask],
                        labels[val_mask],
                        is_multilabel=bundle.is_multilabel,
                    )
                    test_metrics = classification_metrics(
                        logits[test_mask],
                        labels[test_mask],
                        is_multilabel=bundle.is_multilabel,
                    )
                    # card 8 C7: logged only (their loss rule, utils.py:151-154)
                    val_true = labels[val_mask]
                    val_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
                        val_logits[val_mask], val_true.float()) if val_true.ndim > 1 else
                        torch.nn.functional.cross_entropy(val_logits[val_mask], val_true))

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
                    extra={"train/loss": float(train_loss), "val/loss": val_loss,
                           **prefixed_metrics("test", test_metrics)},
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


    def _fit_h2gb(self, bundle: NodeBundle, params: BaselineParams, *, seed: int,
                  device: torch.device) -> dict[str, float]:
        """Card 1c F4: their NeighborLoader scheme on H2GB's graph (`_hetero2net_h2gb.py`)."""
        from copy import copy

        from src.baselines.adapters import _hetero2net_h2gb as h2gb
        from src.baselines.minibatch import h2gb_graph, minibatch_plan
        from src.data.h2gb_nc import ALL_H2GB_NC_SPECS
        from src.evaluation.validation_subset import validation_subset

        seed_everything(seed, device)
        data = h2gb_graph(bundle)
        node_type = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
        metapaths = h2gb.prepare_inputs(data, bundle.dataset, bundle_data_root(bundle))
        num_classes = int(data[node_type].y.max()) + 1
        train_data, test_data = copy(data), copy(data)
        if bool(params.get("mask_lp", True)):
            train_y, test_y = h2gb.label_channel(data, node_type, num_classes, data[node_type].val_mask,
                                                 data[node_type].test_mask)
            train_data[node_type].y_emb, test_data[node_type].y_emb = train_y, test_y
        plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
        train_loader, val_loader, test_loader = h2gb.loaders(train_data, test_data, node_type, plan.batch_size)
        model = h2gb.build_model(data, node_type, num_classes, params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 5e-3)),
                                      weight_decay=float(params.get("weight_decay", 0.0)))

        def evaluate(loader: Any) -> dict[str, float]:
            model.eval()
            logits, labels = [], []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device, "x", "edge_index", "y_emb")  # card 22 A: y_emb with the model
                    logits.append(h2gb.seed_logits(model, batch, node_type))
                    labels.append(batch[node_type].y[: int(batch[node_type].batch_size)].to(device))
            return classification_metrics(torch.cat(logits), torch.cat(labels), is_multilabel=False)

        # Card 1f U5.4 (B1a, decision (a)): the subset of the run seed's split, on train_data as their validation.
        subset_ids = validation_subset(bundle.dataset, data[node_type].val_mask.nonzero().view(-1),
                                       bundle.metadata.get("seed"))
        subset_loader = h2gb.index_loader(train_data, node_type, subset_ids, plan.batch_size)
        state: dict = {}

        def on_device(batches: Any) -> Any:
            """Card 22 A: their `main.py:105-108` puts `y` (and so `y_emb`) on the device before building the loaders;
            their step moves only `x` and `edge_index` (`utils.py:25`). Here the graph stays on the CPU (the loaders'
            workers sample it), so each batch's `y` and `y_emb` move in the main process."""
            for batch in batches:
                yield batch.to(device, "y", "y_emb")

        with pushd_sys_path(VENDOR):
            from hetero2net.utils import train_mini_batch  # type: ignore[import-not-found]

            def train_epoch(epoch: int) -> float:
                return train_mini_batch(model, optimizer, on_device(h2gb.epoch_batches(train_loader, state,
                                                                                    plan.steps_per_epoch)),
                                 device, p=float(params.get("p", 0.7)), metapaths=metapaths,
                                 r=float(params.get("r", 0.5)), alpha=float(params.get("alpha", 0.2)),
                                 beta=float(params.get("beta", 0.4)))

            record = train_h2gb_epochs(model=model, plan=plan, train_epoch=train_epoch, evaluate=evaluate,
                                       subset_loader=subset_loader, subset_ids=subset_ids, val_loader=val_loader,
                                       test_loader=test_loader, report=params.report)
        record["params_source"] = params.source
        record["batch_size"] = float(plan.batch_size)
        record["metapaths"] = float(metapaths is not None)
        return record


def train_h2gb_epochs(*, model: torch.nn.Module, plan: Any, train_epoch: Callable[[int], float | None],
                      evaluate: Callable[[Any], dict[str, float]], subset_loader: Any, subset_ids: torch.Tensor,
                      val_loader: Any, test_loader: Any, report: Any = None) -> dict[str, Any]:
    """Hetero2Net's H2GB NC epoch loop under the U2 rule (card 1f U5.4): at H2GB's eval epochs the B1a validation
    subset only (no test pass); the best subset macro-F1 epoch's weights (CPU copy, ties: earliest) restored after the
    last epoch; one whole-split validation and test pass; early stopping by `stops_early` (card 16 part 2)."""
    from src.baselines.minibatch import _cpu_state, _subset_fields, is_eval_epoch, stops_early

    best_monitor, best_epoch = float("-inf"), -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(plan.epochs):
        train_loss = train_epoch(epoch)
        if not is_eval_epoch(epoch, plan):
            continue
        sub_metrics = evaluate(subset_loader)
        if sub_metrics["macro_f1"] > best_monitor:
            best_monitor, best_epoch, best_state = sub_metrics["macro_f1"], epoch, _cpu_state(model)
        report_epoch(report, epoch=epoch, monitor=sub_metrics["macro_f1"], running_best=best_monitor,
                     metrics=sub_metrics, prefix="val_subset",  # card 8 C7
                     extra={"train/loss": None if train_loss is None else float(train_loss)})
        if stops_early(epoch, best_epoch, plan):
            break
    if best_state is None:
        raise RuntimeError("mini-batch training produced no evaluated epoch.")
    model.load_state_dict(best_state)
    record: dict[str, Any] = node_classification_record(val_metrics=evaluate(val_loader),
                                                        test_metrics=evaluate(test_loader), best_epoch=best_epoch)
    record.update(_subset_fields(subset_ids, best_monitor))
    return record


def build_large_lp(d: Any, params: BaselineParams, device: torch.device, sampler: Any = None) -> tuple:
    """Card 1c F7: their HeteroGNN emitting `embed_dim` rows of the endpoint type (as `Hetero2NetLinkEncoder`), its
    disentanglement term as the auxiliary loss (their utils.py:161-167), on link batches with fan-out [10, 10]."""
    from src.baselines.adapters import _hetero2net_h2gb as h2gb
    from src.baselines.large_lp_training import LargeLPModel, Sampler
    from src.data.large_lp import LARGE_LP_TASKS

    graph, node_type = d.graph, d.node_type
    h2gb.prepare_inputs(graph, LARGE_LP_TASKS[d.task].nc_dataset, d.data_root)  # MAG: the author's zip (card 1d H3)
    sampler = sampler or Sampler("link", list(h2gb.NUM_NEIGHBORS))
    embed_dim = int(params.get("embed_dim", 64))
    hidden = int(params.get("hidden_channels", 128))
    alpha = float(params.get("alpha", 0.2))
    with pushd_sys_path(VENDOR):
        from hetero2net.models import HeteroGNN  # type: ignore[import-not-found]

        model = HeteroGNN(graph.metadata(), in_channels=hidden, hidden_channels=hidden, out_channels=embed_dim,
                          dropout=float(params.get("dropout", 0.0)), project=True)
    model.node_type = node_type
    components: list = []

    def embed(batch: Any, train: bool) -> torch.Tensor:
        batch = batch.to(device, "x", "edge_index")
        out = model(dict(batch.x_dict), batch.edge_index_dict)
        components.clear()
        if isinstance(out, tuple):
            out, homos, heteros = out
            components.append((homos, heteros))
        return out[node_type] if isinstance(out, dict) else out

    def aux_loss() -> torch.Tensor:
        total = torch.zeros((), device=device)
        if alpha <= 0:
            return total
        with pushd_sys_path(VENDOR):
            from hetero2net.utils import dist_corr  # type: ignore[import-not-found]

            for homos, heteros in components:
                for homo_dict, hetero_dict in zip(homos, heteros):
                    for edge_type in homo_dict:
                        total = total + alpha * dist_corr(homo_dict[edge_type], hetero_dict[edge_type])
        return total

    return LargeLPModel(model, embed, embed_dim, aux_loss), sampler


ADAPTER = Hetero2NetAdapter()
