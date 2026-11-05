"""H2SGNN (Lu et al., 2026) -- heterophily-aware heterogeneous spectral GNN.

Same shape as PSHGCN: the authors' ``processing.load_dataset`` runs verbatim and only the
split and the selection criterion are replaced. See
:mod:`src.baselines.adapters._hgb_vendor`.

Unlike PSHGCN, H2SGNN consumes only the target type's features
(``features_list[0]`` in H2SGNN/model.py:71) and propagates through learned combinations
of the relation adjacencies, so its input projection is a single ``nn.Linear`` rather than
one per node type.

Link prediction reuses that same model through
:mod:`src.baselines.adapters._h2sgnn_lp`, which supplies the target block and the
length-2 meta-path operators their loader would have built. Two deviations beyond the
node-classification ones, both on the run record:

* **The target block** is the set of node types the target relation connects, rather
  than a single classified type -- ``user`` and ``artist`` on LastFM, ``disease``
  alone on PubMed.
* **The campaign's shared decoder** scores the pairs, so the table compares encoders.
  The authors publish no link-prediction setting for any of these datasets, so all
  four cells are tuned.

Graph classification uses the same model with the whole minibatch as the target block
(there is no sub-population to crop to) and meta-paths formed from edge types rather
than node-type pairs -- see :mod:`._gc_typed`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from src.baselines.adapters._hgb_vendor import (
    VENDOR_DATASET,
    canonical_split,
    load_vendor_dataset,
    verify_labels,
)
from src.baselines.adapters._gc_typed import batch_metapath_operators
from src.baselines.adapters._gc_typed import relation_key as _gc_relation_key
from src.baselines.adapters._h2sgnn_lp import (
    H2SGNNLinkEncoder,
    build_metapath_operators,
    target_features,
)
from src.baselines.adapters._target_block import target_node_range
from src.baselines.adapters._h2sgnn_lp import parameter_groups as lp_parameter_groups
from src.baselines.bundles import GraphBundle, LinkBundle, NodeBundle
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

NAME = "h2sgnn"
VENDOR = "h2sgnn"
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "youtube", "amazon", "pubmed"})
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})

#: Polynomial bases H2SGNN implements (H2SGNN/model.py:80-86).
POLY_CHOICES = ("GPR_GPR", "GPR_legendre", "GPR_jaco")


def _vendor_args(
    dataset: str, params: BaselineParams, *, device: torch.device
) -> SimpleNamespace:
    poly = str(params.get("poly", "GPR_GPR"))
    if poly not in POLY_CHOICES:
        raise ValueError(f"poly must be one of {POLY_CHOICES}, got {poly!r}.")
    return SimpleNamespace(
        # VENDOR_DATASET maps node-classification datasets only; link-prediction
        # names pass through. The field is read solely by their forward's
        # `if self.dataset in ['IMDB','AMiner']` normalisation branch.
        dataset=VENDOR_DATASET.get(dataset, dataset),
        device=device,
        K=int(params.get("K", 10)),
        emb_dim=int(params.get("emb_dim", 64)),
        hidden=int(params.get("hidden", 32)),
        dropout=float(params.get("dropout", 0.0)),
        input_drop=float(params.get("input_drop", 0.0)),
        lr=float(params.get("lr", 0.01)),
        wd=float(params.get("weight_decay", 0.0)),
        temp_lr=float(params.get("temp_lr", 0.01)),
        temp_wd=float(params.get("temp_wd", 1.0)),
        temp_list_lr=float(params.get("temp_list_lr", 0.01)),
        temp_list_wd=float(params.get("temp_list_wd", 1.0)),
        beta_lr=float(params.get("beta_lr", 0.01)),
        beta_wd=float(params.get("beta_wd", 1.0)),
        alpha=float(params.get("alpha", 0.1)),
        alpha_list=float(params.get("alpha_list", 0.1)),
        a=float(params.get("a", 0.0)),
        b=float(params.get("b", -0.25)),
        poly=poly,
        # Not dataset-specific in their config; H2SGNN/main.py argparse default.
        u=float(params.get("u", 0.5)),
    )


class H2SGNNAdapter:
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
            # At most 16 meta-paths here (MUTAG's 4 bond types squared), so K can go
            # back up towards their published node-classification range.
            return {
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,
                "hidden": grids.HIDDEN,
                "emb_dim": {"values": [64, 128]},
                "embed_dim": {"values": [32, 64, 128]},
                "dropout": grids.DROPOUT,
                "input_drop": grids.INPUT_DROP,
                "K": {"values": [4, 6, 10]},
                "poly": {"values": list(POLY_CHOICES)},
                "alpha": {"values": [0.1, 0.2]},
                "alpha_list": {"values": [0.1, 0.2, 0.8]},
                "temp_lr": {"values": [1e-3, 1e-2, 5e-2]},
                "temp_wd": grids.TEMP_WD,
                "beta_lr": {"values": [1e-5, 1e-3, 1e-2]},
                "beta_wd": {"values": [0.0, 0.3, 1.0]},
            }
        if task == LINK_PREDICTION:
            # K is capped below the node-classification range: each meta-path
            # operator is applied K times per forward, and YouTube has 25 of them.
            return {
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,
                "hidden": grids.HIDDEN,
                "dropout": grids.DROPOUT,
                "emb_dim": {"values": [64, 128, 256]},
                "embed_dim": {"values": [16, 64, 128]},
                "input_drop": grids.INPUT_DROP,
                "K": {"values": [2, 4, 6]},
                "poly": {"values": list(POLY_CHOICES)},
                "alpha": {"values": [0.1, 0.2]},
                "alpha_list": {"values": [0.1, 0.2, 0.8]},
                "temp_lr": {"values": [1e-3, 1e-2, 5e-2]},
                "temp_wd": grids.TEMP_WD,
                "beta_lr": {"values": [1e-5, 1e-3, 1e-2]},
                "beta_wd": {"values": [0.0, 0.3, 1.0]},
                "decoder_type": {"values": ["dot", "distmult"]},
            }
        return {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden": grids.HIDDEN,
            "dropout": grids.DROPOUT,
            "emb_dim": {"values": [64, 128, 256]},
            "input_drop": grids.INPUT_DROP,
            "K": {"values": [6, 10]},
            "poly": {"values": list(POLY_CHOICES)},
            "alpha": {"values": [0.1, 0.2]},
            "alpha_list": {"values": [0.1, 0.2, 0.8]},
            "temp_lr": {"values": [1e-3, 1e-2, 5e-2]},
            "temp_wd": grids.TEMP_WD,
            "beta_lr": {"values": [1e-5, 1e-3, 1e-2]},
            "beta_wd": {"values": [0.0, 0.3, 1.0]},
            # Their per-dataset config sets all four (H2SGNN/main.py:95-102 and their
            # config file); leaving them at one fixed value made every published
            # node-classification setting unreachable by the search.
            "temp_list_lr": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 5e-2,
            },
            "temp_list_wd": grids.TEMP_WD,
            "a": {"distribution": "uniform", "min": -1.0, "max": 0.5},
            "b": {"distribution": "uniform", "min": -0.5, "max": 1.5},
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

        args = _vendor_args(bundle.dataset, params, device=device)
        (
            feature_list,
            adjs,
            lis,
            lis_t,
            labels,
            num_classes,
            _train_idx,
            _val_idx,
            vendor_test_idx,
            adjs_list,
        ) = load_vendor_dataset(VENDOR, bundle.dataset, args)

        verify_labels(bundle, labels, vendor=NAME)
        train_idx, val_idx, test_idx = canonical_split(
            bundle, vendor_test_idx, vendor=NAME
        )

        # Seed every RNG, not just torch: H2SGNN builds its temperature tensors through
        # numpy (model.py:36-42), so seeding torch alone leaves the run irreproducible.
        seed_everything(seed, device)
        with pushd_sys_path(VENDOR):
            from model import H2SGNN  # type: ignore[import-not-found]

            feature_list = [feature.to(device) for feature in feature_list]
            model = H2SGNN(
                int(feature_list[0].shape[1]), num_classes, lis, lis_t, args
            ).to(device)

        # Six parameter groups, as H2SGNN/main.py:23-29: the two temperature sets and the
        # relation-mixing coefficients each get their own lr/weight decay.
        optimizer = torch.optim.AdamW(
            [
                {"params": model.feat_project.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.lin1.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.lin2.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.temp, "weight_decay": args.temp_wd, "lr": args.temp_lr},
                {"params": model.temp_list, "weight_decay": args.temp_list_wd, "lr": args.temp_list_lr},
                {"params": model.beta, "weight_decay": args.beta_wd, "lr": args.beta_lr},
            ]
        )

        record = train_full_batch_node_classification(
            model=model,
            forward=lambda: model(adjs, feature_list, adjs_list),
            labels=labels.to(device),
            train_idx=train_idx.to(device),
            val_idx=val_idx.to(device),
            test_idx=test_idx.to(device),
            optimizer=optimizer,
            criterion=node_classification_criterion(bundle.is_multilabel),
            is_multilabel=bundle.is_multilabel,
            **budget_kwargs(NODE_CLASSIFICATION, params),
            num_target=int(feature_list[0].shape[0]),
        )
        record["params_source"] = params.source
        return record

    def _fit_link_prediction(
        self,
        bundle: LinkBundle,
        params: BaselineParams,
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        # numpy builds the temperature tensors (model.py:36-42), so torch-only
        # seeding would leave every run a fresh init.
        seed_everything(seed, device)

        args = _vendor_args(bundle.dataset, params, device=device)
        node_range = target_node_range(bundle)
        features = target_features(bundle, node_range, device)
        operators, metapath_names = build_metapath_operators(
            bundle, node_range, device
        )

        embed_dim = int(params.get("embed_dim", 64))
        data = bundle.data.to(device)
        model = H2SGNNLinkEncoder(
            int(features.shape[1]),
            embed_dim,
            metapath_names,
            args,
            num_nodes=int(data.x.size(0)),
            node_range=node_range,
        ).to(device)
        head = SharedLinkHead(
            str(params.get("decoder_type", "dot")),
            num_relations=max(int(data.num_target_relations), 1),
            dim=embed_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            lp_parameter_groups(model, head, args, params)
        )

        record = train_link_prediction(
            model=model,
            head=head,
            embed=lambda: model(features, operators),
            data=data,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(LINK_PREDICTION, params),
        )
        record["params_source"] = params.source
        record["num_metapaths"] = float(len(operators))
        record["decoder_type"] = str(params.get("decoder_type", "dot"))
        record["poly"] = args.poly
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

        args = _vendor_args(bundle.dataset, params, device=device)
        num_relations = int(bundle.fold.dims.num_relations)
        # Meta-path names are fixed by the dataset's edge types, so the model can be
        # built once: len(lis) sizes beta and temp_list, and the operators themselves
        # are rebuilt per batch.
        metapath_names = [
            _gc_relation_key(first) + _gc_relation_key(second)
            for first in range(num_relations)
            for second in range(num_relations)
        ]

        embed_dim = int(params.get("embed_dim", 64))
        with pushd_sys_path(VENDOR):
            from model import H2SGNN  # type: ignore[import-not-found]

            model = H2SGNN(
                bundle.in_channels, embed_dim, metapath_names, {}, args
            ).to(device)

        head = SharedGraphHead(
            embed_dim,
            bundle.num_classes,
            dropout=float(params.get("dropout", 0.0)),
        ).to(device)

        # Their six-group AdamW layout, plus the shared head.
        optimizer = torch.optim.AdamW(
            [
                {"params": model.feat_project.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.lin1.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.lin2.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": head.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.temp, "weight_decay": args.temp_wd, "lr": args.temp_lr},
                {"params": model.temp_list.parameters(), "weight_decay": args.temp_list_wd, "lr": args.temp_list_lr},
                {"params": model.beta, "weight_decay": args.beta_wd, "lr": args.beta_lr},
            ]
        )

        def embed_batch(batch: Any) -> torch.Tensor:
            operators, _ = batch_metapath_operators(batch, num_relations, device)
            # Their forward ignores its first argument; pass the list it does read.
            return model(operators, [batch.x.float()], operators)

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
        record["num_metapaths"] = float(len(metapath_names))
        record["num_relation_keys"] = float(num_relations)
        record["poly"] = args.poly
        record["readout"] = "shared_mean_pool_linear"
        return record


ADAPTER = H2SGNNAdapter()
