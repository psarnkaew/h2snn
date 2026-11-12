"""HAN (Wang et al., 2019) as packaged by the HGB benchmark.

HAN is meta-path based, so unlike GCN/GAT it needs the *typed* graph: their loader builds
a DGL heterograph plus a hardcoded meta-path list per dataset
(``HGB/NC/benchmark/methods/HAN/utils.py``, e.g. ACM's ``[['pp','ps','sp'], ['pa','ap'],
...]``). Those meta-paths are the model's inductive bias, so their loader runs unchanged
and only the split is replaced.

Note their validation carve-out is **30% of the training pool, taken sequentially**
(``utils.py:157-163``) -- not HGB's own 20%, and not PSHGCN's 20% either. Three vendors,
three different validation rules, which is why the campaign overrides all of them with one
canonical split.

Deviations, both recorded on the run record:

1. **The split** is the project's canonical HGB carve-out.
2. **Model selection** is on validation macro-F1, uniformly.

Link prediction is the campaign's construction -- HGB's link-prediction benchmark has
no HAN, and the authors publish no link-prediction setting -- so all four cells are
tuned. The model runs as published; :mod:`src.baselines.adapters._han_lp` supplies the
meta-path graph over the target block and the shared decoder scores the pairs.

Graph classification is likewise the campaign's construction, with meta-paths built from
edge types over the whole minibatch -- see :mod:`._gc_typed`. Note the consequence there:
on PROTEINS, ENZYMES and NCI1 a single edge type leaves HAN one meta-path, so its
semantic attention has nothing to weigh.
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines.adapters._gc_typed import batch_metapath_operators
from src.baselines.adapters._gc_typed import relation_key as _gc_relation_key
from src.baselines.adapters._han_lp import HANLinkEncoder, build_metapath_graph
from src.baselines.adapters._target_block import target_node_range
from src.baselines.bundles import (
    GraphBundle,
    LinkBundle,
    NodeBundle,
    hgb_raw_dir,
)
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
from src.baselines.runtime._dgl_compat import resolve_dgl_device
from src.baselines.training import (
    node_classification_criterion,
    train_full_batch_node_classification,
    train_graph_classification,
    train_link_prediction,
)
from src.utils.misc import seed_everything

NAME = "han"
# `hgb_nc` supplies the `scripts.data_loader` package their utils imports via a relative
# sys.path append; `hgb_nc_han` supplies `utils` and `model_hetero` themselves.
VENDORS = ("hgb_nc_han", "hgb_nc")
VENDOR_DATASET = {"acm": "ACM", "dblp": "DBLP", "imdb": "IMDB"}
# YouTube, Amazon removed (spec 1.5); the derived tasks wired in Part P P2.
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "pubmed", "acm_pa", "dblp_ap", "imdb_ma", "freebase_bb"})
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})


def _load_vendor_graph(dataset: str) -> tuple[Any, ...]:
    """Run HAN's own loader against our raw HGB directory.

    Their per-dataset loaders call ``data_loader('../../data/<NAME>')``, a path relative to
    the vendor tree that does not exist in this checkout. Patching the symbol keeps the
    heterograph construction, the feature choice and the meta-path lists verbatim.
    """

    raw_dir = hgb_raw_dir(dataset)
    with pushd_sys_path(*VENDORS):
        import utils as han_utils  # type: ignore[import-not-found]

        original = han_utils.data_loader
        han_utils.data_loader = lambda _path: original(str(raw_dir))
        try:
            return han_utils.load_data(VENDOR_DATASET[dataset], feat_type=0)
        finally:
            han_utils.data_loader = original


class HANAdapter:
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
        space = {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden_dim": {"values": [64, 128, 256]},
            "dropout": grids.DROPOUT,
            "num_heads": {"values": [4, 8]},
        }
        if task == LINK_PREDICTION:
            space["embed_dim"] = {"values": [16, 64, 128]}
            space["decoder_type"] = {"values": ["dot", "distmult"]}
        if task == GRAPH_CLASSIFICATION:
            space["embed_dim"] = {"values": [32, 64, 128]}
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
        if task == LINK_PREDICTION:
            return self._fit_link_prediction(bundle, params, seed=seed, device=device)
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "unsupported task")

        device = resolve_dgl_device(device)
        seed_everything(seed, device)

        (
            graph,
            features,
            vendor_labels,
            num_classes,
            _train_idx,
            _val_idx,
            _test_idx,
            _train_mask,
            _val_mask,
            _test_mask,
            meta_paths,
        ) = _load_vendor_graph(bundle.dataset)

        data = bundle.data
        expected = int(data.target_y.shape[0])
        if int(features.shape[0]) != expected:
            raise RuntimeError(
                f"HAN loaded {features.shape[0]} target features for {bundle.dataset}, "
                f"the project bundle has {expected} target nodes."
            )

        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()
        train_idx = data.target_train_mask.nonzero(as_tuple=False).view(-1).to(device)
        val_idx = data.target_val_mask.nonzero(as_tuple=False).view(-1).to(device)
        test_idx = data.target_test_mask.nonzero(as_tuple=False).view(-1).to(device)

        features = features.to(device)
        graph = graph.to(device)

        with pushd_sys_path(*VENDORS):
            from model_hetero import HAN  # type: ignore[import-not-found]

            num_heads = int(params.get("num_heads", 8))
            model = HAN(
                meta_paths=meta_paths,
                in_size=int(features.shape[1]),
                hidden_size=int(params.get("hidden_dim", 128)),
                out_size=bundle.num_classes,
                num_heads=[num_heads],
                dropout=float(params.get("dropout", 0.5)),
            ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 5e-3)),
            weight_decay=float(params.get("weight_decay", 1e-3)),
        )

        record = train_full_batch_node_classification(
            model=model,
            forward=lambda: model(graph, features),
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            optimizer=optimizer,
            criterion=node_classification_criterion(bundle.is_multilabel),
            is_multilabel=bundle.is_multilabel,
            **budget_kwargs(NODE_CLASSIFICATION, params),
        )
        record["params_source"] = params.source
        record["device"] = device.type
        record["num_meta_paths"] = float(len(meta_paths))
        return record

    def _fit_link_prediction(
        self,
        bundle: LinkBundle,
        params: BaselineParams,
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        device = resolve_dgl_device(device)
        seed_everything(seed, device)

        node_range = target_node_range(bundle)
        graph, meta_paths, num_metapaths = build_metapath_graph(
            bundle, node_range, device
        )

        hidden_dim = int(params.get("hidden_dim", 128))
        embed_dim = int(params.get("embed_dim", 64))
        model = HANLinkEncoder(
            bundle,
            node_range,
            meta_paths=meta_paths,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_heads=int(params.get("num_heads", 8)),
            dropout=float(params.get("dropout", 0.5)),
            vendors=VENDORS,
        ).to(device)

        data = bundle.data.to(device)
        head = SharedLinkHead(
            str(params.get("decoder_type", "dot")),
            num_relations=max(int(data.num_target_relations), 1),
            dim=embed_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 5e-3)),
            weight_decay=float(params.get("weight_decay", 1e-3)),
        )

        record = train_link_prediction(
            model=model,
            head=head,
            embed=lambda: model(graph),
            data=data,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(LINK_PREDICTION, params),
        )
        record["params_source"] = params.source
        record["device"] = device.type
        record["num_meta_paths"] = float(num_metapaths)
        record["input_mode"] = model.input.mode
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
        import dgl

        device = resolve_dgl_device(device)
        seed_everything(seed, device)

        num_relations = int(bundle.fold.dims.num_relations)
        # One meta-path per ordered pair of edge types. Fixed by the dataset, so the
        # per-meta-path GATConv stack is built once; only the graphs change per batch.
        meta_paths = [
            [_gc_relation_key(first) + _gc_relation_key(second)]
            for first in range(num_relations)
            for second in range(num_relations)
        ]

        embed_dim = int(params.get("embed_dim", 64))
        with pushd_sys_path(*VENDORS):
            from model_hetero import HAN  # type: ignore[import-not-found]

            model = HAN(
                meta_paths=meta_paths,
                in_size=bundle.in_channels,
                hidden_size=int(params.get("hidden_dim", 128)),
                out_size=embed_dim,
                num_heads=[int(params.get("num_heads", 8))],
                dropout=float(params.get("dropout", 0.5)),
            ).to(device)

        head = SharedGraphHead(
            embed_dim,
            bundle.num_classes,
            dropout=float(params.get("dropout", 0.0)),
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 5e-3)),
            weight_decay=float(params.get("weight_decay", 1e-3)),
        )

        def embed_batch(batch: Any) -> torch.Tensor:
            operators, names = batch_metapath_operators(
                batch, num_relations, torch.device("cpu")
            )
            edge_dict = {}
            for operator, name in zip(operators, names):
                row, column, _ = operator.coo()
                # Binarised, matching dgl.metapath_reachable_graph's unweighted
                # output; HAN's GATConv learns its own edge weights by attention.
                edge_dict[("node", name, "node")] = (row, column)
            graph = dgl.heterograph(
                edge_dict, num_nodes_dict={"node": int(batch.num_nodes)}
            ).to(device)
            # HANLayer caches the composed graph per object identity, so a fresh graph
            # each batch is required for the cache to invalidate.
            return model(graph, batch.x.float())

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
        record["device"] = device.type
        record["num_meta_paths"] = float(len(meta_paths))
        record["num_relation_keys"] = float(num_relations)
        record["readout"] = "shared_mean_pool_linear"
        return record


ADAPTER = HANAdapter()
