"""H2G-former (Lin et al., 2025) -- the graph transformer from the H2GB benchmark.

H2GB registers it as ``GTModel`` with ``gt.layer_type = SparseNodeTransformer``
(``H2GB/configs/*/**-H2Gformer.yaml``). Unlike H2GB's HINormer, whose transformer stack is
commented out in ``hinormer_model.py``, this model's stack genuinely runs
(``gt_model.py``: ``for i in range(cfg.gt.layers): batch = self.convs[i](batch)``), so it
is used as published.

H2GB is built around graphgym's global ``cfg`` and a ``master_loader`` that only knows its
own datasets (PDNS, mag-year, oag, ogbn-mag, ...). Rather than register an HGB dataset
there, this adapter configures ``cfg`` directly and hands the model a minimal
dataset-shaped object wrapping our ``HeteroData``. That keeps H2GB untouched -- no edits
to a vendored repository, no dependence on their loader's absolute paths.

Deviations, recorded on the run record:

1. **The split** is the project's canonical HGB carve-out.
2. **Model selection** is on validation macro-F1, uniformly.
3. **Node encoder is ``Hetero_Raw``** -- the per-type linear projection into a shared space
   (``hetero_raw_encoder.py:77``), which is the same convention HGB's ``fc_list`` uses and
   the one every other homogeneous/heterogeneous baseline here gets. Their ogbn-mag config
   stacks ``Hetero_Raw+Hetero_Label+Hetero_Metapath``, but the metapath encoder needs
   precomputed metapath2vec embeddings and the label encoder is a label-reuse channel; on
   HGB datasets with real node attributes the raw encoder is the natural choice, and it
   keeps the input treatment aligned with the rest of the table.

Link prediction reuses the same model through
:mod:`src.baselines.adapters._h2gformer_lp`, which swaps only the final head so every
node type is projected rather than one. H2GB ships no link-prediction config for any
dataset, so all four cells are tuned.

Graph classification reuses that same all-types head, with each minibatch presented as a
single-node-type multiplex ``HeteroData`` keyed by edge type (:mod:`._gc_typed`); the
shared mean-pool head then reads the one node type's rows.
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines.adapters._gc_typed import batch_hetero_data, relation_key
from src.baselines.adapters._h2gformer_lp import (
    AllTypesHead,
    H2GformerLinkEncoder,
    build_link_hetero_data,
)
from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
from src.baselines.adapters._target_block import TypedNodeInput
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

NAME = "h2gformer"
VENDOR = "h2gb"
SUPPORTED_DATASETS = frozenset({"acm", "dblp", "imdb"})
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "youtube", "amazon", "pubmed"})
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})


class _DatasetShim:
    """The minimal dataset surface H2GB's model constructors touch.

    ``GTModel`` and the encoders only ever use ``dataset[0]`` and, through it,
    ``.metadata()``; ``.name`` appears in logging. Supplying this avoids registering an
    HGB entry in H2GB's ``master_loader``, which would mean editing a vendored repo.
    """

    def __init__(self, data: Any, name: str) -> None:
        self._data = data
        self.name = name

    def __getitem__(self, index: int) -> Any:
        if index != 0:
            raise IndexError("the shim holds a single graph")
        return self._data

    def __len__(self) -> int:
        return 1


def _configure_graphgym(
    params: BaselineParams,
    *,
    dim_in: dict[str, int],
    target_type: str,
    device: torch.device,
) -> Any:
    """Populate graphgym's global cfg for one trial.

    graphgym keeps configuration in a process-global singleton, so every value the model
    reads is set explicitly here rather than inherited from whatever a previous trial or
    a stray yaml left behind.
    """

    # `import H2GB` runs its registration hooks, and set_cfg installs the custom
    # config groups (cfg.gt, cfg.posenc_*, ...) that H2GB/config/*.py declare. Without
    # set_cfg the bare graphgym cfg has no `gt` group at all.
    import H2GB  # noqa: F401  - registers custom modules
    from H2GB.graphgym.config import cfg, set_cfg  # type: ignore[import-not-found]

    set_cfg(cfg)

    hidden = int(params.get("hidden_dim", 64))

    cfg.device = str(device)
    # yacs validates attribute assignment against primitive types and rejects a dict,
    # but H2GB's encoders expect cfg.share.dim_in to be a per-node-type mapping
    # (hetero_raw_encoder.py:91). CfgNode subclasses dict, so item assignment stores
    # the mapping without tripping the attribute-level type check.
    cfg.share["dim_in"] = dim_in

    cfg.dataset.task = "hetero_node"
    cfg.dataset.task_type = "classification"
    cfg.dataset.task_entity = target_type
    cfg.dataset.transductive = True
    cfg.dataset.node_encoder = True
    cfg.dataset.node_encoder_name = "Hetero_Raw"
    cfg.dataset.node_encoder_bn = False
    cfg.dataset.edge_encoder = False
    cfg.dataset.edge_encoder_name = "none"
    cfg.dataset.edge_encoder_bn = False

    cfg.gnn.dim_inner = hidden
    cfg.gnn.dim_edge = hidden
    cfg.gnn.layers_mp = int(params.get("layers_mp", 1))
    cfg.gnn.dropout = float(params.get("dropout", 0.2))
    cfg.gnn.agg = "mean"

    cfg.gt.layer_type = "SparseNodeTransformer"
    cfg.gt.layers = int(params.get("gt_layers", 3))
    cfg.gt.layers_pre_gt = int(params.get("layers_pre_gt", 1))
    cfg.gt.layers_post_gt = int(params.get("layers_post_gt", 1))
    cfg.gt.attn_heads = int(params.get("attn_heads", 8))
    cfg.gt.dim_hidden = hidden
    cfg.gt.input_dropout = float(params.get("input_dropout", 0.0))
    cfg.gt.dropout = float(params.get("dropout", 0.2))
    cfg.gt.attn_dropout = float(params.get("attn_dropout", 0.0))
    cfg.gt.batch_norm = False
    cfg.gt.layer_norm = True
    cfg.gt.l2_norm = False
    cfg.gt.act = "gelu"
    cfg.gt.attn_mask = "Edge"
    cfg.gt.residual = "Fixed"
    cfg.gt.ffn = "Type"
    cfg.gt.edge_weight = False
    cfg.gt.jumping_knowledge = False
    cfg.gt.virtual_nodes = 0
    cfg.gt.head = "hetero_node"
    cfg.gt.l = 0
    return cfg


class H2GformerAdapter:
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
            "hidden_dim": {"values": [64, 128, 256]},
            "dropout": grids.DROPOUT,
            "attn_dropout": {"values": [0.0, 0.2]},
            "gt_layers": {"values": [2, 3]},
            "attn_heads": {"values": [4, 8]},
        }
        if task == LINK_PREDICTION:
            space["embed_dim"] = {"values": [16, 64, 128]}
            space["decoder_type"] = {"values": ["dot", "distmult"]}
        if task == GRAPH_CLASSIFICATION:
            space["embed_dim"] = {"values": [32, 64, 128]}
        return space

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        # H2GB ships configs for its own datasets only, not ACM/DBLP/IMDB.
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
        data, target_type = _build_hetero_data(bundle, device)
        dim_in = {
            node_type: int(data[node_type].x.size(1)) for node_type in data.node_types
        }

        ensure_pyg_submodules()
        with pushd_sys_path(VENDOR):
            _configure_graphgym(
                params, dim_in=dim_in, target_type=target_type, device=device
            )
            from H2GB.network.gt_model import GTModel  # type: ignore[import-not-found]

            shim = _DatasetShim(data, f"hgb-{bundle.dataset}")
            model = GTModel(dim_in, bundle.num_classes, shim).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 5e-4)),
            weight_decay=float(params.get("weight_decay", 1e-5)),
        )

        labels = bundle.data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()
        masks = bundle.data

        # GTModel and its head mutate the graph in place: the Hetero_Raw encoder
        # overwrites batch[nt].x with its projection, and the hetero_node head then
        # overwrites the target type's x with the class logits (head/hetero_node.py:47).
        # Reusing the same HeteroData across epochs would therefore feed epoch N's logits
        # back in as epoch N+1's features. Keep pristine inputs and restore before each
        # forward -- cheaper and more precise than deep-copying the whole graph.
        pristine_x = {
            node_type: data[node_type].x.detach().clone()
            for node_type in data.node_types
        }

        def forward() -> torch.Tensor:
            for node_type, value in pristine_x.items():
                data[node_type].x = value.clone()
            out = model(data)
            # The hetero_node head returns (prediction, label); take the prediction and
            # keep only the target node type.
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, dict):
                out = out[target_type]
            return out

        record = train_full_batch_node_classification(
            model=model,
            forward=forward,
            labels=labels,
            train_idx=masks.target_train_mask.nonzero(as_tuple=False).view(-1).to(device),
            val_idx=masks.target_val_mask.nonzero(as_tuple=False).view(-1).to(device),
            test_idx=masks.target_test_mask.nonzero(as_tuple=False).view(-1).to(device),
            optimizer=optimizer,
            criterion=node_classification_criterion(bundle.is_multilabel),
            is_multilabel=bundle.is_multilabel,
            **budget_kwargs(NODE_CLASSIFICATION, params),
            num_target=int(labels.shape[0]),
        )
        record["params_source"] = params.source
        record["node_encoder"] = "Hetero_Raw"
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
        node_input = TypedNodeInput(
            bundle, embedding_dim=int(params.get("hidden_dim", 64))
        ).to(device)
        hetero = build_link_hetero_data(bundle, node_input, device)
        dim_in = {
            node_type: int(hetero[node_type].x.size(1))
            for node_type in hetero.node_types
        }

        source_type = bundle.data.target_relation[0]
        ensure_pyg_submodules()
        with pushd_sys_path(VENDOR):
            _configure_graphgym(
                params, dim_in=dim_in, target_type=source_type, device=device
            )
            from H2GB.network.gt_model import GTModel  # type: ignore[import-not-found]

            shim = _DatasetShim(hetero, f"hgb-lp-{bundle.dataset}")
            vendor = GTModel(dim_in, embed_dim, shim)
            # Their head projects one configured node type and then applies a split
            # mask; link prediction needs every type and has no such mask. The MLP
            # itself is reused untouched.
            vendor.post_gt = AllTypesHead(vendor.post_gt)
            vendor = vendor.to(device)

        model = H2GformerLinkEncoder(
            vendor,
            node_input,
            hetero,
            dict(bundle.data.node_slices),
            int(bundle.data.x.size(0)),
        ).to(device)

        data = bundle.data.to(device)
        head = SharedLinkHead(
            str(params.get("decoder_type", "dot")),
            num_relations=max(int(data.num_target_relations), 1),
            dim=embed_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 5e-4)),
            weight_decay=float(params.get("weight_decay", 1e-5)),
        )

        record = train_link_prediction(
            model=model,
            head=head,
            embed=model,
            data=data,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(LINK_PREDICTION, params),
        )
        record["params_source"] = params.source
        record["decoder_type"] = str(params.get("decoder_type", "dot"))
        record["input_modes"] = ",".join(
            f"{k}:{v}" for k, v in sorted(node_input.modes.items())
        )
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
        embed_dim = int(params.get("embed_dim", 64))
        dim_in = {node_type: bundle.in_channels}

        # One prototype batch gives the constructors their metadata; the model is
        # built once and every later batch reuses it.
        prototype = batch_hetero_data(
            next(iter(bundle.loaders[0])), num_relations
        ).to(device)

        ensure_pyg_submodules()
        with pushd_sys_path(VENDOR):
            _configure_graphgym(
                params, dim_in=dim_in, target_type=node_type, device=device
            )
            from H2GB.network.gt_model import GTModel  # type: ignore[import-not-found]

            shim = _DatasetShim(prototype, f"tu-{bundle.dataset}")
            model = GTModel(dim_in, embed_dim, shim)
            # Their head projects one configured node type and then applies a split
            # mask; graph classification has no such mask. The MLP is reused untouched.
            model.post_gt = AllTypesHead(model.post_gt)
            model = model.to(device)

        head = SharedGraphHead(
            embed_dim,
            bundle.num_classes,
            dropout=float(params.get("dropout", 0.0)),
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=float(params.get("lr", 5e-4)),
            weight_decay=float(params.get("weight_decay", 1e-5)),
        )

        def embed_batch(batch: Any) -> torch.Tensor:
            # A fresh HeteroData per batch, which also sidesteps the in-place mutation
            # the encoder and head perform on batch[nt].x.
            hetero = batch_hetero_data(batch, num_relations).to(device)
            out = model(hetero)
            if isinstance(out, tuple):
                out = out[0]
            return out[node_type]

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
        record["num_relation_keys"] = float(num_relations)
        record["readout"] = "shared_mean_pool_linear"
        return record


def _build_hetero_data(bundle: NodeBundle, device: torch.device) -> tuple[Any, str]:
    """Rebuild the PyG HeteroData H2GB expects, from the canonical bundle."""

    from torch_geometric.datasets import HGBDataset

    from src.data.data_layout import REPO_DATA_ROOT, nc_root

    key = bundle.dataset.lower()
    data = HGBDataset(root=str(nc_root(REPO_DATA_ROOT)), name=key.upper())[0]

    # Attribute-less types get an identity-equivalent learned input, matching the
    # convention used everywhere else in this campaign.
    for node_type in data.node_types:
        if "x" not in data[node_type]:
            count = int(data[node_type].num_nodes)
            data[node_type].x = torch.eye(count)
    target_type = bundle.data.target_node_type
    # Their hetero_node head returns either the first `batch_size` rows (the HGTLoader
    # convention) or a `batch.split`-named mask (head/hetero_node.py:29-38). We train
    # full-batch transductively and index the splits ourselves, so declaring the whole
    # target block as the batch makes the head return every target node's prediction.
    data[target_type].batch_size = int(data[target_type].num_nodes)
    return data.to(device), target_type


ADAPTER = H2GformerAdapter()
