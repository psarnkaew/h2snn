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
3. **Node encoder is H2GB's own stack** (spec v1.19 T2, Part P P1b;
   :mod:`src.baselines.adapters._h2gformer_encoders`): H2GB datasets exactly as their configs;
   HGB NC ``Hetero_Raw+Hetero_Label``, LP and GC ``Hetero_Raw``, plus ``Hetero_Metapath`` when
   the tuned key ``metapath_pe`` is on (cells with an HGB-table walk schema). The label channel
   sees training labels only (route test); IMDB (multi-label) has no label channel.

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
from src.baselines.adapters._h2gformer_encoders import (
    H2GB_STACKS,
    LABEL,
    METAPATH,
    RAW,
    configure_encoders,
    encoder_stack,
    has_walk_schema,
    metapath2vec_embeddings,
    training_label_input,
    walk_schema,
)
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
# YouTube, Amazon removed (spec 1.5); the derived tasks wired in Part P P2.
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "pubmed", "acm_pa", "dblp_ap", "imdb_ma", "freebase_bb"})
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
    stack: tuple[str, ...] = (RAW,),
    dim_out: int | None = None,
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
    # spec v1.19 T2 (Part P P1b): H2GB's encoder stack per cell (_h2gformer_encoders).
    configure_encoders(cfg, stack)
    if dim_out is not None:
        cfg.share.dim_out = int(dim_out)  # the label channel's class count (Hetero_Label)
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
        if has_walk_schema(task, dataset) and dataset.lower() not in H2GB_STACKS:
            # spec v1.19 T2 (D-P3): the MetaPath2Vec channel on/off, where a schema exists
            space["metapath_pe"] = {"values": [True, False]}
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
        stack = node_classification_stack(bundle, params)
        model = build_node_classification_model(
            data, target_type, bundle, params, stack, device=device, seed=seed)

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
            # H2GB's label encoder reads `batch.split`; 'test' would add validation labels
            # (hetero_label_encoder.py:42-44), so evaluation runs as 'val' (training labels only).
            data.split = "train" if model.training else "val"
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
        record["node_encoder"] = "+".join(stack)
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
        stack = encoder_stack(LINK_PREDICTION, bundle.dataset, _metapath_key(
            LINK_PREDICTION, bundle.dataset, params))
        ensure_pyg_submodules()
        with pushd_sys_path(VENDOR):
            if METAPATH in stack:  # training message graph only (build_edge_index_dict)
                attach_metapath_embeddings(hetero, bundle.dataset, device=device, seed=seed)
            _configure_graphgym(
                params, dim_in=dim_in, target_type=source_type, device=device, stack=stack
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
        record["node_encoder"] = "+".join(stack)
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


def _metapath_key(task: str, dataset: str, params: BaselineParams) -> bool | None:
    """The tuned `metapath_pe` value of a cell that has the key, else None (spec v1.19 T2)."""
    if not has_walk_schema(task, dataset) or dataset.lower() in H2GB_STACKS:
        return None
    return bool(params.get("metapath_pe", False))


def node_classification_stack(bundle: NodeBundle, params: BaselineParams) -> tuple[str, ...]:
    """T2's NC stack. H2GB's label channel one-hot encodes one label per node
    (hetero_label_encoder.py:49-59), so a multi-label target (IMDB) drops it (stated, D-P5)."""
    stack = encoder_stack(NODE_CLASSIFICATION, bundle.dataset,
                          _metapath_key(NODE_CLASSIFICATION, bundle.dataset, params))
    if bundle.is_multilabel:
        stack = tuple(name for name in stack if name != LABEL)
    return stack


def attach_metapath_embeddings(data: Any, dataset: str, *, device: torch.device,
                               seed: int, epochs: int | None = None) -> None:
    """H2GB's MetaPath2Vec on `data`'s edges with the cell's walk schema, stored as each node
    type's `pestat_Hetero_Metapath` (as H2GB's master_loader.py:266-281 stores it)."""
    embeddings = metapath2vec_embeddings(
        dict(data.edge_index_dict), dict(data.num_nodes_dict), walk_schema(dataset),
        device=device, seed=seed, epochs=epochs)
    dim_pe = next(iter(embeddings.values())).size(1)
    for node_type in data.node_types:
        # H2GB's schemas visit every node type of their graphs; a type the HGB table never
        # visits (ACM term, IMDB keyword, ...) gets a zero encoding (no MetaPath2Vec
        # information; stated, D-P7), so the composed encoder's widths agree.
        value = embeddings.get(node_type)
        if value is None:
            value = torch.zeros(int(data[node_type].num_nodes), dim_pe)
        data[node_type]["pestat_Hetero_Metapath"] = value.to(device)


def build_node_classification_model(data: Any, target_type: str, bundle: NodeBundle,
                                    params: BaselineParams, stack: tuple[str, ...], *,
                                    device: torch.device, seed: int,
                                    metapath_epochs: int | None = None) -> Any:
    """GTModel with `stack` on `data` (prepared in place: training labels for the label
    channel, MetaPath2Vec embeddings for the MetaPath channel)."""
    if LABEL in stack:
        labels, train, val = training_label_input(
            bundle.data.target_y.cpu(), bundle.data.target_train_mask.cpu())
        data[target_type].y = labels.to(device)
        data[target_type].train_mask = train.to(device)
        data[target_type].val_mask = val.to(device)
    ensure_pyg_submodules()
    with pushd_sys_path(VENDOR):
        if METAPATH in stack:
            attach_metapath_embeddings(data, bundle.dataset, device=device, seed=seed,
                                       epochs=metapath_epochs)
        dim_in = {node_type: int(data[node_type].x.size(1)) for node_type in data.node_types}
        _configure_graphgym(params, dim_in=dim_in, target_type=target_type, device=device,
                            stack=stack, dim_out=bundle.num_classes)
        from H2GB.network.gt_model import GTModel  # type: ignore[import-not-found]

        shim = _DatasetShim(data, f"hgb-{bundle.dataset}")
        data.split = "train"
        return GTModel(dim_in, bundle.num_classes, shim).to(device)


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
