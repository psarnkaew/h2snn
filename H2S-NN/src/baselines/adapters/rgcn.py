"""R-GCN (card 8 C2; decisions log 2026-11-20 "Card 8: R-GCN and MAGNN").

* **NC, HGB datasets** (all four cells hold a published number; the adapter supports them for the perturbation study and
  checks): HGB's `NC/benchmark/methods/RGCN` model, run as its code runs it -- `EntityClassify` = HGB's `BaseRGCN`
  (`model.py`) with one input `Linear` per node type, `n_layers - 2` hidden `RelGraphConv(h, h, R, "basis", B,
  relu, dropout)` and an output `RelGraphConv(h, C, R, "basis", B)` (`entity_classify.py:83-95`), on the relation
  graph of the dataset with edge norm `1 / in-degree of the destination under that relation` (`:151-161`), full batch.
  `EntityClassify` is rebuilt here from HGB's own `BaseRGCN` with those three builders verbatim: importing
  `entity_classify.py` runs `pynvml.nvmlInit()` at module level (`:33`), which needs an NVIDIA driver (CPU test jobs).
* **NC, H2GB datasets**: H2GB's own R-GCN (`HomoGNNModel`, `layer_type: RGCN`, `<dataset>-RGCN.yaml`) on H2GB's `hgt`
  sampler, exactly as the yaml states (`h2gb_published.fit_h2gb_published`). Every H2GB R-GCN cell is published (6) or
  filled (ogbn-mag, IEEE-CIS-G); a tuned run takes the same network with the tuned keys (`_fit_h2gb_tuned`).
* **LP (8 tasks)**: HGB's `LP/benchmark/methods/RGCN` encoder -- `RGCN` = `BaseRGCN` with input `Linear`s and
  `num_layers` hidden `RelGraphConv(h, h, 2(L+1), "bdd", n_bases, relu but not the last, self_loop=True, dropout)`
  (`link_predict.py:23-41`) on HGB's LP graph: the message graph's relation adjacencies summed, plus its transpose,
  self-loops removed then added (`:151-155`); an edge (u, v) gets its relation k, a self-loop gets L, an edge present
  only as the reverse of a k-edge gets k + 1 + L (`:141-150`); edge norm = 1 / in-degree of the destination
  (`utils.py:71-84`). Their DistMult decoder is replaced by the shared head (`SharedLinkHead`, dot / DistMult); their
  embedding regulariser `regularization * mean(embed^2)` (`link_predict.py:57-67`, default 0.01) is kept as the
  auxiliary loss (the relation-vector part belongs to their decoder). Small LP: the shared loop (batches of 8,192
  positives, C2b); large LP: `fit_large_lp` with H2GB's R-GCN fan-out, the same encoder on each sampled batch.
* GC: not run (HetSheaf's published number, option (a)).
"""

from __future__ import annotations

from typing import Any

import torch

from src.baselines import search_grids as grids
from src.baselines.budget import budget_kwargs
from src.baselines.bundles import HGB_DEFAULT_FEATS_TYPE, LinkBundle, NodeBundle
from src.baselines.heads import SharedLinkHead
from src.baselines.protocol import (
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
    train_link_prediction,
)
from src.utils.misc import seed_everything

NAME = "rgcn"
VENDOR_NC = "hgb_rgcn_nc"
VENDOR_LP = "hgb_rgcn_lp"
HGB_NC_DATASETS = frozenset(HGB_DEFAULT_FEATS_TYPE)
H2GB_DATASETS = frozenset({"ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"})
SMALL_LP_DATASETS = frozenset({"lastfm", "pubmed", "acm_pa", "dblp_ap", "imdb_ma", "freebase_bb"})
LARGE_LP_DATASETS = frozenset({"ogbn-mag_pp", "h-pokec_uu"})
#: HGB LP RGCN argparse defaults (`link_predict.py:356-381`).
LP_DEFAULTS = {"hidden_dim": 64, "num_layers": 2, "dropout": 0.5, "lr": 5e-4, "weight_decay": 1e-4, "n_bases": 10,
               "regularization": 0.01}
#: HGB NC RGCN argparse defaults (`entity_classify.py:319-344`).
NC_DEFAULTS = {"hidden_dim": 16, "n_layers": 3, "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0, "n_bases": -1}


# ------------------------------------------------------------------------------------------------ HGB's models

#: The HGB root of each R-GCN folder, pushed with it so the folder's `model` module is dropped on exit
#: (`pushd_sys_path` attributes it to that root; card 8 C4 fix, see magnn.py "Vendor modules").
_PARENT_ROOT = {VENDOR_NC: "hgb_nc", VENDOR_LP: "hgb_lp"}


def _vendor_base(vendor: str) -> type:
    with pushd_sys_path(vendor, _PARENT_ROOT[vendor]):
        from model import BaseRGCN  # type: ignore[import-not-found]
    return BaseRGCN


def entity_classify_model(in_dims: list[int], hidden: int, num_classes: int, num_rels: int, num_bases: int,
                          n_layers: int, dropout: float) -> torch.nn.Module:
    """HGB's NC `EntityClassify` (`entity_classify.py:83-95`, built verbatim on HGB's `BaseRGCN`)."""
    import torch.nn.functional as F
    from dgl.nn.pytorch import RelGraphConv
    from torch import nn

    base = _vendor_base(VENDOR_NC)

    class EntityClassify(base):  # type: ignore[misc, valid-type]
        def build_input_layer(self):
            return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims])

        def build_hidden_layer(self, idx):
            return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                                dropout=self.dropout)

        def build_output_layer(self):
            return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                                self.num_bases, activation=None,
                                self_loop=self.use_self_loop)

    # entity_classify.py:226-233: num_hidden_layers = n_layers - 2, use_self_loop False (argparse default)
    return EntityClassify(in_dims, hidden, num_classes, num_rels, num_bases=num_bases,
                          num_hidden_layers=n_layers - 2, dropout=dropout, use_self_loop=False)


def lp_rgcn_model(in_dims: list[int], hidden: int, num_rels: int, num_bases: int, num_layers: int,
                  dropout: float) -> torch.nn.Module:
    """HGB's LP `RGCN` encoder (`link_predict.py:23-31`) with `LinkPredict`'s relation count `2 * num_rels`
    (`:35-39`); `num_rels` = L + 1 (`:177-183`)."""
    import torch.nn.functional as F
    from dgl.nn.pytorch import RelGraphConv
    from torch import nn

    base = _vendor_base(VENDOR_LP)

    class RGCN(base):  # type: ignore[misc, valid-type]
        def build_input_layer(self):
            return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims])

        def build_hidden_layer(self, idx):
            act = F.relu if idx < self.num_hidden_layers - 1 else None
            return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                                self.num_bases, activation=act, self_loop=True,
                                dropout=self.dropout)

    return RGCN(in_dims, hidden, hidden, num_rels * 2, num_bases, num_layers, dropout)


# ------------------------------------------------------------------------------------------------ graphs

def nc_relation_graph(edge_index: torch.Tensor, edge_type: torch.Tensor, num_nodes: int
                      ) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """(DGL graph, edge types, edge norm) of HGB's NC RGCN (`entity_classify.py:151-164`): one edge per stored link,
    norm = 1 / (in-degree of the destination under the edge's relation)."""
    import dgl

    src, dst = edge_index[0].long(), edge_index[1].long()
    etype = edge_type.long()
    key = etype * num_nodes + dst
    _, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
    norm = (1.0 / counts[inverse].float()).unsqueeze(1)
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    return graph, etype, norm


def lp_relation_graph(edge_index: torch.Tensor, edge_type: torch.Tensor, num_nodes: int, num_relations: int
                      ) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """(DGL graph, edge types, edge norm) of HGB's LP RGCN (`link_predict.py:141-162`) on the message graph:
    adjacency + its transpose, self-loops removed then added (one edge per distinct pair); edge types by HGB's dict
    rule, relations visited in id order: a stored pair (u, v) gets its relation k (a later relation overwrites, so the
    largest k), a self-loop not stored gets L, a pair present only as the reverse of stored pairs gets k + 1 + L for the
    first such k (the smallest); norm = 1 / in-degree of the destination (`utils.py:71-84`)."""
    import dgl

    L, n = int(num_relations), int(num_nodes)
    src, dst, etype = edge_index[0].long().cpu(), edge_index[1].long().cpu(), edge_type.long().cpu()
    fkey = src * n + dst
    keys, inverse = torch.unique(fkey, return_inverse=True)
    type_max = torch.full((keys.numel(),), -1, dtype=torch.long).scatter_reduce(0, inverse, etype, "amax")
    type_min = torch.full((keys.numel(),), L, dtype=torch.long).scatter_reduce(0, inverse, etype, "amin")
    pairs = torch.unique(torch.cat([fkey, dst * n + src]))
    u, v = pairs // n, pairs % n
    keep = u != v
    loops = torch.arange(n)
    u, v = torch.cat([u[keep], loops]), torch.cat([v[keep], loops])

    def lookup(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.numel() == 0:
            return torch.zeros_like(query, dtype=torch.bool), torch.zeros_like(query)
        pos = torch.searchsorted(keys, query).clamp(max=keys.numel() - 1)
        return keys[pos] == query, pos

    found, pos = lookup(u * n + v)
    types = torch.where(found, type_max[pos], torch.full_like(u, -1))
    types = torch.where((types < 0) & (u == v), torch.full_like(u, L), types)
    rfound, rpos = lookup(v * n + u)
    types = torch.where((types < 0) & rfound, type_min[rpos] + 1 + L, types)
    if bool((types < 0).any()):
        raise RuntimeError("an edge of the LP RGCN graph got no type")
    graph = dgl.graph((u, v), num_nodes=n)
    in_degree = torch.bincount(v, minlength=n).float()
    norm = torch.where(in_degree > 0, 1.0 / in_degree, torch.zeros_like(in_degree))[v].unsqueeze(1)
    return graph, types, norm


def _features(view: Any, device: torch.device) -> list[torch.Tensor]:
    return [f.coalesce().to(device) if f.is_sparse else f.to(device) for f in view.features_list]


def _check_type_order(view: Any) -> None:
    """HGB concatenates the per-type inputs in type order (`model.py: forward`); our global ids must be the same
    contiguous blocks in that order."""
    start = 0
    for name in view.node_type_names:
        lo, hi = view.node_slices[name]
        if lo != start:
            raise ValueError(f"node type {name!r} starts at {lo}, expected {start}: types are not contiguous blocks")
        start = hi


# ------------------------------------------------------------------------------------------------ adapter

class RGCNAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return dataset in HGB_NC_DATASETS or dataset in H2GB_DATASETS
        if task == LINK_PREDICTION:
            return dataset in SMALL_LP_DATASETS or dataset in LARGE_LP_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task == NODE_CLASSIFICATION and dataset in HGB_NC_DATASETS:
            return {"lr": grids.LR, "weight_decay": grids.WEIGHT_DECAY, "dropout": grids.DROPOUT,
                    "hidden_dim": grids.HIDDEN, "n_layers": {"values": [2, 3, 4]},
                    "feats_type": {"values": [0, 2, 3]}}
        if task == LINK_PREDICTION:
            # HGB's argparse keys; `n_bases` must divide the width under "bdd" (DGL TypedLinear), so the widths are
            # multiples of 10 and 4: 60 is HGB's README width for PubMed/amazon, 10 its default bases; its default
            # width 64 is not divisible by 10 and is left out (stated).
            space = {"lr": grids.LR, "weight_decay": grids.WEIGHT_DECAY, "dropout": grids.DROPOUT,
                     "hidden_dim": {"values": [40, 60, 80, 120]}, "num_layers": {"values": [1, 2, 3]},
                     "n_bases": {"values": [4, 10]}, "decoder_type": {"values": ["dot", "distmult"]}}
            return space
        if task == NODE_CLASSIFICATION and dataset in H2GB_DATASETS:
            # card 8 C4: the tuned H2GB path (H2GB's own R-GCN, its yaml with the tuned keys), as HAN's class A
            from src.configs.config import MINIBATCH_BATCH_SIZES

            return {"lr": grids.LR, "weight_decay": grids.WEIGHT_DECAY, "dropout": grids.DROPOUT,
                    "hidden_dim": {"values": [64, 128, 256, 512]}, "num_layers": {"values": [2, 3, 4]},
                    "batch_size": {"values": list(MINIBATCH_BATCH_SIZES)}}
        raise UnsupportedCell(NAME, task, dataset, "R-GCN runs NC and LP only")

    def published_params(self, task: str, dataset: str) -> BaselineParams | None:
        return load_published_params(NAME, task, dataset)

    def fit_task(self, bundle: Any, params: BaselineParams, *, task: str, seed: int,
                 device: torch.device) -> dict[str, float]:
        if task == NODE_CLASSIFICATION and bundle.dataset in H2GB_DATASETS:
            from src.baselines.h2gb_published import fit_h2gb_published, is_h2gb_published

            if is_h2gb_published(NAME, task, bundle.dataset, params.source):
                return fit_h2gb_published(bundle, params, model_name=NAME, seed=seed, device=device)
            return self._fit_h2gb_tuned(bundle, params, seed=seed, device=device)
        if task == NODE_CLASSIFICATION:
            return self._fit_hgb_nc(bundle, params, seed=seed, device=device)
        if task == LINK_PREDICTION and bundle.dataset in LARGE_LP_DATASETS:
            return self._fit_large_lp(bundle, params, seed=seed, device=device)
        if task == LINK_PREDICTION:
            return self._fit_link_prediction(bundle, params, seed=seed, device=device)
        raise UnsupportedCell(NAME, task, bundle.dataset, "R-GCN runs NC and LP only (GC: HetSheaf's number)")

    # ---------------------------------------------------------------- NC on H2GB datasets, tuned
    def _fit_h2gb_tuned(self, bundle: Any, params: BaselineParams, *, seed: int,
                        device: torch.device) -> dict[str, float]:
        """H2GB's own R-GCN (`HomoGNNModel`, `<dataset>-RGCN.yaml`) with the tuned keys overridden, AdamW, the shared
        H2GB NC loop at batch size B (as HAN's class A, `hgb_han.fit_h2gb_han`; every current R-GCN H2GB cell is
        published or filled, this path serves a tuned run)."""
        from src.baselines.adapters._h2gb_network import configure_from_yaml, seed_predictions
        from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
        from src.baselines.h2gb_published import build_h2gb_network
        from src.baselines.minibatch import (eval_loader, h2gb_graph, h2gb_sample_sizes, hgt_loader, minibatch_plan,
                                             train_minibatch_node_classification, val_subset_loader)
        from src.data.h2gb_nc import ALL_H2GB_NC_SPECS

        seed_everything(seed, device)
        data = h2gb_graph(bundle)
        target = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
        fanout = h2gb_sample_sizes(NAME, bundle.dataset, tuple(data.node_types))
        plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
        labels = data[target].y.long()
        num_classes = int(labels.max()) + 1
        loaders = (hgt_loader(data, fanout, target, "train", plan.batch_size, shuffle=True, steps=plan.steps_per_epoch),
                   eval_loader(data, fanout, target, "val", plan.batch_size),
                   eval_loader(data, fanout, target, "test", plan.batch_size))
        subset_loader, subset_ids = val_subset_loader(data, fanout, target, bundle.dataset, plan.batch_size,
                                                      run_seed=bundle.metadata.get("seed"))
        ensure_pyg_submodules()
        with pushd_sys_path("h2gb"):  # the cfg is configured last: H2GB's model reads it in its forward
            cfg = configure_from_yaml(NAME, bundle.dataset, data, target, device, {
                "gnn.dim_inner": int(params.get("hidden_dim", 64)), "gnn.layers_mp": int(params.get("num_layers", 2)),
                "gnn.dropout": float(params.get("dropout", 0.2)), "train.sampler": "hgt"})
            torch.manual_seed(seed)
            model = build_h2gb_network(data, bundle.dataset, num_classes, cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 5e-4)),
                                      weight_decay=float(params.get("weight_decay", 1e-4)))
        record = train_minibatch_node_classification(
            model=model, forward_seeds=lambda batch: seed_predictions(model, batch.to(device), target),
            train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2],
            val_ids=data[target].val_mask.nonzero().view(-1), test_ids=data[target].test_mask.nonzero().view(-1),
            val_subset_loader=subset_loader, val_subset_ids=subset_ids, target_type=target, optimizer=optimizer,
            criterion=torch.nn.functional.cross_entropy, plan=plan, report=params.report, labels=labels)
        record["params_source"] = params.source
        record["batch_size"] = float(plan.batch_size)
        return record

    # ---------------------------------------------------------------- NC on HGB datasets
    def _fit_hgb_nc(self, bundle: NodeBundle, params: BaselineParams, *, seed: int,
                    device: torch.device) -> dict[str, float]:
        device = resolve_dgl_device(device)
        seed_everything(seed, device)
        feats_type = int(params.get("feats_type", 3))  # entity_classify.py:319 default 3
        view = bundle.homogeneous(feats_type=feats_type)
        _check_type_order(view)
        data = bundle.data
        graph, etype, norm = nc_relation_graph(data.edge_index, data.edge_type, int(view.num_nodes))
        num_rels = len(data.relation_names)
        model = entity_classify_model(view.in_dims, int(params.get("hidden_dim", NC_DEFAULTS["hidden_dim"])),
                                      bundle.num_classes, num_rels, int(params.get("n_bases", NC_DEFAULTS["n_bases"])),
                                      int(params.get("n_layers", NC_DEFAULTS["n_layers"])),
                                      float(params.get("dropout", NC_DEFAULTS["dropout"]))).to(device)
        graph, etype, norm = graph.to(device), etype.to(device), norm.to(device)
        features = _features(view, device)
        start, end = view.node_slices[view.target_node_type]
        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()
        lr = float(params.get("lr", NC_DEFAULTS["lr"]))
        weight_decay = float(params.get("weight_decay", NC_DEFAULTS["weight_decay"]))
        # entity_classify.py:239: Adam for a published run; the shared protocol's AdamW for a tuned one
        optimizer = (torch.optim.Adam if params.get("optimizer") == "adam" else torch.optim.AdamW)(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        record = train_full_batch_node_classification(
            model=model, forward=lambda: model(graph, features, etype, norm)[start:end], labels=labels,
            train_idx=data.target_train_mask.nonzero().view(-1).to(device),
            val_idx=data.target_val_mask.nonzero().view(-1).to(device),
            test_idx=data.target_test_mask.nonzero().view(-1).to(device), optimizer=optimizer,
            criterion=node_classification_criterion(bundle.is_multilabel), is_multilabel=bundle.is_multilabel,
            **budget_kwargs(NODE_CLASSIFICATION, params))
        record["params_source"] = params.source
        record["feats_type"] = float(feats_type)
        return record

    # ---------------------------------------------------------------- small LP
    def _fit_link_prediction(self, bundle: LinkBundle, params: BaselineParams, *, seed: int,
                             device: torch.device) -> dict[str, float]:
        device = resolve_dgl_device(device)
        seed_everything(seed, device)
        data = bundle.data.to(device)
        # as every HGB LP adapter here: feats_type 0 keeps attributes, identity where a type has none (HGB's RGCN
        # default is 3, identity for every type; stated)
        feats_type = int(params.get("feats_type", 0))
        view = bundle.homogeneous(feats_type=feats_type)
        _check_type_order(view)
        num_relations = len(data.relation_names)
        graph, etype, norm = lp_relation_graph(data.edge_index.cpu(), data.edge_type.cpu(), int(view.num_nodes),
                                               num_relations)
        hidden = int(params.get("hidden_dim", LP_DEFAULTS["hidden_dim"]))
        model = lp_rgcn_model(view.in_dims, hidden, num_relations + 1,
                              int(params.get("n_bases", LP_DEFAULTS["n_bases"])),
                              int(params.get("num_layers", LP_DEFAULTS["num_layers"])),
                              float(params.get("dropout", LP_DEFAULTS["dropout"]))).to(device)
        graph, etype, norm = graph.to(device), etype.to(device), norm.to(device)
        features = _features(view, device)
        head = SharedLinkHead(str(params.get("decoder_type", "dot")),
                              num_relations=max(int(data.num_target_relations), 1), dim=hidden).to(device)
        # link_predict.py:184: Adam for a published run; the shared protocol's AdamW for a tuned one
        optimizer = (torch.optim.Adam if params.get("optimizer") == "adam" else torch.optim.AdamW)(
            list(model.parameters()) + list(head.parameters()), lr=float(params.get("lr", LP_DEFAULTS["lr"])),
            weight_decay=float(params.get("weight_decay", LP_DEFAULTS["weight_decay"])))
        regularization = float(params.get("regularization", LP_DEFAULTS["regularization"]))
        cache: dict[str, torch.Tensor] = {}

        def embed() -> torch.Tensor:
            cache["h"] = model(graph, features, etype, norm)
            return cache["h"]

        record = train_link_prediction(
            model=model, head=head, embed=embed, data=data, optimizer=optimizer, device=device,
            aux_loss=lambda: regularization * cache["h"].pow(2).mean(),
            **budget_kwargs(LINK_PREDICTION, params))
        record["params_source"] = params.source
        record["feats_type"] = float(feats_type)
        return record

    # ---------------------------------------------------------------- large LP
    def build_large_lp(self, d: Any, params: BaselineParams, device: torch.device, sampler: Any = None) -> tuple:
        """HGB's LP RGCN encoder on each sampled batch's typed edges (card 1c F7 pattern; H2GB's R-GCN fan-out)."""
        from src.baselines.large_lp_training import LargeLPModel, h2gb_sampler
        from src.baselines.minibatch import batch_type_inputs, h2gb_input_dims

        node_types, counts, dims = h2gb_input_dims(d.graph)
        relations = tuple(d.graph.edge_types)
        sampler = sampler or h2gb_sampler(NAME, d.task, node_types)
        hidden = int(params.get("hidden_dim", LP_DEFAULTS["hidden_dim"]))
        encoder = lp_rgcn_model(dims, hidden, len(relations) + 1, int(params.get("n_bases", LP_DEFAULTS["n_bases"])),
                                int(params.get("num_layers", LP_DEFAULTS["num_layers"])),
                                float(params.get("dropout", LP_DEFAULTS["dropout"])))
        regularization = float(params.get("regularization", LP_DEFAULTS["regularization"]))
        cache: dict[str, torch.Tensor] = {}

        def embed(batch: Any, train: bool) -> torch.Tensor:
            inputs = [x.to(device) for x in batch_type_inputs(batch, node_types, counts)]
            offsets, start = {}, 0
            for node_type in node_types:
                offsets[node_type] = start
                start += int(batch[node_type].num_nodes)
            parts, types = [], []
            for k, (src_t, rel, dst_t) in enumerate(relations):
                if (src_t, rel, dst_t) not in batch.edge_types:
                    continue
                index = batch[src_t, rel, dst_t].edge_index
                parts.append(torch.stack([index[0] + offsets[src_t], index[1] + offsets[dst_t]]))
                types.append(torch.full((index.size(1),), k, dtype=torch.long))
            edge_index = torch.cat(parts, 1) if parts else torch.empty(2, 0, dtype=torch.long)
            edge_type = torch.cat(types) if types else torch.empty(0, dtype=torch.long)
            graph, etype, norm = lp_relation_graph(edge_index, edge_type, start, len(relations))
            out = encoder(graph.to(device), inputs, etype.to(device), norm.to(device))
            lo = offsets[d.node_type]
            cache["h"] = out
            return out[lo:lo + int(batch[d.node_type].num_nodes)]

        model = LargeLPModel(encoder, embed, hidden, aux_loss=lambda: regularization * cache["h"].pow(2).mean())
        return model, sampler

    def _fit_large_lp(self, bundle: Any, params: BaselineParams, *, seed: int, device: torch.device
                      ) -> dict[str, float]:
        from src.baselines.large_lp_training import fit_large_lp

        device = resolve_dgl_device(device)
        record = fit_large_lp(bundle, params, seed=seed, device=device,
                              build=lambda d: self.build_large_lp(d, params, device))
        record["params_source"] = params.source
        return record


ADAPTER = RGCNAdapter()

__all__ = ["ADAPTER", "RGCNAdapter", "entity_classify_model", "lp_relation_graph", "lp_rgcn_model",
           "nc_relation_graph"]
