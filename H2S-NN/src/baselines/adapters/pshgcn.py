"""PSHGCN (He et al., 2024) -- positive-spectral heterogeneous graph convolution.

Node classification runs the authors' pipeline essentially verbatim; see
:mod:`src.baselines.adapters._hgb_vendor` for how their ``load_dataset`` is executed
against our data. Two deliberate deviations, both recorded on the run record:

1. **The train/validation split** is the project's deterministic 20% carve-out rather than
   their global-RNG shuffle (PSHGCN/hgb/processing.py:29-35), so every model in the table
   sees identical splits.
2. **Model selection** is on validation macro-F1 rather than their validation loss
   (PSHGCN/hgb/main.py:53), uniformly with every other baseline.

Link prediction runs their *separate* ``hgb-lp/`` codebase (a different model file and a
different preprocessing file from ``hgb/``) via :mod:`src.baselines.adapters._pshgcn_lp`,
with two further deviations:

3. **One model for all target relations.** Their runner fits a fresh model per test
   relation and averages (hgb-lp/training.py:41-51), as HGB's homogeneous baseline does.
   The campaign trains once over all target relations and macro-averages the metrics per
   relation, which is HGB's other sanctioned protocol (HGT/RGCN link_predict.py) and is
   what every other baseline here does.
4. **The campaign's shared decoder** replaces their Dot/DisMult, so the link-prediction
   table compares encoders. Their published ``decode`` choice is carried through as
   ``decoder_type``.

Cost note: the polynomial expands combinatorially in ``K``. At their published ACM setting
(K=5, 7 relations) that is 381 terms applied twice per forward, and autograd retains every
intermediate -- roughly 8 GB of activations at hidden=256. Budget accordingly, or cap K.
The link-prediction datasets have more relation keys than the node-classification ones
(PubMed 16, YouTube 5), so K compounds faster there; ``polynomial_terms`` is on the record.

Graph classification runs the same ``hgb-lp`` model per minibatch, with relations keyed
by edge type rather than node-type pair -- see :mod:`._gc_typed` for why, and for what
that costs on the three TU datasets that have only one edge type.
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
from src.baselines.adapters._pshgcn_lp import (
    PSHGCNLinkEncoder,
    build_polynomial_inputs,
    parameter_groups,
)
from src.baselines.adapters._gc_typed import batch_relation_operators
from src.baselines.adapters._gc_typed import relation_key as _gc_relation_key
from src.baselines.adapters._pshgcn_lp import vendor_args as lp_vendor_args
from src.baselines.adapters._polynomial_lp import (
    build_term_lists,
    polynomial_term_count,
)
from src.baselines.bundles import GraphBundle, LinkBundle, NodeBundle, bundle_data_root
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

NAME = "pshgcn"
VENDOR = "pshgcn"
# YouTube, Amazon removed (spec 1.5); the derived tasks wired in Part P P2.
SUPPORTED_LP_DATASETS = frozenset({"lastfm", "pubmed", "acm_pa", "dblp_ap", "imdb_ma", "freebase_bb"})
SUPPORTED_GC_DATASETS = frozenset({"mutag", "proteins", "enzymes", "nci1"})
#: Card 1d H1 (G-2, D1-b): ogbn-mag / mag-year with their ogbn-mag code in full (`_pshgcn_mag.py`): their inputs,
#: propagation and expander, the label part with sparse_tools' self-effect diagonals, batches of target rows.
H2GB_MAG_DATASETS = frozenset({"ogbn-mag", "mag-year"})
#: Card 3 Part B (F-1): Freebase NC on the generic branch of their HGB code (`_hgb_generic`, reverses completed).
FREEBASE = "freebase"
#: Card 3 Part B (G-2): the 6 non-MAG H2GB NC datasets on the generic decoupled scheme (`_hgb_generic`), class B.
H2GB_GENERIC_DATASETS = frozenset({"oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"})
#: Card 3 Part B (G-2): the two large LP tasks on the shared loop, the same decoupled scheme on the message graph.
LARGE_LP_TASKS = frozenset({"ogbn-mag_pp", "h-pokec_uu"})
READS_BATCH_PLAN = True


def _vendor_args(
    dataset: str, params: BaselineParams, *, device: torch.device
) -> SimpleNamespace:
    """The argparse-shaped namespace PSHGCN's model and processing expect."""

    return SimpleNamespace(
        dataset=VENDOR_DATASET.get(dataset, dataset),
        device=device,
        K=int(params.get("K", 5)),
        emb_dim=int(params.get("emb_dim", 64)),
        hidden=int(params.get("hidden", 64)),
        dropout=float(params.get("dropout", 0.0)),
        input_drop=float(params.get("input_drop", 0.0)),
        lr=float(params.get("lr", 0.01)),
        wd=float(params.get("weight_decay", 0.0)),
        prop_lr=float(params.get("prop_lr", 0.01)),
        prop_wd=float(params.get("prop_wd", 1.0)),
    )


class PSHGCNAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return (dataset in VENDOR_DATASET or dataset in H2GB_MAG_DATASETS or dataset == FREEBASE
                    or dataset in H2GB_GENERIC_DATASETS)
        if task == LINK_PREDICTION:
            return dataset in SUPPORTED_LP_DATASETS or dataset in LARGE_LP_TASKS
        if task == GRAPH_CLASSIFICATION:
            return dataset in SUPPORTED_GC_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task not in (NODE_CLASSIFICATION, LINK_PREDICTION, GRAPH_CLASSIFICATION):
            raise UnsupportedCell(NAME, task, dataset, "unsupported task")
        if task == GRAPH_CLASSIFICATION:
            # At most 4 relation keys here (MUTAG's bond types), so K can go higher
            # than on the link-prediction graphs without the monomial count exploding.
            return {
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,
                "emb_dim": {"values": [64, 128]},
                "hidden": grids.HIDDEN,
                "embed_dim": {"values": [32, 64, 128]},
                "input_drop": grids.INPUT_DROP,
                "dropout": grids.DROPOUT,
                "K": {"values": [1, 2, 3, 5]},
                "prop_lr": grids.PROP_LR,
                "prop_wd": {"values": [5e-5, 1e-3]},
            }
        if task == LINK_PREDICTION and dataset in LARGE_LP_TASKS:  # the decoupled scheme's knobs (G-2), no label part
            return {"lr": grids.LR, "weight_decay": grids.WEIGHT_DECAY, "emb_dim": {"values": [128, 256, 512]},
                    "hidden_x": {"values": [256, 512, 1024]}, "layers_x": {"values": [2, 3]},
                    "embed_dim": {"values": [16, 64, 256]}, "dropout": grids.DROPOUT, "input_drop": grids.INPUT_DROP,
                    "decoder_type": {"values": ["dot", "distmult"]}}
        if task == LINK_PREDICTION:
            # Centred on the two published link-prediction runners, whose values differ
            # sharply from the node-classification ones (hgb-lp/training.py and
            # training_dis.py). K is capped at 3: PubMed has 16 relation keys, so K=3
            # already expands to 4,369 monomials applied twice per forward.
            return {
                "lr": grids.LR,
                "weight_decay": grids.WEIGHT_DECAY,  # shared axis (spec v1.15 §6)
                "emb_dim": {"values": [128, 256]},
                "hidden": {"values": [128, 256]},
                "embed_dim": {"values": [16, 64, 256]},
                "input_drop": grids.INPUT_DROP,
                "dropout": grids.DROPOUT,  # shared axis (spec v1.15 §6)
                "K": {"values": [1, 2, 3]},
                "prop_lr": grids.PROP_LR,
                "prop_wd": {"values": [5e-5, 1e-3]},
                "decoder_type": {"values": ["dot", "distmult"]},
            }
        if dataset in H2GB_MAG_DATASETS:  # their ogbn-mag model's knobs (ogbn-mag/main.py:186-200); K = 2 (label keys)
            from src.configs.config import MINIBATCH_BATCH_SIZES

            return {"lr": grids.LR, "weight_decay": grids.WEIGHT_DECAY, "emb_dim": {"values": [128, 256, 512]},
                    "hidden_x": {"values": [256, 512, 1024]}, "hidden_l": {"values": [256, 512]},
                    "layers_x": {"values": [2, 3]}, "layers_l": {"values": [2, 3]}, "dropout": grids.DROPOUT,
                    "input_drop": grids.INPUT_DROP, "batch_size": {"values": list(MINIBATCH_BATCH_SIZES)}}
        if dataset in H2GB_GENERIC_DATASETS:
            # G-2: their ogbn-mag knobs (main.py:186-200) without the label part; K = 2, their default (stated).
            from src.configs.config import MINIBATCH_BATCH_SIZES

            return {"lr": grids.LR, "weight_decay": grids.WEIGHT_DECAY, "emb_dim": {"values": [128, 256, 512]},
                    "hidden_x": {"values": [256, 512, 1024]}, "layers_x": {"values": [2, 3]},
                    "dropout": grids.DROPOUT, "input_drop": grids.INPUT_DROP,
                    "batch_size": {"values": list(MINIBATCH_BATCH_SIZES)}}
        space = {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden": grids.HIDDEN,
            "dropout": grids.DROPOUT,
            # K drives a combinatorial blow-up in both time and activation memory; the
            # published values are 1 (IMDB) and 5 (ACM/DBLP).
            "K": {"values": [1, 3, 5]},
            "emb_dim": {"values": [16, 64, 128]},
            "input_drop": grids.INPUT_DROP,
            "prop_lr": grids.PROP_LR,
            "prop_wd": {"values": [0.0, 0.3, 0.8]},
        }
        space.update(grids.dataset_override(NAME, task, dataset))  # H-Pokec: K = 1, hidden <= 64 (author)
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
        if task == LINK_PREDICTION and bundle.dataset in LARGE_LP_TASKS:
            return fit_large_lp(bundle, params, seed=seed, device=device)
        if task == LINK_PREDICTION:
            return self._fit_link_prediction(bundle, params, seed=seed, device=device)
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "unsupported task")
        if bundle.dataset in H2GB_MAG_DATASETS:
            from src.baselines.adapters import _pshgcn_mag

            return _pshgcn_mag.fit(bundle, params, seed=seed, device=device)
        if bundle.dataset in H2GB_GENERIC_DATASETS:
            return fit_h2gb_generic(bundle, params, seed=seed, device=device)

        args = _vendor_args(bundle.dataset, params, device=device)
        if bundle.dataset == FREEBASE:
            from src.baselines.adapters._hgb_generic import hgb_features, hgb_generic_inputs, read_hgb_raw
            from src.baselines.bundles import hgb_raw_dir

            raw = read_hgb_raw(VENDOR, hgb_raw_dir(FREEBASE, bundle_data_root(bundle)))
            adjs = hgb_generic_inputs(raw)["adjs"]
            lis, lis_t = build_term_lists(adjs, args.K, enforce_type_chaining=True)  # their rule, processing.py:109-128
            feature_list, labels, num_classes = hgb_features(raw), raw["labels"], raw["num_classes"]
            vendor_test_idx = raw["test_idx"]
        else:
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
            ) = load_vendor_dataset(VENDOR, bundle.dataset, args, bundle_data_root(bundle))

        if bundle.dataset == FREEBASE:  # bundle rows (node.dat order) -> their rows (HGB id order)
            from src.baselines.adapters._hgb_generic import hgb_target_rows, project_bundle_split

            train_idx, val_idx, test_idx = project_bundle_split(
                bundle, raw, hgb_target_rows(hgb_raw_dir(FREEBASE, bundle_data_root(bundle))), vendor=NAME)
        else:
            verify_labels(bundle, labels, vendor=NAME)
            train_idx, val_idx, test_idx = canonical_split(
                bundle, vendor_test_idx, vendor=NAME
            )

        # Seed every RNG, not just torch: PSHGCN initialises its propagation coefficients
        # with np.random.uniform (model.py:init_coe), so seeding torch alone leaves the
        # run irreproducible. Equivalent to their own set_seed (PSHGCN/hgb/utils.py).
        seed_everything(seed, device)
        with pushd_sys_path(VENDOR):
            from model import PSHGCN  # type: ignore[import-not-found]

            in_dims = [feature.shape[1] for feature in feature_list]
            feature_list = [feature.to(device) for feature in feature_list]
            adjs = {key: adj.to(device) for key, adj in adjs.items()}
            model = PSHGCN(in_dims, num_classes, lis, lis_t, args).to(device)

        # Four parameter groups, as PSHGCN/hgb/main.py:23-26: the propagation
        # coefficients W carry their own learning rate and weight decay.
        optimizer = torch.optim.AdamW(
            [
                {"params": model.feat_project.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.lin1.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.lin2.parameters(), "weight_decay": args.wd, "lr": args.lr},
                {"params": model.W, "weight_decay": args.prop_wd, "lr": args.prop_lr},
            ]
        )

        record = train_full_batch_node_classification(
            model=model,
            forward=lambda: model(adjs, feature_list),
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
        record["polynomial_terms"] = float(1 + sum(len(v) for v in lis.values()))
        return record

    def _fit_link_prediction(
        self,
        bundle: LinkBundle,
        params: BaselineParams,
        *,
        seed: int,
        device: torch.device,
    ) -> dict[str, float]:
        # np.random.uniform seeds the polynomial coefficients (hgb-lp/pshgcn.py:58-62),
        # so torch-only seeding would leave every run a fresh random init.
        seed_everything(seed, device)

        args = lp_vendor_args(params, device=device)
        adjacencies, lis, lis_t, terms = build_polynomial_inputs(bundle, args.K, device)

        # feats_type 0 is their convention: real attributes where a type has them,
        # identity where it does not (hgb-lp/processing.py:18-24).
        view = bundle.homogeneous(feats_type=0)
        features = [
            feature.coalesce().to(device) if feature.is_sparse else feature.to(device)
            for feature in view.features_list
        ]

        embed_dim = int(params.get("embed_dim", 16))
        model = PSHGCNLinkEncoder(view.in_dims, embed_dim, lis, lis_t, args).to(device)
        data = bundle.data.to(device)
        head = SharedLinkHead(
            str(params.get("decoder_type", "dot")),
            num_relations=max(int(data.num_target_relations), 1),
            dim=embed_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(parameter_groups(model, head, args, params))

        record = train_link_prediction(
            model=model,
            head=head,
            embed=lambda: model(adjacencies, features),
            data=data,
            optimizer=optimizer,
            device=device,
            **budget_kwargs(LINK_PREDICTION, params),
        )
        record["params_source"] = params.source
        record["polynomial_terms"] = float(terms)
        record["num_relation_keys"] = float(len(adjacencies))
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

        args = lp_vendor_args(params, device=device)
        num_relations = int(bundle.fold.dims.num_relations)
        # The monomial lists depend only on the key set, which is fixed by the
        # dataset's edge types, so they are built once rather than per batch. Chaining
        # is off: one node type means every composition is valid.
        keys = {
            _gc_relation_key(index): None for index in range(num_relations)
        }
        lis, lis_t = build_term_lists(keys, args.K, enforce_type_chaining=False)

        embed_dim = int(params.get("embed_dim", 64))
        model = PSHGCNLinkEncoder(
            [bundle.in_channels], embed_dim, lis, lis_t, args
        ).to(device)
        head = SharedGraphHead(
            embed_dim,
            bundle.num_classes,
            dropout=float(params.get("dropout", 0.0)),
        ).to(device)

        optimizer = torch.optim.AdamW(
            parameter_groups(model, head, args, params)
        )

        def embed_batch(batch: Any) -> torch.Tensor:
            adjacencies = batch_relation_operators(batch, num_relations, device)
            return model(adjacencies, [batch.x.float()])

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
        record["polynomial_terms"] = float(polynomial_term_count(lis))
        record["num_relation_keys"] = float(num_relations)
        record["readout"] = "shared_mean_pool_linear"
        return record


def fit_h2gb_generic(bundle: Any, params: BaselineParams, *, seed: int, device: torch.device) -> dict[str, float]:
    """G-2: the generic decoupled scheme (`_hgb_generic`, class B) on an H2GB NC graph: their propagation once, batches
    of B target rows in the shared H2GB NC loop (`minibatch.train_index_minibatch`)."""
    from src.baselines.adapters._hgb_generic import build_decoupled_model, decoupled_forward, decoupled_inputs
    from src.baselines.minibatch import h2gb_graph, minibatch_plan, train_index_minibatch
    from src.data.h2gb_nc import ALL_H2GB_NC_SPECS

    seed_everything(seed, device)
    data = h2gb_graph(bundle)
    target = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
    labels = data[target].y.long()
    num_classes = int(labels.max()) + 1
    inputs = decoupled_inputs(data, target, seed)
    torch.manual_seed(seed)
    model = build_decoupled_model(inputs, num_classes, params).to(device)
    feats = inputs["feats"]
    plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 1e-3)),
                                  weight_decay=float(params.get("weight_decay", 5e-5)))

    def forward_rows(index: torch.Tensor) -> torch.Tensor:
        rows = index.cpu()
        return decoupled_forward(model, {k: v[rows].to(device) for k, v in feats.items()})

    record = train_index_minibatch(
        model=model, forward_rows=forward_rows, labels=labels.to(device),
        train_idx=data[target].train_mask.nonzero().view(-1), val_idx=data[target].val_mask.nonzero().view(-1),
        test_idx=data[target].test_mask.nonzero().view(-1), optimizer=optimizer,
        criterion=torch.nn.functional.cross_entropy, plan=plan, base_seed=seed, report=getattr(params, "report", None))
    record["params_source"] = params.source
    record["batch_size"] = float(plan.batch_size)
    record["polynomial_terms"] = float(inputs["coe_num"])
    return record


def build_large_lp(d: Any, params: Any, device: torch.device, sampler: Any = None) -> tuple[Any, Any]:
    """G-2 on a large LP task (shared loop `large_lp_training.fit_large_lp`): the decoupled per-node table of the
    endpoint type computed once on the **message graph** `d.graph` (held-out facts removed), its model's output as the
    embedding; a batch's rows are read by `n_id`, so the default sampler takes the endpoints only (fan-out 0)."""
    from src.baselines import large_lp_training as llp
    from src.baselines.adapters._hgb_generic import build_decoupled_model, decoupled_forward, decoupled_inputs

    inputs = decoupled_inputs(d.graph, d.node_type, int(torch.initial_seed()))  # the loop seeded every RNG
    embed_dim = int(params.get("embed_dim", 64))
    model = build_decoupled_model(inputs, embed_dim, params).to(device)
    feats = inputs["feats"]

    def embed(batch: Any, train: bool) -> torch.Tensor:  # noqa: ARG001 - train mode is set by the loop
        rows = batch[d.node_type].n_id.cpu()
        return decoupled_forward(model, {k: v[rows].to(device) for k, v in feats.items()})

    return (llp.LargeLPModel(modules=model, embed=embed, embed_dim=embed_dim,
                             extra={"polynomial_terms": inputs["coe_num"]}),
            sampler or llp.Sampler("link", [0]))


def fit_large_lp(bundle: Any, params: Any, *, seed: int, device: torch.device) -> dict[str, float]:
    from src.baselines import large_lp_training as llp

    record = llp.fit_large_lp(bundle, params, seed=seed, device=device,
                              build=lambda d: build_large_lp(d, params, device))
    record["params_source"] = params.source
    return record


ADAPTER = PSHGCNAdapter()
