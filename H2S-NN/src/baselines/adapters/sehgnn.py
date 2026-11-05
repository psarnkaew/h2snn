"""SeHGNN (Yang et al., AAAI 2023) -- simple and efficient heterogeneous GNN.

Structurally unlike the other baselines: SeHGNN precomputes meta-path neighbour
aggregations *before* any model exists, then trains a transformer-style network over that
fixed feature dictionary in minibatches. The adapter therefore reuses their
``load_dataset``, their ``hg_propagate_feat_dgl`` propagation and their ``SeHGNN`` model,
and only supplies the loop.

Their ``load_dataset`` reads ``f'{args.root}/{args.dataset}'``, so unlike PSHGCN and HAN
it needs no symbol patching -- just a root pointing at our raw HGB tree.

Deviations, all recorded on the run record:

1. **The split** is the project's canonical HGB carve-out. Theirs takes the last 10% of
   the shuffled training pool (``SeHGNN/hgb/main.py``), a third distinct rule.
2. **Model selection** is on validation macro-F1, uniformly.
3. **Label-propagation features** (``--label-feats``, which every published command
   passes) are implemented, because they are part of the published method and omitting
   them would bias the comparison against SeHGNN. They are safe under our protocol: the
   one-hot matrix is populated at *training* rows only, exactly as
   ``SeHGNN/hgb/main.py:211`` does, and ``remove_diag`` stops a node's own label feeding
   back to itself. Since the campaign replaces their split, the one-hot is built from
   *our* training mask, so our validation labels never enter the propagation.

Their multi-label rule for IMDB is ``(output > 0)`` (``hgb/utils.py:228``), matching the
campaign's ``threshold`` protocol, so IMDB needs no reconciliation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from src.baselines.bundles import NodeBundle, hgb_raw_dir
from src.baselines.metrics import classification_metrics, node_classification_record
from src.baselines.budget import resolve_budget
from src.baselines.progress import report_epoch
from src.baselines import search_grids as grids
from src.baselines.protocol import (
    NODE_CLASSIFICATION,
    BaselineParams,
    UnsupportedCell,
)
from src.baselines.published import load_published_params
from src.baselines.runtime import pushd_sys_path
from src.baselines.runtime._dgl_compat import resolve_dgl_device
from src.utils.misc import seed_everything

NAME = "sehgnn"
VENDOR = "sehgnn"
# utils.py imports `data_loader` from the sibling SeHGNN/data directory.
VENDORS = ("sehgnn", "sehgnn_data")

VENDOR_DATASET = {"acm": "ACM", "dblp": "DBLP", "imdb": "IMDB"}
#: Target node type and the type set, from SeHGNN/hgb/main.py:31-44.
TARGET_TYPE = {"ACM": "P", "DBLP": "A", "IMDB": "M"}


def _vendor_args(dataset: str, params: BaselineParams) -> SimpleNamespace:
    vendor_name = VENDOR_DATASET[dataset]
    raw_dir = hgb_raw_dir(dataset)
    return SimpleNamespace(
        # load_dataset reads f'{root}/{dataset}', so root is the raw tree's parent.
        root=str(raw_dir.parent),
        dataset=vendor_name,
        num_hops=int(params.get("num_hops", 2)),
        num_label_hops=int(params.get("num_label_hops", 2)),
        embed_size=int(params.get("embed_size", 256)),
        hidden=int(params.get("hidden", 512)),
        dropout=float(params.get("dropout", 0.5)),
        input_drop=float(params.get("input_drop", 0.5)),
        att_drop=float(params.get("att_drop", 0.0)),
        n_fp_layers=int(params.get("n_fp_layers", 2)),
        n_task_layers=int(params.get("n_task_layers", 1)),
        act=str(params.get("act", "none")),
        residual=bool(params.get("residual", False)),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        batch_size=int(params.get("batch_size", 10000)),
        # Read by their loader (hgb/utils.py). ACM_keep_F keeps ACM's extra term/
        # field node types; their argparse default is False, matching the published
        # ACM command, which does not pass --ACM_keep_F.
        ACM_keep_F=bool(params.get("ACM_keep_F", False)),
        two_layer=bool(params.get("two_layer", False)),
    )


def _precompute_features(
    args: SimpleNamespace,
    device: torch.device,
    *,
    label_onehot: torch.Tensor | None,
) -> tuple[Any, ...]:
    """Run SeHGNN's own loader, meta-path feature propagation and label propagation."""

    with pushd_sys_path(*VENDORS):
        from torch_sparse import remove_diag  # type: ignore[import-not-found]
        from utils import (  # type: ignore[import-not-found]
            hg_propagate_feat_dgl,
            hg_propagate_sparse_pyg,
            load_dataset,
        )

        graph, adjs, init_labels, num_classes, dl, trainval_nid, test_nid = load_dataset(
            args
        )

        # Row-normalise each relation, as main.py:24-26 does before propagation.
        for key in adjs.keys():
            adjs[key].storage._value = None
            adjs[key].storage._value = (
                torch.ones(adjs[key].nnz())
                / adjs[key].sum(dim=-1)[adjs[key].storage.row()]
            )

        target_type = TARGET_TYPE[args.dataset]
        max_length = args.num_hops + 1
        graph = hg_propagate_feat_dgl(
            graph, target_type, args.num_hops, max_length, [], echo=False
        )

        feats = {}
        for key in list(graph.nodes[target_type].data.keys()):
            feats[key] = graph.nodes[target_type].data.pop(key)

        # Label propagation, mirroring main.py:255-300. `prop_feats=False` returns the
        # meta-path adjacencies themselves; remove_diag drops the self-loop so a node's
        # own label cannot propagate back into its own feature.
        label_feats: dict[str, torch.Tensor] = {}
        if label_onehot is not None:
            label_length = args.num_label_hops + 1
            meta_adjs = hg_propagate_sparse_pyg(
                adjs,
                target_type,
                args.num_label_hops,
                label_length,
                [],
                prop_feats=False,
                echo=False,
                prop_device="cpu",
            )
            for key, adj in meta_adjs.items():
                label_feats[key] = remove_diag(adj) @ label_onehot

    return feats, label_feats, init_labels, num_classes, trainval_nid, test_nid


class SeHGNNAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        return task == NODE_CLASSIFICATION and dataset in VENDOR_DATASET

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, dataset, "node classification only")
        return {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden": {"values": [256, 512]},
            "embed_size": {"values": [256, 512]},
            "dropout": grids.DROPOUT,
            "input_drop": grids.INPUT_DROP,
            "num_hops": {"values": [2, 3, 4]},
            "num_label_hops": {"values": [2, 3, 4]},
            "label_feats": {"values": [True, False]},
            "n_task_layers": {"values": [1, 2, 3, 4]},
            "n_fp_layers": {"values": [2]},
            # Their DBLP command enables the residual connection; the adapter defaulted
            # it off and never searched it.
            "residual": {"values": [True, False]},
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
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "node classification only")

        device = resolve_dgl_device(device)
        seed_everything(seed, device)

        args = _vendor_args(bundle.dataset, params)
        data = bundle.data
        use_label_feats = bool(params.get("label_feats", True))

        # One-hot over target nodes, populated at OUR training rows only -- never
        # validation or test. Mirrors main.py:210-214, but keyed to the campaign split.
        label_onehot = None
        if use_label_feats:
            train_rows = data.target_train_mask.nonzero(as_tuple=False).view(-1)
            label_onehot = torch.zeros(
                (int(data.target_y.shape[0]), bundle.num_classes), dtype=torch.float32
            )
            if bundle.is_multilabel:
                label_onehot[train_rows] = data.target_y[train_rows].float()
            else:
                label_onehot[train_rows] = torch.nn.functional.one_hot(
                    data.target_y[train_rows].long(), bundle.num_classes
                ).float()

        (
            feats,
            label_feats,
            _init_labels,
            num_classes,
            _trainval_nid,
            test_nid,
        ) = _precompute_features(args, device, label_onehot=label_onehot)
        expected = int(data.target_y.shape[0])
        first = next(iter(feats.values()))
        if int(first.shape[0]) != expected:
            raise RuntimeError(
                f"SeHGNN propagated {first.shape[0]} target rows for {bundle.dataset}, "
                f"the project bundle has {expected}."
            )

        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()
        train_idx = data.target_train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = data.target_val_mask.nonzero(as_tuple=False).view(-1)
        test_idx = data.target_test_mask.nonzero(as_tuple=False).view(-1)

        feats = {key: value.to(device) for key, value in feats.items()}
        label_feats = {key: value.to(device) for key, value in label_feats.items()}
        data_size = {key: int(value.size(-1)) for key, value in feats.items()}

        with pushd_sys_path(*VENDORS):
            from model import SeHGNN  # type: ignore[import-not-found]

            model = SeHGNN(
                args.dataset,
                args.embed_size,
                args.hidden,
                num_classes,
                feats.keys(),
                label_feats.keys(),
                TARGET_TYPE[args.dataset],
                args.dropout,
                args.input_drop,
                args.att_drop,
                args.n_fp_layers,
                args.n_task_layers,
                args.act,
                args.residual,
                data_size=data_size,
            ).to(device)

        criterion = (
            torch.nn.BCEWithLogitsLoss()
            if bundle.is_multilabel
            else torch.nn.CrossEntropyLoss()
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        def forward(indices: torch.Tensor) -> torch.Tensor:
            # SeHGNN.forward(batch, feature_dict, label_dict, mask); the leading `batch`
            # carries the node ids.
            batch_feats = {key: value[indices] for key, value in feats.items()}
            batch_labels = {key: value[indices] for key, value in label_feats.items()}
            return model(indices, batch_feats, batch_labels, None)

        def evaluate(indices: torch.Tensor) -> dict[str, float]:
            model.eval()
            with torch.no_grad():
                logits = forward(indices.to(device))
            return classification_metrics(
                logits, labels[indices.to(device)], is_multilabel=bundle.is_multilabel
            )

        budget = resolve_budget(NODE_CLASSIFICATION, params)
        epochs, patience = budget.epochs, budget.patience
        batch_size = args.batch_size
        best_monitor = float("-inf")
        best_epoch, stale = 0, 0
        best_val: dict[str, float] = {}
        best_test: dict[str, float] = {}

        for epoch in range(epochs):
            model.train()
            permutation = train_idx[torch.randperm(train_idx.numel())]
            for start in range(0, permutation.numel(), batch_size):
                batch = permutation[start : start + batch_size].to(device)
                optimizer.zero_grad()
                loss = criterion(forward(batch), labels[batch])
                loss.backward()
                optimizer.step()

            val_metrics = evaluate(val_idx)
            test_metrics = evaluate(test_idx)
            monitor = val_metrics["macro_f1"]
            if monitor > best_monitor:
                best_monitor = monitor
                best_epoch, stale = epoch, 0
                best_val, best_test = val_metrics, test_metrics
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
        record["device"] = device.type
        record["num_metapaths"] = float(len(feats))
        record["num_label_metapaths"] = float(len(label_feats))
        return record


ADAPTER = SeHGNNAdapter()
