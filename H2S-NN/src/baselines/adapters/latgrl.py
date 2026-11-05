"""LatGRL (Shen & Kang, TNNLS 2025) -- latent graphs guided unsupervised representation learning.

Structurally unlike every other baseline here: LatGRL trains an encoder with **no labels**,
then evaluates a probe on the frozen embeddings. So it does not use
``train_full_batch_node_classification``; the adapter runs their unsupervised objective and
then fits the probe on our canonical split.

Their loader reads preprocessed ``.npz`` caches of the HAN-style datasets, which this
project does not have and which are different graphs from HGB's for ACM and IMDB. That is
not a real obstacle: their method needs only four things -- a target-node feature matrix, a
list of meta-path adjacencies over target nodes, labels, and split indices
(``LatGRL/utils/load_data.py``) -- and all four are derivable from the canonical bundle.
``graph_construction(x, adjs, ...)`` is type-agnostic, so the meta-path adjacencies are
built here from HGB's graph and handed to their unchanged code.

Deviations, recorded on the run record:

1. **Dataset version.** Their published ACM/IMDB are the HAN-preprocessed graphs (ACM 4019
   papers / 3 classes, IMDB 4278 movies / 3 classes); ours are HGB's (ACM 3025, IMDB 4932 /
   5 classes multi-label). DBLP is the same graph in both. Numbers here are therefore not
   directly comparable to their paper.
2. **Split and probe.** Their protocol is 20/40/60 labelled nodes per class with a
   50-restart logistic-regression probe (``utils/evaluate.py``). We use the campaign's HGB
   split, which gives the probe *more* supervision, and select on validation macro-F1.
3. **Multi-label IMDB.** Their ``evaluate`` does ``argmax`` with ``CrossEntropyLoss``,
   which is wrong for HGB's multi-label IMDB. The probe here uses BCE-with-logits and the
   campaign's ``threshold`` binarisation when the task is multi-label.

Link prediction follows the same freeze-then-probe shape, with the shared decoder as the
probe -- see :mod:`src.baselines.adapters._latgrl_lp`, including why only Amazon and
PubMed are covered.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from src.baselines.adapters._latgrl_lp import (
    SUPPORTED_LP_DATASETS,
    train_unsupervised,
)
from src.baselines.bundles import LinkBundle, NodeBundle
from src.baselines.heads import SharedLinkHead
from src.baselines.metrics import classification_metrics, node_classification_record
from src.baselines.budget import budget_kwargs, resolve_budget
from src.baselines import search_grids as grids
from src.baselines.protocol import (
    LINK_PREDICTION,
    NODE_CLASSIFICATION,
    BaselineParams,
    UnsupportedCell,
)
from src.baselines.published import load_published_params
from src.baselines.runtime import pushd_sys_path
from src.baselines.training import train_link_prediction
from src.utils.misc import seed_everything

NAME = "latgrl"
VENDOR = "latgrl"

#: Meta-paths over the target node type, matching LatGRL's Table IV for each dataset,
#: expressed in HGB's node-type names.
METAPATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "acm": (("paper", "author", "paper"), ("paper", "subject", "paper")),
    "dblp": (
        ("author", "paper", "author"),
        ("author", "paper", "venue", "paper", "author"),
        ("author", "paper", "term", "paper", "author"),
    ),
    "imdb": (("movie", "actor", "movie"), ("movie", "director", "movie")),
}


def _relation_adjacency(
    bundle: NodeBundle, source_type: str, destination_type: str
) -> torch.Tensor:
    """Sparse [n_src, n_dst] adjacency for one typed relation, in type-local indices."""

    data = bundle.data
    slices = dict(data.node_slices)
    source_start, source_end = slices[source_type]
    destination_start, destination_end = slices[destination_type]

    row, column = data.edge_index
    mask = (
        (row >= source_start)
        & (row < source_end)
        & (column >= destination_start)
        & (column < destination_end)
    )
    local_row = row[mask] - source_start
    local_column = column[mask] - destination_start
    values = torch.ones(local_row.numel(), dtype=torch.float32)
    return torch.sparse_coo_tensor(
        torch.stack([local_row, local_column]),
        values,
        (source_end - source_start, destination_end - destination_start),
    ).coalesce()


def build_metapath_adjacencies(bundle: NodeBundle) -> list[torch.Tensor]:
    """Compose per-relation adjacencies into meta-path adjacencies over target nodes.

    A meta-path such as ``paper-author-paper`` is the product of the paper->author and
    author->paper adjacencies, giving a symmetric [n_target, n_target] matrix -- exactly
    the ``A^Phi`` of the paper's Eq. 7, and the shape their ``.npz`` caches hold.
    """

    dataset = bundle.dataset.lower()
    if dataset not in METAPATHS:
        raise UnsupportedCell(NAME, NODE_CLASSIFICATION, dataset, "no meta-paths defined")

    adjacencies: list[torch.Tensor] = []
    for path in METAPATHS[dataset]:
        product = _relation_adjacency(bundle, path[0], path[1])
        for step in range(1, len(path) - 1):
            product = torch.sparse.mm(
                product, _relation_adjacency(bundle, path[step], path[step + 1])
            )
        # Binarise: LatGRL treats these as unweighted meta-path subgraphs.
        product = product.coalesce()
        adjacencies.append(
            torch.sparse_coo_tensor(
                product.indices(),
                torch.ones_like(product.values()),
                product.shape,
            ).coalesce()
        )
    return adjacencies


def _fit_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    splits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    is_multilabel: bool,
    num_classes: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict[str, float], dict[str, float], int]:
    """Logistic-regression probe on frozen embeddings, selected on validation macro-F1.

    Mirrors the shape of ``LatGRL/utils/evaluate.py`` -- a single linear layer over frozen
    embeddings -- but uses BCE for multi-label targets, which theirs does not support.
    """

    train_idx, val_idx, test_idx = splits
    probe = torch.nn.Linear(embeddings.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = (
        torch.nn.BCEWithLogitsLoss() if is_multilabel else torch.nn.CrossEntropyLoss()
    )

    best_monitor = float("-inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    best_test: dict[str, float] = {}

    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        criterion(probe(embeddings[train_idx]), labels[train_idx]).backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            logits = probe(embeddings)
        val_metrics = classification_metrics(
            logits[val_idx], labels[val_idx], is_multilabel=is_multilabel
        )
        if val_metrics["macro_f1"] > best_monitor:
            best_monitor = val_metrics["macro_f1"]
            best_epoch = epoch
            best_val = val_metrics
            best_test = classification_metrics(
                logits[test_idx], labels[test_idx], is_multilabel=is_multilabel
            )
    return best_val, best_test, best_epoch


class LatGRLAdapter:
    name = NAME

    def supports(self, task: str, dataset: str) -> bool:
        if task == NODE_CLASSIFICATION:
            return dataset in METAPATHS
        if task == LINK_PREDICTION:
            return dataset in SUPPORTED_LP_DATASETS
        return False

    def search_space(self, task: str, dataset: str) -> dict[str, Any]:
        if task not in (NODE_CLASSIFICATION, LINK_PREDICTION):
            raise UnsupportedCell(NAME, task, dataset, "unsupported task")
        # Ranges from their utils/params.py and paper section V-A. The encoder is
        # unsupervised, so the same space serves both tasks; link prediction adds the
        # decoder choice, which is the only supervised component in that cell.
        space = {
            "lr": grids.LR,
            "weight_decay": grids.WEIGHT_DECAY,
            "hidden_dim": {"values": [128, 256]},
            "dropout": grids.DROPOUT,
            "tau": {"values": [0.4, 0.7, 1.0]},
            "graph_k": {"values": [3, 5, 10]},
            "k_pos": {"values": [0, 1, 2, 3, 4]},
            "alpha": {"values": [1.0, 2.0]},
            "nlayer": {"values": [1, 2, 3]},
            # The unsupervised stage and the probe are where LatGRL's published settings
            # actually differ per dataset (their utils/params.py); none of these were
            # searched, so no tuned cell could reach the configuration they report.
            "nb_epochs": {"values": [200, 500, 1000]},
            "drop_feat": grids.DROPOUT,
            "eva_lr": {"values": [0.01, 0.03, 0.05]},
            "eva_wd": {"values": [1e-4, 1e-3, 1e-2]},
        }
        if task == LINK_PREDICTION:
            # distmult only, unlike every other link-prediction cell. LatGRL's encoder is
            # frozen after its unsupervised stage, and the dot decoder has no parameters,
            # so `dot` would leave nothing at all to train -- not a probe but a fixed
            # scoring function. distmult is the parameterised decoder, which is what
            # their own freeze-then-probe protocol calls for.
            space["decoder_type"] = {"values": ["distmult"]}
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
        if task == LINK_PREDICTION:
            return self._fit_link_prediction(bundle, params, seed=seed, device=device)
        if task != NODE_CLASSIFICATION:
            raise UnsupportedCell(NAME, task, bundle.dataset, "unsupported task")

        seed_everything(seed, device)
        data = bundle.data

        # Target-node features and meta-path adjacencies -- the only two inputs their
        # method needs beyond labels and splits.
        target_start, target_end = dict(data.node_slices)[data.target_node_type]
        feature_width = dict(data.feature_dims)[data.target_node_type]
        feat = data.x[target_start:target_end, :feature_width].to(device)
        adjs = [adj.to(device) for adj in build_metapath_adjacencies(bundle)]

        args = SimpleNamespace(
            graph_k=int(params.get("graph_k", 5)),
            k_pos=int(params.get("k_pos", 0)),
            dataset=bundle.dataset,
        )

        with pushd_sys_path(VENDOR):
            from module.graph_generating import graph_process  # type: ignore[import-not-found]
            from module.LatGRL import LatGRL  # type: ignore[import-not-found]

            adjs_l, adjs_h, adjs_o, pos = graph_process(adjs, feat, args)
            model = LatGRL(
                int(feat.size(1)),
                len(adjs),
                int(params.get("hidden_dim", 128)),
                int(params.get("embed_dim", 64)),
                float(params.get("tau", 1.0)),
                float(params.get("dropout", 0.2)),
                torch.nn.ELU(),
                float(params.get("drop_feat", 0.1)),
                int(feat.size(0)),
                bundle.dataset,
                float(params.get("alpha", 2.0)),
                int(params.get("nlayer", 2)),
            ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params.get("lr", 5e-4)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )

        # Stage 1: unsupervised. Early stopping on the training objective, as their
        # main.py does -- no labels are involved here at all.
        best_loss = float("inf")
        stale = 0
        patience = int(params.get("patience", 10))
        for _ in range(int(params.get("nb_epochs", 200))):
            model.train()
            optimizer.zero_grad()
            loss = model(feat, adjs_l, adjs_h, adjs_o, pos[0])
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        # Stage 2: probe the frozen embeddings on the canonical split.
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embeds(feat, adjs_o).detach()

        labels = data.target_y.to(device)
        if bundle.is_multilabel:
            labels = labels.float()
        splits = (
            data.target_train_mask.nonzero(as_tuple=False).view(-1).to(device),
            data.target_val_mask.nonzero(as_tuple=False).view(-1).to(device),
            data.target_test_mask.nonzero(as_tuple=False).view(-1).to(device),
        )
        best_val, best_test, best_epoch = _fit_probe(
            embeddings,
            labels,
            splits,
            is_multilabel=bundle.is_multilabel,
            num_classes=bundle.num_classes,
            device=device,
            # Their published configs name the probe budget `probe_epochs`; a tuned run
            # has no such key and gets the campaign's shared node-classification budget,
            # like every other baseline.
            epochs=int(
                params.get(
                    "probe_epochs", resolve_budget(NODE_CLASSIFICATION, params).epochs
                )
            ),
            learning_rate=float(params.get("eva_lr", 0.05)),
            weight_decay=float(params.get("eva_wd", 1e-3)),
        )

        record = node_classification_record(
            val_metrics=best_val, test_metrics=best_test, best_epoch=best_epoch
        )
        record["params_source"] = params.source
        record["unsupervised_loss"] = float(best_loss)
        record["num_metapaths"] = float(len(adjs))
        record["protocol"] = "unsupervised_then_probe"
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

        embeddings, best_loss, num_metapaths = train_unsupervised(
            bundle, params, device
        )
        embed_dim = int(embeddings.size(1))

        data = bundle.data.to(device)
        decoder_type = str(params.get("decoder_type", "distmult"))
        if decoder_type == "dot":
            raise UnsupportedCell(
                NAME,
                LINK_PREDICTION,
                bundle.dataset,
                "the dot decoder has no parameters, and LatGRL's encoder is frozen, so "
                "this combination has nothing to train",
            )
        head = SharedLinkHead(
            decoder_type,
            num_relations=max(int(data.num_target_relations), 1),
            dim=embed_dim,
        ).to(device)
        optimizer = torch.optim.Adam(
            head.parameters(),
            lr=float(params.get("eva_lr", 0.05)),
            weight_decay=float(params.get("eva_wd", 1e-3)),
        )

        # The encoder is frozen: only the decoder trains, which is their own
        # freeze-then-probe protocol with the campaign's probe.
        #
        # Their published configs name the probe budget `probe_epochs`; a tuned run has
        # no such key and takes the campaign's shared link-prediction budget, like every
        # other baseline.
        probe_budget = budget_kwargs(LINK_PREDICTION, params)
        probe_budget["epochs"] = int(
            params.get("probe_epochs", probe_budget["epochs"])
        )
        record = train_link_prediction(
            model=head,
            head=head,
            embed=lambda: embeddings,
            data=data,
            optimizer=optimizer,
            device=device,
            **probe_budget,
        )
        record["params_source"] = params.source
        record["unsupervised_loss"] = float(best_loss)
        record["num_metapaths"] = float(num_metapaths)
        record["protocol"] = "unsupervised_then_probe"
        record["decoder_type"] = decoder_type
        return record


ADAPTER = LatGRLAdapter()
