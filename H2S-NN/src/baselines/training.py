"""Training loops shared by the baseline adapters.

Adapters build their own model and optimizer -- that is where the baselines genuinely
differ -- and hand the loop a closure that produces logits or embeddings. Everything the
comparison requires to be identical across models lives here: the stopping rule, the
selection criterion, and which epoch's metrics get reported.

Selection criteria follow ``src/utils/run_reports.py: default_monitor_for_task``:
validation macro-F1 for node classification *and graph classification*, validation AUROC
for link prediction. Each is the metric that task's table leads with, because selecting
on one metric and reporting another is how a model ends up tuned for something the paper
never shows. On a multi-relation link target the monitor is the per-relation macro
average, which is what every HGB link-prediction runner reports.

Several vendored trainers select on validation loss instead (PSHGCN/hgb/main.py:53,
H2SGNN/main.py:53); using each author's own criterion would make the tables a comparison
of stopping rules as much as of models. Adapters record the deviation on the run record.

Epochs and patience are not set here or by an adapter -- they come from
:mod:`src.baselines.budget`, so every model in a table trains under the same budget.

Link prediction and graph classification additionally attach a head from
:mod:`src.baselines.heads`, because most of these models were published as node
classifiers only. The head is trained jointly with the encoder, not as a frozen probe.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from src.baselines.metrics import (
    DEFAULT_MULTILABEL_PROTOCOL,
    classification_metrics,
    node_classification_record,
    prefixed,
)
from src.baselines.progress import EpochReport, report_epoch


# --------------------------------------------------------------------------------------
# Node classification
# --------------------------------------------------------------------------------------


def train_full_batch_node_classification(
    *,
    model: torch.nn.Module,
    forward: Callable[[], torch.Tensor],
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    is_multilabel: bool,
    epochs: int,
    patience: int,
    num_target: int | None = None,
    multilabel_protocol: str = DEFAULT_MULTILABEL_PROTOCOL,
    report: EpochReport | None = None,
) -> dict[str, float]:
    """Full-graph transductive training with early stopping on validation macro-F1.

    ``forward`` returns logits for the whole flattened node set; ``num_target`` truncates
    to the target type when the model emits rows for every node type (the HGB polynomial
    models do).
    """

    best_monitor = float("-inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    best_test: dict[str, float] = {}
    stale = 0

    def target_logits() -> torch.Tensor:
        logits = forward()
        return logits if num_target is None else logits[:num_target]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(target_logits()[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = target_logits()
            val_metrics = classification_metrics(
                logits[val_idx],
                labels[val_idx],
                is_multilabel=is_multilabel,
                multilabel_protocol=multilabel_protocol,
            )
            test_metrics = classification_metrics(
                logits[test_idx],
                labels[test_idx],
                is_multilabel=is_multilabel,
                multilabel_protocol=multilabel_protocol,
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
            report,
            epoch=epoch,
            monitor=monitor,
            running_best=best_monitor,
            metrics=val_metrics,
        )
        if stale >= patience:
            break

    if not best_val:
        raise RuntimeError(
            "Training produced no evaluated epoch; check epochs/patience settings."
        )
    return node_classification_record(
        val_metrics=best_val, test_metrics=best_test, best_epoch=best_epoch
    )


def node_classification_criterion(
    is_multilabel: bool,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """BCE-with-logits for the multi-label HGB IMDB, cross-entropy otherwise."""

    if is_multilabel:
        return torch.nn.BCEWithLogitsLoss()
    return torch.nn.functional.cross_entropy


# --------------------------------------------------------------------------------------
# Link prediction
# --------------------------------------------------------------------------------------


def sample_train_negatives(
    data: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw training negatives, mirroring src/trainers/link_prediction_trainer.py.

    Negatives are drawn per pooled target relation and tagged with their relation id, so
    a multi-relation target such as Amazon (co-view plus co-purchase) gets each relation's
    negatives from its own positive count rather than pooling them into one untyped
    sample.

    Validation and test negatives are deliberately NOT drawn here: those are the fixed
    HGB two-hop hard negatives already in the bundle, identical for every model.
    """

    from torch_geometric.utils import negative_sampling

    pos = data.train_pos_edge_index
    relations = data.train_pos_edge_relation
    ratio = int(data.neg_sampling_ratio)

    neg_parts: list[torch.Tensor] = []
    relation_parts: list[torch.Tensor] = []
    for relation_id in relations.unique().tolist():
        subset = pos[:, relations == relation_id]
        needed = subset.size(1) * ratio
        if data.train_neg_candidates is not None:
            pool = data.train_neg_candidates
            index = torch.randint(0, pool.size(1), (needed,), device=pool.device)
            sampled = pool[:, index]
        else:
            local = negative_sampling(
                torch.stack(
                    [
                        subset[0] - data.src_node_offset,
                        subset[1] - data.dst_node_offset,
                    ]
                ),
                num_nodes=(data.num_src_nodes, data.num_dst_nodes),
                num_neg_samples=needed,
            )
            sampled = torch.stack(
                [local[0] + data.src_node_offset, local[1] + data.dst_node_offset]
            )
        neg_parts.append(sampled.to(device))
        relation_parts.append(
            torch.full((sampled.size(1),), relation_id, dtype=torch.long, device=device)
        )

    return torch.cat(neg_parts, dim=1), torch.cat(relation_parts, dim=0)


def train_link_prediction(
    *,
    model: torch.nn.Module,
    head: torch.nn.Module,
    embed: Callable[[], torch.Tensor],
    data: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    aux_loss: Callable[[], torch.Tensor] | None = None,
    report: EpochReport | None = None,
) -> dict[str, float]:
    """Train an encoder plus the shared link decoder, selecting on validation AUROC.

    ``embed`` returns node embeddings for the whole graph; the head scores candidate
    pairs. Both are optimised together, so the decoder is not a frozen probe.

    On targets spanning several relations (Amazon, and YouTube once all five are
    targeted) selection and reporting use the per-relation macro average, which is what
    every HGB link-prediction runner reports; the pooled figures are recorded alongside.

    ``aux_loss`` lets a baseline whose method includes a task-independent regulariser
    keep it -- Hetero2Net's disentanglement term is added to every batch of its own
    training loop, so dropping it here would be running a different model. It is called
    after ``embed()`` in the same graph, so it can read whatever that pass cached.
    """

    from src.utils.misc import compute_link_prediction_metrics

    criterion = torch.nn.BCEWithLogitsLoss()
    best_monitor = float("-inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    best_test: dict[str, float] = {}
    stale = 0

    def evaluate(
        embeddings: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        pos_rel: torch.Tensor,
        neg_rel: torch.Tensor,
    ) -> dict[str, float]:
        return compute_link_prediction_metrics(
            head(embeddings, pos, pos_rel),
            head(embeddings, neg, neg_rel),
            pos,
            neg,
            pos_rel,
            neg_rel,
        )

    for epoch in range(epochs):
        model.train()
        head.train()
        optimizer.zero_grad()

        neg_edge_index, neg_relation = sample_train_negatives(data, device=device)
        embeddings = embed()
        pos_scores = head(
            embeddings, data.train_pos_edge_index, data.train_pos_edge_relation
        )
        neg_scores = head(embeddings, neg_edge_index, neg_relation)
        loss = criterion(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]),
        )
        if aux_loss is not None:
            loss = loss + aux_loss()
        loss.backward()
        optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            embeddings = embed()
            val_metrics = evaluate(
                embeddings,
                data.val_pos_edge_index,
                data.val_neg_edge_index,
                data.val_pos_edge_relation,
                data.val_neg_edge_relation,
            )
            test_metrics = evaluate(
                embeddings,
                data.test_pos_edge_index,
                data.test_neg_edge_index,
                data.test_pos_edge_relation,
                data.test_neg_edge_relation,
            )

        # HGB reports the per-relation (macro) average on multi-relation targets, so
        # select on that where it exists; it equals the pooled AUROC on single-relation
        # datasets, leaving LastFM and PubMed selection unchanged.
        monitor = val_metrics.get("auroc_macro", val_metrics["auroc"])
        if monitor > best_monitor:
            best_monitor = monitor
            best_epoch = epoch
            best_val = val_metrics
            best_test = test_metrics
            stale = 0
        else:
            stale += 1

        report_epoch(
            report,
            epoch=epoch,
            monitor=monitor,
            running_best=best_monitor,
            metrics=val_metrics,
        )
        if stale >= patience:
            break

    if not best_val:
        raise RuntimeError("Link-prediction training produced no evaluated epoch.")
    record: dict[str, float] = {}
    record.update(prefixed(best_val, "val"))
    record.update(prefixed(best_test, "test"))
    record["best_epoch"] = float(best_epoch)
    return record


# --------------------------------------------------------------------------------------
# Graph classification
# --------------------------------------------------------------------------------------


def train_graph_classification(
    *,
    model: torch.nn.Module,
    head: torch.nn.Module,
    embed_batch: Callable[[Any], torch.Tensor],
    loaders: tuple[Any, Any, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    aux_loss: Callable[[], torch.Tensor] | None = None,
    report: EpochReport | None = None,
) -> dict[str, float]:
    """Train an encoder plus the shared mean-pool head, selecting on validation macro-F1.

    ``embed_batch`` maps one PyG batch to per-node embeddings; the head pools them per
    graph. Minibatched over the fold's loaders, which carry one (repetition, fold) of
    the repeated 10-fold split described in :mod:`src.data.graph_classification_dataset`.

    ``aux_loss`` carries a baseline's own task-independent regulariser, as on link
    prediction. It is called once per batch after ``embed_batch``, so it can read whatever
    that pass cached.
    """

    from src.utils.misc import compute_graph_classification_metrics

    train_loader, val_loader, test_loader = loaders
    criterion = torch.nn.CrossEntropyLoss()
    best_monitor = float("-inf")
    best_epoch = 0
    best_val: dict[str, float] = {}
    best_test: dict[str, float] = {}
    stale = 0

    def run_split(loader: Any) -> dict[str, float]:
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits_all.append(head(embed_batch(batch), batch.batch))
                labels_all.append(batch.y.view(-1))
        return compute_graph_classification_metrics(
            torch.cat(logits_all), torch.cat(labels_all)
        )

    for epoch in range(epochs):
        model.train()
        head.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(head(embed_batch(batch), batch.batch), batch.y.view(-1))
            if aux_loss is not None:
                loss = loss + aux_loss()
            loss.backward()
            optimizer.step()

        model.eval()
        head.eval()
        val_metrics = run_split(val_loader)
        test_metrics = run_split(test_loader)

        # Macro-F1, matching what the graph-classification table reports. Selecting on
        # a different metric than the one reported is how a model ends up tuned for
        # something the paper never shows.
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
            report,
            epoch=epoch,
            monitor=monitor,
            running_best=best_monitor,
            metrics=val_metrics,
        )
        if stale >= patience:
            break

    if not best_val:
        raise RuntimeError("Graph-classification training produced no evaluated epoch.")
    record: dict[str, float] = {}
    record.update(prefixed(best_val, "val"))
    record.update(prefixed(best_test, "test"))
    record["best_epoch"] = float(best_epoch)
    return record


__all__ = [
    "node_classification_criterion",
    "sample_train_negatives",
    "train_full_batch_node_classification",
    "train_graph_classification",
    "train_link_prediction",
]
