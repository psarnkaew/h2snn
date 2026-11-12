"""Training loops shared by the baseline adapters.

Adapters build their own model and optimizer -- that is where the baselines genuinely
differ -- and hand the loop a closure that produces logits or embeddings. Everything the
comparison requires to be identical across models lives here: the stopping rule, the
selection criterion, and which epoch's metrics get reported.

Selection criteria follow ``src/utils/run_reports.py: default_monitor_for_task``:
validation macro-F1 for node classification, validation AUROC for link prediction (spec
v1.15 §6, J4). Graph classification follows HetSheaf's protocol (card Part M, M5/M9): early
stopping on the validation loss, test with the final weights; the validation macro-F1 of those
weights selects configurations. Each is the metric that task's table leads with, because selecting
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
from src.models.anchored_solve import check_finite_loss


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
        check_finite_loss(loss, epoch)  # spec v1.16 K5: the trial fails
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
    epoch: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training negatives of one epoch: HGB's get_train_neg (spec v1.16 K3), the same function
    H2S-NN's LP trainer calls (`src.data.hgb_lp_negatives.hgb_epoch_train_negatives`), seed
    run seed + epoch. Validation and test negatives are the fixed files of the bundle."""

    from src.data.hgb_lp_negatives import hgb_epoch_train_negatives

    negatives, relations = hgb_epoch_train_negatives(data, epoch)
    return negatives.to(device), relations.to(device)


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

    from src.utils.misc import compute_link_prediction_metrics, lp_test_set_metrics

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

        neg_edge_index, neg_relation = sample_train_negatives(data, device=device, epoch=epoch)
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
        check_finite_loss(loss, epoch)  # spec v1.16 K5: the trial fails
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
            # HGB's two test sets (spec v1.16 K1): 2-hop and random; an empty set -> None.
            test_metrics = lp_test_set_metrics(
                lambda edges, relation: head(embeddings, edges, relation), data)

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
    """Train an encoder plus the shared head under HetSheaf's protocol (spec v1.15 §6 GC): early
    stopping on the validation loss (patience from the budget, strict improvement), and the
    reported record is the final weights' (validation macro-F1 of those weights selects
    configurations).

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
    best_loss = float("inf")
    final_epoch = -1
    final_val: dict[str, float] = {}
    final_test: dict[str, float] = {}
    stale = 0

    def run_split(loader: Any) -> dict[str, float]:
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits_all.append(head(embed_batch(batch), batch.batch))
                labels_all.append(batch.y.view(-1))
        logits, labels = torch.cat(logits_all), torch.cat(labels_all)
        return {"loss": float(criterion(logits, labels)),
                **compute_graph_classification_metrics(logits, labels)}

    for epoch in range(epochs):
        model.train()
        head.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(head(embed_batch(batch), batch.batch), batch.y.view(-1))
            if aux_loss is not None:
                loss = loss + aux_loss()
            check_finite_loss(loss, epoch)  # spec v1.16 K5: the trial fails
            loss.backward()
            optimizer.step()

        model.eval()
        head.eval()
        val_metrics = run_split(val_loader)
        test_metrics = run_split(test_loader)

        # HetSheaf: early stopping on the validation loss; the final weights are reported.
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            stale = 0
        else:
            stale += 1
        final_epoch, final_val, final_test = epoch, val_metrics, test_metrics

        report_epoch(
            report,
            epoch=epoch,
            monitor=val_metrics["macro_f1"],
            running_best=val_metrics["macro_f1"],
            metrics=val_metrics,
        )
        if stale >= patience:
            break

    if not final_val:
        raise RuntimeError("Graph-classification training produced no evaluated epoch.")
    record: dict[str, float] = {}
    record.update(prefixed(final_val, "val"))
    record.update(prefixed(final_test, "test"))
    record["best_epoch"] = float(final_epoch)  # the reported (final) weights' epoch
    record["final_epoch"] = float(final_epoch)
    return record


__all__ = [
    "node_classification_criterion",
    "sample_train_negatives",
    "train_full_batch_node_classification",
    "train_graph_classification",
    "train_link_prediction",
]
