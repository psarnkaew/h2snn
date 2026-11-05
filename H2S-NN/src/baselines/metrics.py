"""Metric computation shared by every baseline adapter.

Uses the same definitions as ``src/trainers/`` so a baseline row and an H2S-NN row in the
same table are the same measurement. Record keys match
``src.utils.run_reports.report_metrics_for_task`` plus the validation key that
``default_monitor_for_task`` selects on.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


# How multi-label predictions (HGB IMDB) are binarised before F1. The two conventions are
# not comparable, and the literature is split:
#
#   "threshold"    -- predict every class with a positive logit. Used by HGB's own runners
#                     (HGB/NC/benchmark/methods/GNN/run_multi.py:141), by SeHGNN
#                     (SeHGNN/hgb/utils.py:228), and by this project's H2S-NN
#                     (src/utils/misc.py). Consumes no test-set information.
#
#   "oracle_topk"  -- predict the k highest-scoring classes, where k is the node's *true*
#                     label count. Used by PSHGCN (PSHGCN/hgb/utils.py:Evaluation) and by
#                     H2SGNN, which copied it. It reads the ground-truth cardinality of
#                     each test node, so precision and recall are equal by construction
#                     and micro-F1 degenerates to "fraction of labels recovered". On a
#                     fixed trained PSHGCN it scores ~9 macro-F1 points higher than
#                     "threshold" on IMDB.
#
# NB: HGB's own evaluate() (HGB/NC/benchmark/scripts/data_loader.py:182) is metric-only --
# it takes an already-binarised prediction. So the benchmark mandates neither convention;
# calling top-k "the HGB convention" is wrong.
#
# The campaign uses "threshold" for every row so the IMDB column is internally consistent
# and no row is scored using its own test labels. "oracle_topk" is kept so PSHGCN's and
# H2SGNN's published numbers can be reproduced and cited as such.
MULTILABEL_PROTOCOLS = ("threshold", "oracle_topk")
DEFAULT_MULTILABEL_PROTOCOL = "threshold"


def binarize_multilabel(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    protocol: str = DEFAULT_MULTILABEL_PROTOCOL,
) -> torch.Tensor:
    """Turn multi-label logits into a 0/1 prediction matrix under ``protocol``."""

    if protocol == "threshold":
        return (logits > 0).long()
    if protocol != "oracle_topk":
        raise ValueError(
            f"multilabel protocol must be one of {MULTILABEL_PROTOCOLS}, got {protocol!r}."
        )

    # Vectorised equivalent of PSHGCN/hgb/utils.py:Evaluation -- rank the logits per node
    # and keep as many as that node truly has.
    counts = labels.long().sum(dim=1)
    order = logits.argsort(dim=1, descending=True)
    ranks = order.argsort(dim=1)
    # Their `preds[i].argsort()[-k:]` selects the whole row when k == 0, because Python's
    # [-0:] is [0:]. Every HGB IMDB node carries at least one label so that branch is
    # unreachable upstream; predicting nothing is the correct reading of "top 0".
    return (ranks < counts.unsqueeze(1)).long()


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    is_multilabel: bool,
    multilabel_protocol: str = DEFAULT_MULTILABEL_PROTOCOL,
) -> dict[str, float]:
    """Macro-F1, micro-F1 and accuracy for one split.

    Single-label splits take an argmax. Multi-label splits are binarised by
    :func:`binarize_multilabel`; the protocol must be the same for every row of a table,
    since the two conventions are not comparable.
    """

    with torch.no_grad():
        if is_multilabel:
            prediction_tensor = binarize_multilabel(
                logits, labels, protocol=multilabel_protocol
            )
            predictions = prediction_tensor.cpu().numpy()
            targets = labels.long().cpu().numpy()
            accuracy = float((predictions == targets).all(axis=1).mean())
        else:
            predictions = logits.argmax(dim=-1).cpu().numpy()
            targets = labels.cpu().numpy()
            accuracy = float((predictions == targets).mean())

    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "acc": accuracy,
    }


def link_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """AUROC and AUPR for one link-prediction split."""

    with torch.no_grad():
        score_array = scores.detach().float().cpu().numpy()
        label_array = labels.detach().float().cpu().numpy()

    if label_array.min() == label_array.max():
        # Degenerate split: both metrics are undefined rather than zero. Surfacing 0.0
        # here would look like a real (bad) result in the table.
        return {"auroc": float("nan"), "aupr": float("nan")}

    return {
        "auroc": float(roc_auc_score(label_array, score_array)),
        "aupr": float(average_precision_score(label_array, score_array)),
    }


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Re-key ``{"macro_f1": ...}`` as ``{"val_macro_f1": ...}`` for the run record."""

    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def node_classification_record(
    *,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_epoch: int,
) -> dict[str, float]:
    record: dict[str, float] = {}
    record.update(prefixed(val_metrics, "val"))
    record.update(prefixed(test_metrics, "test"))
    record["best_epoch"] = float(best_epoch)
    return record


def as_float_labels(labels: torch.Tensor, *, is_multilabel: bool) -> torch.Tensor:
    return labels.float() if is_multilabel else labels.long()


def stack_scores(
    positive: torch.Tensor, negative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate positive/negative scores with their 1/0 labels."""

    scores = torch.cat([positive, negative])
    labels = torch.cat(
        [torch.ones_like(positive), torch.zeros_like(negative)]
    )
    return scores, labels


__all__ = [
    "as_float_labels",
    "classification_metrics",
    "link_metrics",
    "node_classification_record",
    "prefixed",
    "stack_scores",
]
