"""Node-classification F1 shared by H2S-NN and every baseline adapter (spec v1.15 §6; card Part M,
M8).

Macro-F1 averaging rule (spec 1.14, every NC dataset, every method): the class list `L` is fixed
once per split from that split's true labels and passed explicitly,
``f1_score(y_true, y_pred, average="macro", labels=L)``; a class absent from the split's true
labels is never averaged, whatever a model predicts. Multi-label (IMDB): a genre counts when at
least one node of the split carries it. Micro-F1 is not restricted (on single-label data it
equals accuracy, spec §6).
"""

from __future__ import annotations

import numpy as np


def split_classes(targets: np.ndarray, *, is_multilabel: bool) -> list[int]:
    """The classes present in one split's true labels: single-label = the distinct labels;
    multi-label = the columns with at least one positive."""
    if is_multilabel:
        return [int(c) for c in np.flatnonzero(np.asarray(targets).sum(axis=0) > 0)]
    return sorted(int(c) for c in np.unique(np.asarray(targets)))


def node_classification_f1(targets: np.ndarray, predictions: np.ndarray, *,
                           is_multilabel: bool) -> dict[str, float]:
    """Macro-F1 over the split's classes (``labels=L``) and unrestricted micro-F1."""
    from sklearn.metrics import f1_score

    classes = split_classes(targets, is_multilabel=is_multilabel)
    if not classes:
        return {"macro_f1": 0.0, "micro_f1": 0.0}
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", labels=classes,
                                   zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
    }


__all__ = ["node_classification_f1", "split_classes"]
