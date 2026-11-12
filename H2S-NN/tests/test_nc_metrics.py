"""Part M, M8: Macro-F1 over the classes of each split's true labels (spec v1.15 §6), one module
for H2S-NN and every baseline adapter."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.baselines.metrics import classification_metrics
from src.utils.misc import (compute_classification_metrics,
                            compute_graph_classification_metrics,
                            compute_multilabel_classification_metrics)
from src.utils.nc_metrics import node_classification_f1, split_classes


def _logits_for(predictions: list[int], num_classes: int) -> torch.Tensor:
    logits = torch.full((len(predictions), num_classes), -5.0)
    logits[torch.arange(len(predictions)), torch.tensor(predictions)] = 5.0
    return logits


class SplitClassTests(unittest.TestCase):
    def test_a_class_absent_from_the_split_is_not_averaged(self) -> None:
        """True labels {0, 1}; the model predicts class 2 once. F1(0) = 2/3, F1(1) = 1, so
        Macro-F1 over L = {0, 1} is 5/6; sklearn's default (classes of truth or prediction)
        would average a third class with F1 0 -> 5/9."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 2, 1, 1]
        self.assertEqual(split_classes(np.array(y_true), is_multilabel=False), [0, 1])
        expected = (2 / 3 + 1.0) / 2
        ours = node_classification_f1(np.array(y_true), np.array(y_pred), is_multilabel=False)
        self.assertAlmostEqual(ours["macro_f1"], expected, places=12)
        self.assertAlmostEqual(ours["micro_f1"], 3 / 4, places=12)  # = accuracy (unrestricted)
        h2snn = compute_classification_metrics(_logits_for(y_pred, 3), torch.tensor(y_true))
        baseline = classification_metrics(_logits_for(y_pred, 3), torch.tensor(y_true),
                                          is_multilabel=False)
        for record in (h2snn, baseline):
            self.assertAlmostEqual(record["macro_f1"], expected, places=12)
            self.assertAlmostEqual(record["micro_f1"], record["acc"], places=12)

    def test_imdb_genre_absent_from_the_split_is_not_averaged(self) -> None:
        """Genre 2 has no positive in the split; the model predicts it for one movie. Macro-F1 is
        the mean over genres 0, 1, 3."""
        y_true = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
        y_pred = np.array([[1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(split_classes(y_true, is_multilabel=True), [0, 1, 3])
        f1_0 = 1.0
        f1_1 = 2 * 1 / (2 * 1 + 0 + 1)      # tp 1, fp 0, fn 1
        f1_3 = 2 * 1 / (2 * 1 + 0 + 1)      # tp 1, fp 0, fn 1
        expected = (f1_0 + f1_1 + f1_3) / 3
        ours = node_classification_f1(y_true, y_pred, is_multilabel=True)
        self.assertAlmostEqual(ours["macro_f1"], expected, places=12)
        logits = torch.tensor(y_pred, dtype=torch.float) * 10 - 5
        h2snn = compute_multilabel_classification_metrics(logits, torch.tensor(y_true))
        baseline = classification_metrics(logits, torch.tensor(y_true), is_multilabel=True)
        for record in (h2snn, baseline):
            self.assertAlmostEqual(record["macro_f1"], expected, places=12)

    def test_every_class_present_changes_nothing(self) -> None:
        """On a split that contains every class (HGB's case) the value is sklearn's default."""
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 4, 200)
        y_true[:4] = [0, 1, 2, 3]
        y_pred = rng.integers(0, 4, 200)
        ours = node_classification_f1(y_true, y_pred, is_multilabel=False)
        self.assertAlmostEqual(ours["macro_f1"], f1_score(y_true, y_pred, average="macro"))

    def test_graph_classification_metrics_unchanged(self) -> None:
        """M8 is node classification; GC uses HetSheaf's torchmetrics call (Part N, N3), which
        gives sklearn's default value on this example (class 2 predicted but absent: F1 0)."""
        from sklearn.metrics import f1_score

        y_true, y_pred = [0, 0, 1, 1], [0, 2, 1, 1]
        record = compute_graph_classification_metrics(_logits_for(y_pred, 3), torch.tensor(y_true))
        self.assertAlmostEqual(record["macro_f1"],
                               f1_score(y_true, y_pred, average="macro", zero_division=0))


if __name__ == "__main__":
    unittest.main()
