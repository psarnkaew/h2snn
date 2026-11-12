"""Part N, N3: graph-classification Macro/Micro-F1 = HetSheaf's torchmetrics call (spec v1.16 K4):
`F1Score(task="multiclass", average="macro" | "micro", num_classes=C)` on the logits; used by
H2S-NN's GC trainer and the baselines' GC loop. NC metrics are unchanged (M8)."""

from __future__ import annotations

import inspect
import unittest

import torch
from sklearn.metrics import f1_score

from src.utils.misc import compute_graph_classification_metrics, hetsheaf_f1


def _logits(predictions: list[int], num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(predictions), num_classes).float() * 4 - 2


class HetSheafF1Tests(unittest.TestCase):
    def test_every_class_present_equals_sklearn(self) -> None:
        y = [0, 1, 2, 2, 1, 0, 2]
        p = [0, 2, 2, 1, 1, 0, 2]
        f1 = hetsheaf_f1(_logits(p, 3), torch.tensor(y))
        self.assertAlmostEqual(f1["macro_f1"], f1_score(y, p, average="macro"), places=6)
        self.assertAlmostEqual(f1["micro_f1"], f1_score(y, p, average="micro"), places=6)

    def test_class_absent_from_truth_and_prediction(self) -> None:
        """C = 3, class 2 in neither truth nor prediction. torchmetrics 1.8.2 (the version in
        h2snn, pinned to HetSheaf's uv.lock; Part N-fix F3) does not average it: macro =
        mean(F1 class 0 = 2/3, F1 class 1 = 4/5) = 11/15; micro = 3/4 (1.9.0 gave the same
        values). Recorded as torchmetrics computes it, whatever sklearn would do."""
        y, p = [0, 0, 1, 1], [0, 1, 1, 1]
        f1 = hetsheaf_f1(_logits(p, 3), torch.tensor(y))
        self.assertAlmostEqual(f1["macro_f1"], 11 / 15, places=6)
        self.assertAlmostEqual(f1["micro_f1"], 0.75, places=6)

    def test_h2snn_and_baselines_use_the_same_function(self) -> None:
        import src.trainers.graph_classification_trainer as gc_trainer
        from src.baselines import training

        self.assertIs(gc_trainer.compute_graph_classification_metrics,
                      compute_graph_classification_metrics)
        self.assertIn("compute_graph_classification_metrics",
                      inspect.getsource(training.train_graph_classification))
        logits, labels = _logits([0, 1, 1, 2], 3), torch.tensor([0, 1, 2, 2])
        record = compute_graph_classification_metrics(logits, labels)
        f1 = hetsheaf_f1(logits, labels)
        self.assertEqual((record["macro_f1"], record["micro_f1"]),
                         (f1["macro_f1"], f1["micro_f1"]))


if __name__ == "__main__":
    unittest.main()
