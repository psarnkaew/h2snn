"""J4.5 (task card Part J, spec v1.8 §6 "IMDB protocol"): unlabelled movies enter neither loss
nor metrics, and the prediction rule is "genre g iff logit > 0" (never oracle top-k), for the
H2S-NN trainer and for the shared baseline metric.

Read-only checks of existing code: no model or trainer code is changed by J4.5.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMDB_RAW = PROJECT_ROOT / "data" / "imdb" / "imdb" / "raw" / "IMDB"


def _logits_where_threshold_and_topk_differ():
    """Three labelled rows and one unlabelled row (all-zero labels). Row 0 has two true genres
    but one positive logit, so threshold and oracle top-k disagree; row 3 (unlabelled) has huge
    logits that would dominate the loss and the F1 if it were included."""
    labels = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.float)
    logits = torch.tensor([[2.0, -0.5, -3.0], [-1.0, 1.0, -2.0], [0.5, -0.2, 0.7],
                           [50.0, 50.0, 50.0]])
    mask = torch.tensor([True, True, True, False])
    return logits, labels, mask


class TrainerProtocolTests(unittest.TestCase):
    """The H2S-NN trainer (`src/trainers/trainer.py:242-273`): BCE and macro/micro-F1 on the
    masked rows only, threshold rule."""

    def _metrics(self):
        from src.trainers.trainer import Trainer

        logits, labels, mask = _logits_where_threshold_and_topk_differ()
        trainer = Trainer.__new__(Trainer)  # _compute_metrics reads no other state for multilabel
        output = SimpleNamespace(logits=logits)
        data = SimpleNamespace(target_y=labels, is_multilabel=True)
        loss, metrics = trainer._compute_metrics(output, data, mask)
        return loss, metrics, logits, labels, mask

    def test_loss_is_bce_over_labelled_rows_only(self) -> None:
        loss, _, logits, labels, mask = self._metrics()
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_f1_uses_logit_above_zero_on_labelled_rows_only(self) -> None:
        _, metrics, logits, labels, mask = self._metrics()
        preds = (logits[mask] > 0).long().numpy()
        truth = labels[mask].long().numpy()
        self.assertAlmostEqual(metrics["macro_f1"], f1_score(truth, preds, average="macro",
                                                             zero_division=0), places=12)
        self.assertAlmostEqual(metrics["micro_f1"], f1_score(truth, preds, average="micro",
                                                             zero_division=0), places=12)

    def test_default_is_not_oracle_topk(self) -> None:
        from src.utils.misc import compute_multilabel_classification_metrics

        logits, labels, mask = _logits_where_threshold_and_topk_differ()
        default = compute_multilabel_classification_metrics(logits[mask], labels[mask])
        topk = compute_multilabel_classification_metrics(logits[mask], labels[mask],
                                                         protocol="oracle_topk")
        self.assertNotAlmostEqual(default["macro_f1"], topk["macro_f1"], places=6)


class BaselineProtocolTests(unittest.TestCase):
    """The shared baseline metric (`src/baselines/metrics.py`): default threshold rule."""

    def test_classification_metrics_default_is_threshold(self) -> None:
        from src.baselines.metrics import DEFAULT_MULTILABEL_PROTOCOL, classification_metrics

        self.assertEqual(DEFAULT_MULTILABEL_PROTOCOL, "threshold")
        logits, labels, mask = _logits_where_threshold_and_topk_differ()
        got = classification_metrics(logits[mask], labels[mask], is_multilabel=True)
        preds = (logits[mask] > 0).long().numpy()
        self.assertAlmostEqual(got["macro_f1"], f1_score(labels[mask].long().numpy(), preds,
                                                         average="macro", zero_division=0),
                               places=12)


@unittest.skipUnless((IMDB_RAW / "label.dat").is_file(), "HGB IMDB not on disk")
class RealIMDBSplitTests(unittest.TestCase):
    """HGB IMDB through `build_dataset`, the pipeline both H2S-NN and every baseline adapter use
    (`src/baselines/bundles.py:376-385`): the 359 unlabelled movies are in no split."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.configs.config import TrainConfig
        from src.data.dataset import build_dataset

        cls.data = build_dataset(TrainConfig.from_dict({"device": "cpu", "data": {
            "dataset_name": "imdb", "root": str(PROJECT_ROOT / "data")}}))

    def test_unlabelled_movies_are_in_no_split(self) -> None:
        d = self.data
        self.assertTrue(d.is_multilabel)
        unlabelled = d.target_y.sum(dim=1) == 0
        self.assertEqual(int(unlabelled.sum()), 359)
        for mask in (d.target_train_mask, d.target_val_mask, d.target_test_mask):
            self.assertFalse(bool((mask & unlabelled).any()))
        covered = d.target_train_mask | d.target_val_mask | d.target_test_mask
        self.assertEqual(int(covered.sum()), 4573)
        self.assertTrue(bool((covered == ~unlabelled).all()))

    def test_split_is_hgb_24_6_70(self) -> None:
        d = self.data
        n_train, n_val = int(d.target_train_mask.sum()), int(d.target_val_mask.sum())
        n_test = int(d.target_test_mask.sum())
        self.assertEqual(n_train + n_val, 1371)  # label.dat
        self.assertEqual(n_test, 3202)  # label.dat.test
        self.assertLessEqual(abs(n_val / (n_train + n_val) - 0.2), 0.01)  # 24/6 of HGB's 30 %


if __name__ == "__main__":
    unittest.main()
