"""Part MB-fix, MF1 (`docs/H2S-NN_partMBfix_card.md`; decision D-MB2): a training batch never crosses
a pass boundary (H2GB's loader).

(i) 18 targets, B = 4: batch sizes over two passes are 4,4,4,4,2,4,4,4,4,2; each pass is a
    permutation; no batch holds a target twice;
(ii) steps per epoch stay `epoch_targets / B`; the short batch counts as one step;
(iii) the loss of a short batch is the mean over its own targets (spy on the loss input size);
(iv) the order is deterministic per seed and differs per pass (`test_partMB_trainer`, kept).
Card mutation: the old crossing behaviour (fill up to B from the next pass) -> (i) fails.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from src.trainers.minibatch_trainer import TrainingOrder
from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.test_partMB_trainer import trainer_for


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class PassBoundaryTests(unittest.TestCase):
    def test_batches_stop_at_the_pass_boundary(self) -> None:
        targets = torch.arange(200, 218)
        order = TrainingOrder(targets, base_seed=6)
        batches = [order.next(4) for _ in range(10)]
        self.assertEqual([b.numel() for b in batches], [4, 4, 4, 4, 2, 4, 4, 4, 4, 2])
        for first in (0, 5):
            one_pass = torch.cat(batches[first:first + 5])
            self.assertEqual(sorted(one_pass.tolist()), targets.tolist())
        for b in batches:
            self.assertEqual(len(set(b.tolist())), b.numel())

    def test_steps_and_the_short_batch_loss(self) -> None:
        with trainer_for(8, 16) as (cfg, graph, _, trainer):
            train = graph.target_indices[graph.target_train_mask]
            self.assertNotEqual(train.numel() % 8, 0, "the fixture needs a short last batch")
            expected_order = TrainingOrder(train, cfg.seed)
            expected = [expected_order.next(8).numel() for _ in range(3 * 2)]
            self.assertIn(train.numel() % 8, expected)  # a short batch occurs in 3 epochs
            seen = []
            real = trainer._loss

            def spy(logits, labels, multilabel):
                seen.append(int(logits.size(0)))
                return real(logits, labels, multilabel)

            order = TrainingOrder(train, cfg.seed)
            with mock.patch.object(trainer, "_loss", spy):
                for epoch in range(3):
                    metrics = trainer.train_epoch(graph, order, epoch)
                    self.assertEqual(int(metrics["steps"]), 16 // 8)
            self.assertEqual(trainer.steps_taken, 3 * 2)
            self.assertEqual(seen, expected)  # each step's loss over its own targets only


if __name__ == "__main__":
    unittest.main()
