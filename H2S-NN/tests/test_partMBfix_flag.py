"""Part MB-fix, MF2 (`docs/H2S-NN_partMBfix_card.md`; decision "Part MB review (analysis flag)"):
the ACM/DBLP mini-batch analysis flag with `batch_size`/`epoch_targets` unset trains on whole passes.

- config: the flag resolves to B = 0, epoch = 0 (whole pass), evaluation every epoch; `epochs`,
  `early_stopping_patience`, `checkpoint_metric`/`checkpoint_mode` equal the full-batch run's;
- trainer (synthetic ACM-shaped graph): one step per epoch, the batch is every training target
  (count and ids); evaluation every epoch;
- explicit B still works (`test_partMB_trainer` unchanged); the tuner still refuses the flag; an
  H2GB dataset without B still raises.
Card mutation: the flag falls back to 8,192 epoch targets -> the one-step test fails.
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

import src.trainers.minibatch_trainer as mbt
from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.partO_fixtures import acm_like_hetero, patched_acm_loader


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _raw(tmp: str, **data) -> dict:
    return {
        "device": "cpu", "seed": 3, "exp_name": "flag", "output_dir": tmp,
        "data": {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 3, **data},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
        "ablation": {"coupling": "symmetric"}}


@contextmanager
def flag_trainer(epochs: int):
    from src.builders import build_experiment_components
    from src.configs.config import TrainConfig

    with tempfile.TemporaryDirectory() as tmp, patched_acm_loader(
            acm_like_hetero(num_papers=30, num_authors=16, seed=13)):
        raw = _raw(tmp, training_mode="minibatch")
        cfg = TrainConfig.from_dict(raw)
        cfg.epochs = epochs
        torch.manual_seed(0)
        graph, _, _, _, logger, _, trainer = build_experiment_components(cfg)
        try:
            yield cfg, graph, trainer
        finally:
            logger.close()


class FlagConfigTests(unittest.TestCase):
    def test_flag_resolves_to_whole_passes_with_the_full_batch_budget(self) -> None:
        from src.configs.config import TrainConfig

        with tempfile.TemporaryDirectory() as tmp:
            full = TrainConfig.from_dict(_raw(tmp))
            flag = TrainConfig.from_dict(_raw(tmp, training_mode="minibatch"))
        self.assertEqual((flag.data.training_mode, flag.data.batch_size, flag.data.epoch_targets,
                          flag.data.eval_period), ("minibatch", 0, 0, 1))
        for field in ("epochs", "early_stopping_patience", "checkpoint_metric", "checkpoint_mode"):
            self.assertEqual(getattr(flag, field), getattr(full, field), field)
        self.assertEqual((full.epochs, full.early_stopping_patience), (300, 30))  # HGB budget

    def test_tuner_refusal_and_h2gb_without_b(self) -> None:
        from src.configs.config import TrainConfig, reject_never_tuned_settings

        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"data": {"dataset_name": "acm",
                                                  "training_mode": "minibatch"}})
        with self.assertRaisesRegex(ValueError, "batch_size"):
            TrainConfig.from_dict({"data": {"dataset_name": "ogbn-mag",
                                            "task_type": "node_classification",
                                            "wisp_walks_per_anchor": 8}})
        with self.assertRaises(ValueError):  # an epoch without B on the flag is ambiguous
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "training_mode": "minibatch",
                                            "epoch_targets": 64}})


class FlagTrainerTests(unittest.TestCase):
    def test_one_step_per_epoch_over_every_training_target(self) -> None:
        with flag_trainer(epochs=2) as (cfg, graph, trainer):
            train = graph.target_indices[graph.target_train_mask]
            self.assertEqual(trainer.batch_plan(graph), (train.numel(), train.numel()))
            seen = []
            real = mbt.build_scope_batch

            def spy(g, targets, cells):
                seen.append(targets.clone())
                return real(g, targets, cells)

            order = mbt.TrainingOrder(train, cfg.seed)
            with mock.patch.object(mbt, "build_scope_batch", spy):
                for epoch in range(2):
                    metrics = trainer.train_epoch(graph, order, epoch)
                    self.assertEqual(int(metrics["steps"]), 1)
            self.assertEqual(trainer.steps_taken, 2)
            self.assertEqual(len(seen), 2)
            for targets in seen:
                self.assertEqual(targets.numel(), train.numel())
                self.assertEqual(sorted(targets.tolist()), sorted(train.tolist()))

    def test_evaluation_every_epoch(self) -> None:
        with flag_trainer(epochs=3) as (_, graph, trainer):
            calls = []
            real = trainer.evaluate_split

            def spy(g, mask):
                calls.append(mask)
                return real(g, mask)

            with mock.patch.object(trainer, "evaluate_split", spy):
                trainer.fit(graph)
            self.assertEqual(len(calls), 3 * 2)  # validation and test, every epoch


if __name__ == "__main__":
    unittest.main()
