"""Part MB, MB4 (`docs/H2S-NN_partMB_card.md`): the mini-batch node-classification trainer.

On a synthetic ACM-shaped graph with the analysis flag (`data.training_mode: minibatch`, N4):
- steps per epoch = epoch_targets / B for B in {32, 256, 1024} (`epoch_targets` set explicitly);
- the training order visits every training target once per pass, is deterministic per seed and
  differs per pass; it continues across epochs;
- the training draw seeds differ per (epoch, batch) and the trainer uses them;
- the evaluated set is the whole split (count and ids); the metric equals the metric on the
  concatenated predictions; predictions do not depend on the evaluation batch size (float64);
- a stop condition raised in one batch stops the run with its named exception;
- the training mode is per dataset; the tuner refuses `batch_size` on ACM, the flag and the epoch
  field; an evaluation graph different from the training graph is used for its scopes (N6).
Card mutations: epoch = fixed number of steps regardless of B -> the steps test fails; evaluation on
a random subset -> the whole-split test fails.
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

import src.trainers.minibatch_trainer as mbt
from src.trainers.minibatch_trainer import TrainingOrder, training_draw_seed
from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.partO_fixtures import acm_like_hetero, patched_acm_loader


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _raw(tmp: str, batch_size: int, epoch_targets: int, **top) -> dict:
    return {
        "device": "cpu", "seed": 3, "epochs": 2, "exp_name": "mb", "output_dir": tmp,
        "early_stopping_patience": 100, "checkpoint_metric": "macro_f1", "checkpoint_mode": "max",
        **top,
        "data": {"dataset_name": "acm", "training_mode": "minibatch", "batch_size": batch_size,
                 "epoch_targets": epoch_targets, "wisp_max_length": 3,
                 "wisp_walks_per_anchor": 3},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
        "ablation": {"coupling": "symmetric"}}


@contextmanager
def trainer_for(batch_size: int, epoch_targets: int, hetero=None, **top):
    from src.builders import build_experiment_components
    from src.configs.config import TrainConfig

    with tempfile.TemporaryDirectory() as tmp, patched_acm_loader(
            hetero or acm_like_hetero(num_papers=30, num_authors=16, seed=13)):
        cfg = TrainConfig.from_dict(_raw(tmp, batch_size, epoch_targets, **top))
        torch.manual_seed(0)
        graph, model, _, _, logger, _, trainer = build_experiment_components(cfg)
        try:
            yield cfg, graph, model, trainer
        finally:
            logger.close()


class EpochTests(unittest.TestCase):
    def test_steps_per_epoch_follow_the_batch_size(self) -> None:
        for batch_size in (32, 256, 1024):
            with self.subTest(B=batch_size), trainer_for(batch_size, 1024) as (cfg, graph, _, tr):
                self.assertIsInstance(tr, mbt.MiniBatchNodeTrainer)
                self.assertEqual(graph.face_groups, ())  # no whole-graph draw in mini-batch
                order = TrainingOrder(graph.target_indices[graph.target_train_mask], cfg.seed)
                metrics = tr.train_epoch(graph, order, 0)
                self.assertEqual(tr.steps_taken, 1024 // batch_size)
                self.assertEqual(int(metrics["steps"]), 1024 // batch_size)

    def test_training_order_passes(self) -> None:
        targets = torch.arange(100, 118)
        order = TrainingOrder(targets, base_seed=4)
        stream = torch.cat([order.next(5) for _ in range(18)])  # 90 = 5 passes of 18
        passes = stream.view(5, 18)
        for p in passes:
            self.assertEqual(sorted(p.tolist()), targets.tolist())
        self.assertGreater(len({tuple(p.tolist()) for p in passes}), 1)
        again = TrainingOrder(targets, base_seed=4)
        self.assertTrue(torch.equal(torch.cat([again.next(5) for _ in range(18)]), stream))
        other = TrainingOrder(targets, base_seed=5)
        self.assertFalse(torch.equal(torch.cat([other.next(5) for _ in range(18)]), stream))

    def test_training_draw_seeds(self) -> None:
        seeds = {training_draw_seed(3, e, b) for e in range(50) for b in range(256)}
        self.assertEqual(len(seeds), 50 * 256)
        with trainer_for(32, 64) as (cfg, graph, _, tr):
            used = []
            real = tr._cells

            def spy(g, anchors, seed):
                used.append(seed)
                return real(g, anchors, seed)

            order = TrainingOrder(graph.target_indices[graph.target_train_mask], cfg.seed)
            with mock.patch.object(tr, "_cells", spy):
                tr.train_epoch(graph, order, 0)
                tr.train_epoch(graph, order, 1)
            self.assertEqual(used, [training_draw_seed(3, e, b) for e in (0, 1) for b in (0, 1)])


class EvaluationTests(unittest.TestCase):
    def test_whole_split_and_metric_on_the_concatenation(self) -> None:
        from src.utils.misc import compute_classification_metrics

        with trainer_for(32, 64) as (_, graph, _, tr):
            for mask in (graph.target_val_mask, graph.target_test_mask):
                ids, logits, labels = tr.evaluate(graph, mask, batch_size=4)
                expected = graph.target_indices[mask]
                self.assertEqual(ids.numel(), int(mask.sum()))
                self.assertTrue(torch.equal(ids, expected))
                self.assertTrue(torch.equal(labels, graph.target_y[mask]))
                metrics = tr.evaluate_split(graph, mask)
                whole = compute_classification_metrics(logits, labels)
                self.assertEqual(metrics["macro_f1"], whole["macro_f1"])

    def test_predictions_do_not_depend_on_the_evaluation_batch_size(self) -> None:
        with trainer_for(32, 64) as (_, graph, model, tr):
            model.double()
            model.solver_settings = model.solver_settings[:2] + (1e-10, 2000)
            graph64 = replace(graph, x=graph.x.double())
            _, a, _ = tr.evaluate(graph64, graph64.target_test_mask, batch_size=1024)
            _, b, _ = tr.evaluate(graph64, graph64.target_test_mask, batch_size=3)
            rel = float((a - b).abs().max() / b.abs().max())
            self.assertLessEqual(rel, 1e-8)

    def test_evaluation_graph_is_used_for_its_scopes(self) -> None:
        from src.data.dataset import build_dataset

        with trainer_for(32, 64) as (cfg, graph, _, tr):
            with patched_acm_loader(acm_like_hetero(num_papers=30, num_authors=16, seed=99)):
                eval_graph = build_dataset(cfg)
            seen = []
            real = mbt.build_scope_batch

            def spy(g, targets, cells):
                seen.append(g)
                return real(g, targets, cells)

            with mock.patch.object(mbt, "build_scope_batch", spy):
                tr.evaluate(eval_graph, eval_graph.target_val_mask)
            self.assertTrue(seen and all(g is eval_graph for g in seen))
            self.assertFalse(torch.equal(eval_graph.edge_index, graph.edge_index))


class StopAndConfigTests(unittest.TestCase):
    def test_stop_condition_in_one_batch_stops_the_run(self) -> None:
        from src.models.anchored_solve import LogWeightRangeStopError

        with trainer_for(32, 64) as (_, graph, model, tr):
            real = model.forward
            calls = {"n": 0}

            def forward(data, epoch=-1):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise LogWeightRangeStopError(31.0, 30.0, epoch)
                return real(data, epoch=epoch)

            with mock.patch.object(model, "forward", forward), \
                    self.assertRaises(LogWeightRangeStopError):
                tr.fit(graph)
            self.assertEqual(calls["n"], 2)

    def test_mode_per_dataset_and_tuner_refusals(self) -> None:
        from src.configs.config import (
            H2GB_NC_DATASETS,
            TrainConfig,
            default_training_mode,
            reject_never_tuned_settings,
        )

        for name in H2GB_NC_DATASETS:
            self.assertEqual(default_training_mode("node_classification", name), "minibatch")
        for name in ("acm", "dblp", "imdb", "freebase"):
            self.assertEqual(default_training_mode("node_classification", name), "full_batch")
        self.assertEqual(default_training_mode("link_prediction", "ogbn-mag_pp"), "minibatch")
        self.assertEqual(default_training_mode("link_prediction", "lastfm"), "full_batch")
        self.assertEqual(TrainConfig.from_dict({"data": {"dataset_name": "acm"}}).data.training_mode,
                         "full_batch")
        with self.assertRaises(ValueError):  # the flag exists on ACM and DBLP only
            TrainConfig.from_dict({"data": {"dataset_name": "imdb", "training_mode": "minibatch",
                                            "batch_size": 32}})
        with self.assertRaises(ValueError):  # B belongs to mini-batch
            TrainConfig.from_dict({"data": {"dataset_name": "acm", "batch_size": 32}})
        refused = [
            ({"data": {"dataset_name": "acm"}}, ["data.batch_size"]),
            ({"data": {"dataset_name": "acm", "training_mode": "minibatch", "batch_size": 32}}, []),
            ({"data": {"dataset_name": "acm"}}, ["data.epoch_targets"]),
            ({"data": {"dataset_name": "ogbn-mag", "epoch_targets": 4096}}, []),
            ({"data": {"dataset_name": "ogbn-mag"}}, ["data.training_mode"]),
        ]
        for raw, params in refused:
            with self.subTest(raw=raw, params=params), self.assertRaises(ValueError):
                reject_never_tuned_settings(raw, params)
        reject_never_tuned_settings({"data": {"dataset_name": "ogbn-mag"}}, ["data.batch_size"])


if __name__ == "__main__":
    unittest.main()
