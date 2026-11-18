"""Card 1f U3 (B1a): the fixed validation subset in the shared H2GB NC loops.

Synthetic graphs only (the real data root is unreachable). Three tests (author cut 2026-11-18):
1. determinism, and the same nodes for every method: two calls are equal; the size follows the table; the subset is
   a subset of the validation ids; it does not depend on the training seed; the class A/C loop and the class B loop
   evaluate exactly the same subset ids during training;
2. no test pass during training: test targets are touched only after the last epoch, once each;
3. best weights restored: a toy model whose weights are scripted per epoch (best subset epoch != last epoch), and
   the record's test metric equals the metric of the best epoch's weights, recomputed independently here.

Card mutation 1: the loops keep the last epoch's weights instead of restoring the best (drop
`model.load_state_dict(best_state)`) -> test 3 fails.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.test_baselines_minibatch import full_fanout, synthetic_hetero

DATASET = "ogbn-mag"
SMALL = {DATASET: 5}  # synthetic graphs have 13 validation targets; the table's 16,384 would take them all


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def toy_data():
    """`synthetic_hetero` with the target features set to the one-hot label, so a 3x3 weight decides every
    prediction: identity = all correct, a cyclic shift = all wrong."""
    data = synthetic_hetero()
    data["a"].x = torch.nn.functional.one_hot(data["a"].y, 3).double()
    return data


GOOD = torch.eye(3, dtype=torch.float64)
BAD = torch.roll(torch.eye(3, dtype=torch.float64), 1, dims=0)


class ScriptedOptimizer:
    """Sets the weight to `schedule[epoch]` at every step (the loop's backward still runs)."""

    def __init__(self, weight: torch.nn.Parameter, schedule: list[torch.Tensor], steps_per_epoch: int) -> None:
        self.weight, self.schedule, self.steps_per_epoch, self.steps = weight, schedule, steps_per_epoch, 0

    def zero_grad(self) -> None:
        self.weight.grad = None

    def step(self) -> None:
        with torch.no_grad():
            self.weight.copy_(self.schedule[self.steps // self.steps_per_epoch])
        self.steps += 1


def run_loop(kind: str, data, *, schedule=None, calls=None, snapshots=None):
    """One run of the class A/C loop (`kind` "ac") or the class B loop ("b") on `data` with a 3x3 linear model.
    `calls` collects (model.training, seed ids) per forward; `snapshots` the model's weight at each evaluation."""
    from src.baselines.minibatch import (MinibatchPlan, eval_loader, hgt_loader, train_index_minibatch,
                                         train_minibatch_node_classification, val_subset_loader)

    plan = MinibatchPlan(DATASET, 4, 12, 3, 5, 2)  # eval epochs 0, 1, 3, 4
    model = torch.nn.Linear(3, 3, bias=False).double()
    schedule = schedule or [GOOD, BAD, GOOD, BAD, GOOD]
    optimizer = ScriptedOptimizer(model.weight, schedule, plan.steps_per_epoch)
    x, labels = data["a"].x.double(), data["a"].y
    calls = [] if calls is None else calls

    def report(*, epoch, monitor, running_best, metrics):
        if snapshots is not None:
            snapshots[epoch] = model.weight.detach().clone()
    if kind == "ac":
        fanout = full_fanout(data, 1)

        def forward(batch):
            size = int(batch["a"].batch_size)
            calls.append((model.training, batch["a"].n_id[:size].clone()))
            return model(batch["a"].x[:size].double())

        subset_loader, subset_ids = val_subset_loader(data, fanout, "a", DATASET, 4)
        return train_minibatch_node_classification(
            model=model, forward_seeds=forward,
            train_loader=hgt_loader(data, fanout, "a", "train", 4, shuffle=True, steps=plan.steps_per_epoch),
            val_loader=eval_loader(data, fanout, "a", "val", 4), test_loader=eval_loader(data, fanout, "a", "test", 4),
            val_ids=data["a"].val_mask.nonzero().view(-1), test_ids=data["a"].test_mask.nonzero().view(-1),
            val_subset_loader=subset_loader, val_subset_ids=subset_ids, target_type="a", optimizer=optimizer,
            criterion=torch.nn.functional.cross_entropy, plan=plan, report=report)

    def forward_rows(index):
        calls.append((model.training, index.clone()))
        return model(x[index])

    return train_index_minibatch(
        model=model, forward_rows=forward_rows, labels=labels, train_idx=data["a"].train_mask.nonzero().view(-1),
        val_idx=data["a"].val_mask.nonzero().view(-1), test_idx=data["a"].test_mask.nonzero().view(-1),
        optimizer=optimizer, criterion=torch.nn.functional.cross_entropy, plan=plan, base_seed=7, report=report)


class B1aTests(unittest.TestCase):
    def test_subset_is_deterministic_and_the_same_for_every_method(self) -> None:
        from src.evaluation.validation_subset import SUBSET_SIZES, subset_fingerprint, validation_subset

        val_ids = torch.randperm(50_000, generator=torch.Generator().manual_seed(3))[:20_000] * 2
        for dataset, size in SUBSET_SIZES.items():
            if dataset == "pdns":
                with self.assertRaises(ValueError):  # card 1f stop item: no single validation id set
                    validation_subset(dataset, val_ids)
                continue
            with self.subTest(dataset=dataset):
                torch.manual_seed(1)
                first = validation_subset(dataset, val_ids)
                torch.manual_seed(2)  # a different training seed changes nothing
                second = validation_subset(dataset, val_ids.flip(0))
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(first.numel(), min(size, val_ids.numel()))
                self.assertTrue(bool(torch.isin(first, val_ids).all()))
                self.assertTrue(bool((first[1:] > first[:-1]).all()))
                self.assertEqual(subset_fingerprint(first), subset_fingerprint(second))
        with self.assertRaises(ValueError):  # outside the table: no silent whole-split fallback
            validation_subset("acm", val_ids)

        data = toy_data()
        with mock.patch.dict("src.evaluation.validation_subset.SUBSET_SIZES", SMALL):
            expected = validation_subset(DATASET, data["a"].val_mask.nonzero().view(-1))
            seen = {}
            for kind in ("ac", "b"):
                calls: list = []
                record = run_loop(kind, data, calls=calls)
                eval_calls = [ids for training, ids in calls if not training]
                # 4 eval epochs x 2 subset batches, then the whole validation (4) and test (3) batches
                seen[kind] = torch.unique(torch.cat(eval_calls[:-7]))
                self.assertEqual(record["val_subset_size"], float(expected.numel()))
                self.assertEqual(record["val_subset_fingerprint"], subset_fingerprint(expected))
            self.assertEqual(expected.numel(), 5)
            self.assertTrue(torch.equal(seen["ac"], expected))
            self.assertTrue(torch.equal(seen["b"], expected))

    def test_no_test_pass_during_training(self) -> None:
        data = toy_data()
        test_ids = data["a"].test_mask.nonzero().view(-1)
        with mock.patch.dict("src.evaluation.validation_subset.SUBSET_SIZES", SMALL):
            for kind in ("ac", "b"):
                with self.subTest(loop=kind):
                    calls: list = []
                    run_loop(kind, data, calls=calls)
                    touching = [i for i, (_, ids) in enumerate(calls) if bool(torch.isin(ids, test_ids).any())]
                    # only the last 3 forwards (the final test pass: 9 targets in batches of 4) touch test targets
                    self.assertEqual(touching, list(range(len(calls) - 3, len(calls))))
                    final = torch.cat([calls[i][1] for i in touching])
                    self.assertEqual(sorted(final.tolist()), sorted(test_ids.tolist()))

    def test_best_weights_are_restored(self) -> None:
        from src.baselines.metrics import classification_metrics

        data = toy_data()
        x, labels = data["a"].x.double(), data["a"].y
        test_ids = data["a"].test_mask.nonzero().view(-1)
        # Subset macro-F1 is 1 under GOOD and 0 under BAD: best eval epoch 1 (ties: earliest), last epoch 4 is BAD.
        schedule = [BAD, GOOD, BAD, BAD, BAD]
        with mock.patch.dict("src.evaluation.validation_subset.SUBSET_SIZES", SMALL):
            for kind in ("ac", "b"):
                with self.subTest(loop=kind):
                    snapshots: dict[int, torch.Tensor] = {}
                    record = run_loop(kind, data, schedule=schedule, snapshots=snapshots)
                    self.assertEqual(sorted(snapshots), [0, 1, 3, 4])
                    self.assertTrue(torch.equal(snapshots[4], BAD))  # the last epoch's weights are not the best
                    self.assertEqual(record["best_epoch"], 1.0)
                    best = snapshots[int(record["best_epoch"])]
                    expected = classification_metrics(x[test_ids] @ best.T, labels[test_ids], is_multilabel=False)
                    self.assertAlmostEqual(record["test_macro_f1"], expected["macro_f1"], places=12)
                    self.assertAlmostEqual(record["test_macro_f1"], 1.0, places=12)
                    self.assertAlmostEqual(record["val_subset_macro_f1"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
