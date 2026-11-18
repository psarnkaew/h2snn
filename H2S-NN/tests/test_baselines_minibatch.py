"""Card baselines-h2gb C1: the shared mini-batch infrastructure of the baselines (`src/baselines/minibatch.py`).

Synthetic graphs and H2GB's YAML files only; the real data root is unreachable.
* steps per epoch = 8,192 / B for every B of the shared range; other B refused;
* the fan-out equals H2GB's YAML for every (class-A method, H2GB dataset) pair (YAML parsed here independently);
* evaluation covers every target of a split exactly once, the last partial batch included;
* the training loop runs the plan's steps, evaluates by H2GB's eval-epoch rule and keeps the best validation epoch.

Card mutation: evaluation drops the last partial batch (`eval_steps` = floor) -> `test_evaluation_covers_split_once`
fails.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch
import yaml

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

REPO = Path(__file__).resolve().parents[1]


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def synthetic_hetero(num_a: int = 40, num_b: int = 25, seed: int = 0, val: int = 13, test: int = 9):
    """Target type `a` (labels 0..2, train/val/test masks), type `b`; a-b in both directions, a-a symmetric.
    Features are float32 values stored as float64 (H2GB's `convert_batch` casts batches to float32)."""
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["a"].x = torch.randn(num_a, 5, generator=g).float().double()
    data["b"].x = torch.randn(num_b, 3, generator=g).float().double()
    data["a"].y = torch.randint(0, 3, (num_a,), generator=g)
    order = torch.randperm(num_a, generator=g)
    for name, part in (("val", order[:val]), ("test", order[val:val + test]), ("train", order[val + test:])):
        mask = torch.zeros(num_a, dtype=torch.bool)
        mask[part] = True
        data["a"][f"{name}_mask"] = mask
    ab = torch.unique(torch.stack([torch.randint(0, num_a, (70,), generator=g),
                                   torch.randint(0, num_b, (70,), generator=g)]), dim=1)
    aa = torch.stack([torch.randint(0, num_a, (50,), generator=g), torch.randint(0, num_a, (50,), generator=g)])
    aa = aa[:, aa[0] != aa[1]]
    data["a", "to", "b"].edge_index = ab
    data["b", "rev_to", "a"].edge_index = ab.flip(0)
    data["a", "link", "a"].edge_index = torch.unique(torch.cat([aa, aa.flip(0)], dim=1), dim=1)
    return data


def full_fanout(data, depth: int) -> dict[str, list[int]]:
    return {t: [10 ** 6] * depth for t in data.node_types}


class BatchPlanTests(unittest.TestCase):
    def test_steps_per_epoch_for_each_batch_size(self) -> None:
        from src.baselines.minibatch import minibatch_plan

        for dataset in ("ogbn-mag", "oag-cs", "h-pokec", "pdns"):
            for batch_size, steps in ((32, 256), (256, 32), (1024, 8)):
                plan = minibatch_plan(dataset, batch_size)
                self.assertEqual(plan.steps_per_epoch, steps)
                self.assertEqual(plan.steps_per_epoch * batch_size, 8192)
                self.assertEqual(plan.epochs, 500)
            self.assertEqual(minibatch_plan(dataset, 32).eval_period, 2 if dataset.startswith("oag") else 4)
        with self.assertRaises(ValueError):
            minibatch_plan("ogbn-mag", 64)
        with self.assertRaises(ValueError):
            minibatch_plan("acm", 32)

    def test_eval_epoch_rule_is_h2gbs(self) -> None:
        from src.baselines.minibatch import MinibatchPlan, is_eval_epoch

        plan = MinibatchPlan("ogbn-mag", 32, 8192, 256, 10, 4)
        self.assertEqual([e for e in range(10) if is_eval_epoch(e, plan)], [0, 3, 7, 9])


class FanoutTests(unittest.TestCase):
    def _yaml_sizes(self, path: Path, node_types: tuple[str, ...]) -> dict[str, list[int]]:
        train = yaml.safe_load(path.read_text())["train"]
        if "neighbor_sizes_dict" in train:
            text = train["neighbor_sizes_dict"].strip().strip('"').replace("\\", "")
            return {k: list(v) for k, v in ast.literal_eval(text).items()}
        return {t: list(train["neighbor_sizes"]) for t in node_types}

    def test_fanouts_equal_h2gb_yaml(self) -> None:
        from src.baselines.minibatch import H2GB_CONFIG_FOLDERS, H2GB_CONFIG_METHODS, h2gb_config_file, \
            h2gb_sample_sizes

        node_types = ("t0", "t1")
        checked = 0
        for method in H2GB_CONFIG_METHODS:
            for dataset in H2GB_CONFIG_FOLDERS:
                with self.subTest(method=method, dataset=dataset):
                    path = h2gb_config_file(method, dataset)
                    got = h2gb_sample_sizes(method, dataset, node_types)
                    self.assertEqual(got, self._yaml_sizes(path, node_types))
                    checked += 1
        self.assertEqual(checked, 32)

    def test_repeat_fanout_refuses_non_uniform_lists(self) -> None:
        from src.baselines.minibatch import repeat_fanout

        self.assertEqual(repeat_fanout({"a": [512] * 6}, 3), {"a": [512, 512, 512]})
        with self.assertRaises(ValueError):
            repeat_fanout({"user": [2048, 2048, 512, 512]}, 3)


class EvaluationCoverageTests(unittest.TestCase):
    def test_evaluation_covers_split_once(self) -> None:
        from src.baselines.minibatch import eval_loader, evaluate_split

        data = synthetic_hetero(val=13, test=9)
        for split in ("val", "test"):
            expected = data["a"][f"{split}_mask"].nonzero().view(-1)
            for batch_size in (4, 5, 32):  # 13 and 9 are not multiples of 4 or 5
                loader = eval_loader(data, full_fanout(data, 1), "a", split, batch_size)
                for _ in range(2):  # a second pass covers the same targets in the same order
                    ids, logits, _ = evaluate_split(loader, lambda b: torch.zeros(int(b["a"].batch_size), 3),
                                                    "a", expected)
                    self.assertEqual(sorted(ids.tolist()), sorted(expected.tolist()))
                    self.assertEqual(ids.numel(), torch.unique(ids).numel())
                    self.assertEqual(logits.size(0), expected.numel())

    def test_incomplete_pass_is_refused(self) -> None:
        from src.baselines.minibatch import evaluate_split, hgt_loader

        data = synthetic_hetero(val=13)
        expected = data["a"]["val_mask"].nonzero().view(-1)
        loader = hgt_loader(data, full_fanout(data, 1), "a", "val", 4, shuffle=False, steps=3)  # 12 of 13
        with self.assertRaises(RuntimeError):
            evaluate_split(loader, lambda b: torch.zeros(int(b["a"].batch_size), 3), "a", expected)


class LoopTests(unittest.TestCase):
    def test_loop_runs_plan_and_keeps_best_validation_epoch(self) -> None:
        from src.baselines.minibatch import MinibatchPlan, eval_loader, hgt_loader, \
            train_minibatch_node_classification, val_subset_loader

        torch.manual_seed(0)
        data = synthetic_hetero()
        fanout = full_fanout(data, 1)
        plan = MinibatchPlan("ogbn-mag", 4, 12, 3, 5, 2)
        model = torch.nn.Linear(5, 3).double()
        steps = []

        def forward(batch):
            steps.append(int(batch["a"].batch_size))
            return model(batch["a"].x[: int(batch["a"].batch_size)].double())

        subset_loader, subset_ids = val_subset_loader(data, fanout, "a", "ogbn-mag", 4)  # B1a: all 13 (< 16,384)
        record = train_minibatch_node_classification(
            model=model, forward_seeds=forward, val_subset_loader=subset_loader, val_subset_ids=subset_ids,
            train_loader=hgt_loader(data, fanout, "a", "train", 4, shuffle=True, steps=plan.steps_per_epoch),
            val_loader=eval_loader(data, fanout, "a", "val", 4), test_loader=eval_loader(data, fanout, "a", "test", 4),
            val_ids=data["a"].val_mask.nonzero().view(-1), test_ids=data["a"].test_mask.nonzero().view(-1),
            target_type="a", optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
            criterion=torch.nn.functional.cross_entropy, plan=plan)
        # 5 epochs x 3 training steps of 4 targets; subset evaluations at epochs 0, 1, 3, 4 (4 batches each); card 1f
        # B1a: then one whole-split pass (val 4 batches, test 3)
        self.assertEqual(len(steps), 5 * 3 + 4 * 4 + (4 + 3))
        self.assertIn(record["best_epoch"], (0.0, 1.0, 3.0, 4.0))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in record.values() if isinstance(v, float)))


if __name__ == "__main__":
    unittest.main()
