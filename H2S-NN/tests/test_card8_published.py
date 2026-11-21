"""Card 8 C1: published configurations run exactly as the source file states.

* H2GB yaml cell (GCN on mag-year, amendment 1): the published file equals the yaml key by key, and the run arguments
  resolved from H2GB's cfg (`h2gb_published.resolve_published`) equal the yaml key by key (the yaml is parsed here,
  decoded as yacs decodes it); H2GB's scheduler is the yaml's (cosine with 5 warm-up epochs), the step rule is
  H2GB's (accumulation, clipping).
* HGB command cell (MAGNN on Freebase): MAGNN's run arguments (`_magnn_args.run_arguments`) equal `run_Freebase.py`'s
  constants and argparse defaults (parsed with `ast`), and the source's optimizer is Adam.
* A tuned parameter set with a source-schedule key is refused; the published runner skips filled cells and its default
  base seed is 0.

Card mutation: drop the scheduler from the H2GB mapping (`"optim.scheduler"` out of `RUN_ARGUMENTS`) ->
`test_h2gb_run_arguments_equal_the_yaml` fails. The real data root is unreachable.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch
import yaml

from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.test_baselines_minibatch import synthetic_hetero

PROJECT_ROOT = Path(__file__).resolve().parents[1]
H2GB_CELL = ("gcn", "mag-year")
MAGNN_SOURCE = PROJECT_ROOT / "vendor/HGB/NC/benchmark/methods/MAGNN/run_Freebase.py"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _yacs_decode(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _source_yaml_values(path: Path) -> dict:
    """The source yaml, parsed here (not by the module under test), dotted and decoded, minus the excluded keys."""
    from src.baselines.h2gb_published import EXCLUDED_KEYS

    out: dict = {}

    def walk(node: dict, prefix: str) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            else:
                out[f"{prefix}{key}"] = _yacs_decode(value)

    walk(yaml.safe_load(path.read_text(encoding="utf-8")), "")
    return {k: v for k, v in out.items()
            if not any(k == p or k.startswith(p + ".") for p in EXCLUDED_KEYS)}


def _resolve(model: str, dataset: str, values: dict):
    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.h2gb_published import resolve_published
    from src.baselines.runtime import pushd_sys_path

    ensure_pyg_submodules()
    data = synthetic_hetero()
    with pushd_sys_path("h2gb"):
        cfg, args = resolve_published(model, dataset, data, "a", torch.device("cpu"), values)
    return data, cfg, args


class H2GBPublishedTests(unittest.TestCase):
    def test_published_file_equals_the_yaml(self) -> None:
        from src.baselines.minibatch import h2gb_config_file
        from src.baselines.published import load_published_params

        model, dataset = H2GB_CELL
        source = _source_yaml_values(h2gb_config_file(model, dataset))
        params = load_published_params(model, "node_classification", dataset)
        self.assertIsNotNone(params)
        self.assertEqual(params.values, source)

    def test_h2gb_run_arguments_equal_the_yaml(self) -> None:
        from src.baselines.minibatch import h2gb_config_file
        from src.baselines.published import load_published_params

        model, dataset = H2GB_CELL
        source = _source_yaml_values(h2gb_config_file(model, dataset))
        values = load_published_params(model, "node_classification", dataset).values
        _, _, args = _resolve(model, dataset, values)
        for key, value in source.items():
            with self.subTest(key=key):
                self.assertIn(key, args, f"{key} is not a resolved run argument")
                self.assertEqual(args[key], value)
        # the schedule keys this cell must carry (the yaml's optim block)
        for key in ("optim.scheduler", "optim.num_warmup_epochs", "optim.clip_grad_norm",
                    "optim.batch_accumulation", "optim.max_epoch", "optim.base_lr", "optim.optimizer"):
            self.assertIn(key, source)

    def test_settings_scheduler_and_step_rule_are_h2gbs(self) -> None:
        from src.baselines.h2gb_published import (build_h2gb_network, h2gb_optimizer_and_schedule, published_plan,
                                                  run_settings)
        from src.baselines.published import load_published_params
        from src.baselines.runtime import pushd_sys_path

        model, dataset = H2GB_CELL
        values = load_published_params(model, "node_classification", dataset).values
        data, cfg, args = _resolve(model, dataset, values)
        settings = run_settings(args)
        self.assertEqual((settings.batch_size, settings.iter_per_epoch, settings.epochs, settings.eval_period,
                          settings.accumulation, settings.clip_value), (32, 256, 500, 4, 32, 1.0))
        plan = published_plan(dataset, settings)
        self.assertEqual((plan.batch_size, plan.steps_per_epoch, plan.epochs, plan.eval_period), (32, 256, 500, 4))
        with pushd_sys_path("h2gb"):
            torch.manual_seed(0)
            network = build_h2gb_network(data, "synthetic", 3, cfg)
            optimizer, schedule = h2gb_optimizer_and_schedule(network, cfg, args, settings)
        self.assertEqual(type(network).__name__, "HomoGNNModel")
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        # H2GB's adamW: no decay on biases (extra_optimizers.py:22-32)
        self.assertEqual(sorted(g["weight_decay"] for g in optimizer.param_groups), [0.0, 1e-5])
        lrs = []
        for _ in range(6):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            schedule.scheduler.step()
        # warm-up: lr_lambda(0) = max(1e-6, 0 / 5) (extra_optimizers.py:194-196), then 1/5, 2/5, ...
        self.assertAlmostEqual(lrs[0], 0.001 * 1e-6)
        self.assertAlmostEqual(lrs[1], 0.001 * 0.2)
        self.assertAlmostEqual(lrs[5], 0.001)  # 5 warm-up epochs -> base_lr
        self.assertEqual((schedule.accumulation, schedule.clip_value), (32, 1.0))

    def test_accumulation_rule_is_h2gbs(self) -> None:
        from src.baselines.minibatch import is_source_step

        steps32 = [it for it in range(1, 257) if is_source_step(it, 256, 32)]
        self.assertEqual(steps32, [31, 63, 95, 127, 159, 191, 223, 255])
        self.assertEqual(len([it for it in range(1, 257) if is_source_step(it, 256, 1)]), 256)
        self.assertEqual([it for it in range(1, 257) if is_source_step(it, 256, 8)][:2], [7, 15])

    def test_source_epoch_steps_clips_and_schedules(self) -> None:
        from src.baselines.minibatch import SourceSchedule, _source_epoch

        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        class Batch(dict):
            pass

        batches = []
        for _ in range(10):
            b = Batch()
            b["a"] = type("S", (), {"batch_size": 5, "y": torch.randint(0, 3, (5,)), "n_id": torch.arange(5)})()
            b["x"] = torch.randn(5, 4) * 100
            batches.append(b)
        calls = {"step": 0}
        original = optimizer.step

        def counting_step(*a, **k):
            calls["step"] += 1
            for p in model.parameters():
                self.assertLessEqual(float(p.grad.norm()), 0.5 + 1e-5 if p.grad is not None else 0.0)
            return original(*a, **k)

        optimizer.step = counting_step
        steps = _source_epoch(model, lambda b: model(b["x"]), batches, "a", None, optimizer,
                              torch.nn.functional.cross_entropy, SourceSchedule(scheduler, 0.5, 4), 0)
        self.assertEqual(steps, calls["step"])
        self.assertEqual(steps, len([it for it in range(1, 11) if (it + 1) % 4 == 0 or it + 1 == 10]))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)  # one scheduler step per epoch

    def test_published_homo_gnn_step_is_finite_and_moves(self) -> None:
        """The published H2GB path's network (H2GB's HomoGNNModel, GCN yaml) on one sampled batch."""
        from src.baselines.adapters._h2gb_network import seed_predictions
        from src.baselines.h2gb_published import build_h2gb_network, h2gb_optimizer_and_schedule, run_settings
        from src.baselines.minibatch import hgt_loader
        from src.baselines.published import load_published_params
        from src.baselines.runtime import pushd_sys_path

        model, dataset = H2GB_CELL
        values = dict(load_published_params(model, "node_classification", dataset).values)
        # small network for the unit test; H2GB's `Raw` encoder shares one Linear across types (raw_encoder.py:99-101,
        # equal widths on mag-year), the synthetic graph's two types have widths 5 and 3: `Hetero_Raw` here
        values.update({"gnn.dim_inner": 8, "gnn.layers_mp": 2, "dataset.node_encoder_name": "Hetero_Raw"})
        data = synthetic_hetero()
        loader = hgt_loader(data, {t: [10 ** 6] * 2 for t in data.node_types}, "a", "train", 8, shuffle=False, steps=1)
        batch = next(iter(loader))
        data2, cfg, args = _resolve(model, dataset, values)
        with pushd_sys_path("h2gb"):
            torch.manual_seed(0)
            network = build_h2gb_network(data2, "synthetic", 3, cfg)
            optimizer, _ = h2gb_optimizer_and_schedule(network, cfg, args, run_settings(args))
            for group in optimizer.param_groups:
                group["lr"] = 1e-2  # the cosine warm-up starts at 0
            before = [p.detach().clone() for p in network.parameters()]
            network.train()
            size = int(batch["a"].batch_size)
            loss = torch.nn.functional.cross_entropy(seed_predictions(network, batch.clone(), "a"), batch["a"].y[:size])
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            optimizer.step()
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, network.parameters())))


class MagnnFreebaseTests(unittest.TestCase):
    @staticmethod
    def _source() -> tuple[dict, str]:
        text = MAGNN_SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(text)
        values: dict = {}
        dest = {"feats-type": "feats_type", "hidden-dim": "hidden_dim", "num-heads": "num_heads",
                "attn-vec-dim": "attn_vec_dim", "rnn-type": "rnn_type", "epoch": "num_epochs", "patience": "patience",
                "batch-size": "batch_size", "samples": "neighbor_samples"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node in tree.body:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("dropout_rate", "lr", "weight_decay"):
                        values[target.id] = ast.literal_eval(node.value)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                flag = ast.literal_eval(node.args[0]).lstrip("-")
                for kw in node.keywords:
                    if kw.arg == "default" and flag in dest:
                        values[dest[flag]] = ast.literal_eval(kw.value)
        return values, text

    def test_run_arguments_equal_run_freebase(self) -> None:
        from src.baselines.adapters._magnn_args import run_arguments
        from src.baselines.published import load_published_params

        source, text = self._source()
        self.assertEqual(len(source), 12)
        params = load_published_params("magnn", "node_classification", "freebase")
        self.assertIsNotNone(params)
        args = run_arguments(params.values)
        for key, value in source.items():
            with self.subTest(key=key):
                self.assertEqual(args[key], value)
        self.assertIn("torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)", text)
        self.assertEqual(args["optimizer"], "adam")


class SourceScheduleRefusalTests(unittest.TestCase):
    def test_tuned_runs_refuse_source_schedule_keys(self) -> None:
        from src.baselines.published import refuse_source_schedule

        for key in ("optim.scheduler", "optim.batch_accumulation", "optim.clip_grad_norm", "scheduler",
                    "clip_grad_norm", "batch_accumulation", "lp_batch_size", "optimizer"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    refuse_source_schedule({key: 1}, "tuned")
                refuse_source_schedule({key: 1}, "published")
        refuse_source_schedule({"lr": 1e-3, "hidden_dim": 64}, "tuned")

    def test_fit_cell_refuses_before_building_anything(self) -> None:
        from src.baselines.campaign import fit_cell

        with self.assertRaises(ValueError) as ctx:
            fit_cell({"model": "gcn", "task": "node_classification", "dataset": "mag-year", "seed": 0},
                     {"optim.scheduler": "cosine_with_warmup"}, source="tuned", device=torch.device("cpu"))
        self.assertIn("source-schedule", str(ctx.exception))


class PublishedRunnerTests(unittest.TestCase):
    def test_filled_cells_skipped_and_seed_default_zero(self) -> None:
        import importlib.util
        import sys
        from unittest import mock

        spec = importlib.util.spec_from_file_location("run_baseline_published",
                                                      PROJECT_ROOT / "scripts/run_baseline_published.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with mock.patch.object(sys, "argv", ["x", "--task", "node_classification"]):
            self.assertEqual(runner.parse_args().base_seed, 0)
        cells = {(c.model, c.dataset) for c in runner.published_cells("node_classification", ["all"], ["all"])}
        self.assertNotIn(("sehgnn", "acm"), cells)  # filled (register), file exists
        self.assertNotIn(("pshgcn", "acm"), cells)
        self.assertIn(("pshgcn", "imdb"), cells)
        self.assertIn(("gcn", "mag-year"), cells)
        self.assertEqual([u.seed for u in runner.seed_units([0, 1, 2, 3, 4])], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()


class HGBLPPublishedTests(unittest.TestCase):
    """Card 8 amendments 3-4: the LP published files equal HGB's LP code key by key (parsed here)."""

    LP = PROJECT_ROOT / "vendor/HGB/LP/benchmark/methods"

    @staticmethod
    def _argparse_defaults(path: Path) -> dict:
        out = {}
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                flag = ast.literal_eval(node.args[0]).lstrip("-")
                for kw in node.keywords:
                    if kw.arg == "default":
                        try:
                            out[flag] = ast.literal_eval(kw.value)
                        except ValueError:
                            pass
        return out

    def _published(self, model: str, dataset: str) -> dict:
        from src.baselines.published import load_published_params

        params = load_published_params(model, "link_prediction", dataset)
        self.assertIsNotNone(params)
        return params.values

    def test_gcn_equals_homognn(self) -> None:
        source = self._argparse_defaults(self.LP / "GNN/utils.py")
        text = (self.LP / "GNN/homoGNN.py").read_text(encoding="utf-8")
        self.assertIn("model = GCN(in_feats=dim_list, hid_feats=64,", text)
        self.assertIn("th.optim.Adam(", text)
        for dataset in ("pubmed", "lastfm"):
            values = self._published("gcn", dataset)
            expected = {"hidden_dim": 64, "num_layers": source["n_layers"], "dropout": source["dropout"],
                        "lr": source["lr"], "weight_decay": source["weight_decay"], "epochs": source["epochs"],
                        "early_stopping": int(source["patience"]),
                        "lp_batch_size": source["batch_size"] // 2,  # 1,024 alternating pos/neg rows
                        "decoder_type": {"dismult": "distmult", "dot": "dot"}[source["decoder"]],
                        "feats_type": 3, "optimizer": "adam"}
            self.assertEqual(values, expected)

    def test_rgcn_pubmed_equals_link_predict(self) -> None:
        source = self._argparse_defaults(self.LP / "RGCN/link_predict.py")
        readme = (self.LP / "RGCN/README.md").read_text(encoding="utf-8")
        self.assertIn("python link_predict.py --dataset=PubMed  --hidden-dim=60", readme)
        expected = {"feats_type": source["feats-type"], "hidden_dim": 60, "num_layers": source["num-layers"],
                    "dropout": source["dropout"], "lr": source["lr"], "weight_decay": source["weight-decay"],
                    "epochs": source["epoch"], "early_stopping": source["patience"],
                    "lp_batch_size": source["batch-size"], "n_bases": source["n-bases"],
                    "regularization": source["regularization"], "optimizer": "adam", "decoder_type": "distmult"}
        self.assertEqual(self._published("rgcn", "pubmed"), expected)

    def test_magnn_lastfm_equals_run_lastfm(self) -> None:
        path = self.LP / "MAGNN/run_LastFM.py"
        source = self._argparse_defaults(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and t.id in ("dropout_rate", "lr", "weight_decay")}
        self.assertEqual(source["feats-type"], 0)  # MAGNN's 0 = all id vectors = our feats_type 3
        expected = {"dropout": consts["dropout_rate"], "lr": consts["lr"], "weight_decay": consts["weight_decay"],
                    "optimizer": "adam", "feats_type": 3, "hidden_dim": source["hidden-dim"],
                    "num_heads": source["num-heads"], "attn_vec_dim": source["attn-vec-dim"],
                    "rnn_type": source["rnn-type"], "epochs": source["epoch"], "early_stopping": source["patience"],
                    "lp_batch_size": source["batch-size"], "samples": source["samples"], "decoder_type": "dot"}
        self.assertEqual(self._published("magnn", "lastfm"), expected)
