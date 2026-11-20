"""Card 8 C7 (amendment 6): W&B logging for every run, with a fake `wandb` module (nothing reaches W&B).

* Per epoch, exactly the amendment's keys: HGB NC loop (`train/loss`, `val/loss`, `val/<metrics>`, `test/<metrics>`,
  `best/macro_f1`, `epoch`), LP loop (`test/auroc`, `test/mrr`, `test/aupr` of the random set), H2GB NC loop
  (`train/loss`, `val_subset/<metrics>`; no `test/*`, no `val/*`).
* An H2S-NN final-seed run and a published run each open a W&B run (group `<token>_final` / `<token>_published`, tag,
  name `<token>_seed<s>`), log per epoch, and write the summary (selected epoch, `val_*`, `test_*`, wall time, GPU,
  git HEAD) and the resolved configuration.
* Selection is unchanged: the HGB NC loop gives the same record with and without logging.

Card mutation: drop `test/*` from the HGB NC payload -> `test_hgb_nc_epoch_keys` fails.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class FakeRun:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.summary: dict = {}
        self.config = types.SimpleNamespace(updates=[], update=lambda value, allow_val_change=False:
                                            self.config.updates.append(value))
        self.finished = False
        self.id, self.name = "fake-id", kwargs.get("name")

    def finish(self) -> None:
        self.finished = True


def fake_wandb() -> types.ModuleType:
    module = types.ModuleType("wandb")
    module.run = None
    module.logged = []
    module.runs = []

    def init(**kwargs):
        run = FakeRun(**kwargs)
        module.run = run
        module.runs.append(run)
        return run

    def log(payload, step=None):
        module.logged.append((step, dict(payload)))

    module.init, module.log = init, log
    return module


@contextlib.contextmanager
def only_wandb(module: types.ModuleType):
    """`sys.modules["wandb"]` = the fake for the block; only that key is restored (`mock.patch.dict` would restore the
    whole dict and drop modules first imported inside the block, e.g. DGL, whose second import aborts)."""
    missing = object()
    old = sys.modules.get("wandb", missing)
    sys.modules["wandb"] = module
    try:
        yield module
    finally:
        if old is missing:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = old


def _nc_loop(report):
    from src.baselines.training import train_full_batch_node_classification

    torch.manual_seed(0)
    x = torch.randn(30, 4)
    labels = torch.randint(0, 3, (30,))
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    return train_full_batch_node_classification(
        model=model, forward=lambda: model(x), labels=labels, train_idx=torch.arange(0, 15),
        val_idx=torch.arange(15, 22), test_idx=torch.arange(22, 30), optimizer=optimizer,
        criterion=torch.nn.functional.cross_entropy, is_multilabel=False, epochs=4, patience=4, report=report)


class EpochKeyTests(unittest.TestCase):
    def test_hgb_nc_epoch_keys(self) -> None:
        from src.baselines.metrics import classification_metrics
        from src.baselines.progress import WandbEpochReport

        wandb = fake_wandb()
        with only_wandb(wandb):
            wandb.init(project="p")
            _nc_loop(WandbEpochReport("macro_f1"))
        metric_keys = set(classification_metrics(torch.randn(5, 3), torch.tensor([0, 1, 2, 0, 1]),
                                                 is_multilabel=False))
        expected = ({"best/macro_f1", "epoch", "train/loss", "val/loss"} | {f"val/{k}" for k in metric_keys}
                    | {f"test/{k}" for k in metric_keys})
        self.assertEqual(len(wandb.logged), 4)
        for step, payload in wandb.logged:
            self.assertEqual(set(payload), expected, step)

    def test_selection_is_unchanged_by_logging(self) -> None:
        from src.baselines.progress import WandbEpochReport

        plain = _nc_loop(None)
        wandb = fake_wandb()
        with only_wandb(wandb):
            wandb.init(project="p")
            logged = _nc_loop(WandbEpochReport("macro_f1"))
        self.assertEqual(plain, logged)

    def test_lp_epoch_keys(self) -> None:
        from src.baselines.heads import SharedLinkHead
        from src.baselines.progress import WandbEpochReport
        from src.baselines.training import train_link_prediction
        from tests.test_card8_lp_step import _synthetic_link_data

        data = _synthetic_link_data(200)
        torch.manual_seed(0)
        embedding = torch.nn.Parameter(torch.randn(300, 8))
        model = torch.nn.Module()
        model.register_parameter("emb", embedding)
        head = SharedLinkHead("dot", num_relations=1, dim=8)
        wandb = fake_wandb()
        fake_test = {"random_auroc": 0.5, "random_mrr": 0.3, "random_aupr": 0.4, "random_negatives": 10,
                     "2hop_auroc": 0.9}
        with only_wandb(wandb), \
                mock.patch("src.utils.misc.lp_test_set_metrics", lambda scorer, d: dict(fake_test)):
            wandb.init(project="p")
            train_link_prediction(model=model, head=head, embed=lambda: embedding, data=data,
                                  optimizer=torch.optim.SGD([embedding], lr=0.01), device=torch.device("cpu"),
                                  epochs=2, patience=2, report=WandbEpochReport("auroc"))
        self.assertEqual(len(wandb.logged), 2)
        for _, payload in wandb.logged:
            test_keys = {k for k in payload if k.startswith("test/")}
            self.assertEqual(test_keys, {"test/auroc", "test/mrr", "test/aupr"})
            self.assertEqual((payload["test/auroc"], payload["test/mrr"]), (0.5, 0.3))
            self.assertIn("train/loss", payload)
            self.assertIn("val/loss", payload)
            self.assertIn("val/auroc", payload)
            self.assertIn("best/auroc", payload)

    def test_h2gb_epoch_keys(self) -> None:
        from src.baselines.minibatch import MinibatchPlan, eval_loader, hgt_loader, train_minibatch_node_classification
        from src.baselines.progress import WandbEpochReport
        from tests.test_baselines_minibatch import full_fanout, synthetic_hetero

        data = synthetic_hetero()
        fanout = full_fanout(data, 1)
        steps = 2
        plan = MinibatchPlan("ogbn-mag", 8, 8 * steps, steps, 2, 1)
        torch.manual_seed(0)
        model = torch.nn.Linear(5, 3)
        ids = {s: data["a"][f"{s}_mask"].nonzero().view(-1) for s in ("val", "test")}
        wandb = fake_wandb()
        with only_wandb(wandb):
            wandb.init(project="p")
            train_minibatch_node_classification(
                model=model, forward_seeds=lambda b: model(b["a"].x[: int(b["a"].batch_size)].float()),
                train_loader=hgt_loader(data, fanout, "a", "train", 8, shuffle=True, steps=steps),
                val_loader=eval_loader(data, fanout, "a", "val", 8), test_loader=eval_loader(data, fanout, "a", "test", 8),
                val_ids=ids["val"], test_ids=ids["test"], val_subset_loader=eval_loader(data, fanout, "a", "val", 8),
                val_subset_ids=ids["val"], target_type="a", optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                criterion=torch.nn.functional.cross_entropy, plan=plan, report=WandbEpochReport("macro_f1"))
        self.assertEqual(len(wandb.logged), 2)
        for _, payload in wandb.logged:
            self.assertIn("train/loss", payload)
            self.assertIn("val_subset/macro_f1", payload)
            self.assertFalse([k for k in payload if k.startswith("test/") or k.startswith("val/")], payload)


class RunTests(unittest.TestCase):
    def test_open_run_is_a_no_op_without_a_project_or_when_disabled(self) -> None:
        """Review F1: a final-seed call without W&B arguments (tests, `--final_seeds` without a project) opens no run."""
        from src.baselines.wandb_runs import open_run

        wandb = fake_wandb()
        with only_wandb(wandb):
            self.assertIsNone(open_run(project=None, kind="final_seed", token="t", seed=0, config={}))
            self.assertIsNone(open_run(project="p", mode="disabled", kind="final_seed", token="t", seed=0, config={}))
            with mock.patch.dict("os.environ", {"WANDB_MODE": "disabled"}):
                self.assertIsNone(open_run(project="p", kind="published", token="t", seed=0, config={}))
            self.assertEqual(wandb.runs, [])
            self.assertIsNotNone(open_run(project="p", mode="offline", kind="published", token="t", seed=0, config={}))

    def test_published_run_opens_logs_and_summarises(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_baseline_published",
                                                      PROJECT_ROOT / "scripts/run_baseline_published.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        from src.baselines.campaign import Cell
        from src.utils.wandb_campaign import CampaignUnit

        def fake_fit_cell(raw_base, values, *, source, device, report=None):
            report(epoch=0, monitor=0.5, running_best=0.5, metrics={"macro_f1": 0.5},
                   extra={"train/loss": 1.0, "val/loss": 0.9, "test/macro_f1": 0.4})
            return {"best_epoch": 0.0, "val_macro_f1": 0.5, "test_macro_f1": 0.4, "model": "gcn"}

        wandb = fake_wandb()
        with tempfile.TemporaryDirectory() as d, only_wandb(wandb), \
                mock.patch.object(runner, "fit_cell", fake_fit_cell):
            args = types.SimpleNamespace(wandb_project="proj", wandb_entity="", wandb_mode="offline",
                                         output_dir=Path(d))
            cell = Cell("gcn", "node_classification", "mag-year")
            record = runner.run_published_seed(cell, CampaignUnit(run_index=0, num_runs=5, seed=3, label="run"),
                                               {"optim.base_lr": 0.001}, device=torch.device("cpu"), args=args)
            written = json.loads((Path(d) / cell.token / "published_seed3.json").read_text())
        run = wandb.runs[0]
        self.assertEqual((run.kwargs["group"], run.kwargs["tags"], run.kwargs["name"], run.kwargs["project"]),
                         (f"{cell.token}_published", ["published"], f"{cell.token}_seed3", "proj"))
        self.assertTrue(run.finished)
        self.assertEqual(set(wandb.logged[0][1]), {"best/macro_f1", "val/macro_f1", "epoch", "train/loss", "val/loss",
                                                    "test/macro_f1"})
        for key in ("selected_epoch", "val_macro_f1", "test_macro_f1", "wall_time_s", "gpu_name", "git_head"):
            self.assertIn(key, run.summary)
        self.assertEqual(run.config.updates[-1]["params"], {"optim.base_lr": 0.001})
        self.assertEqual((record["published_seed"], written["published_seed"]), (3, 3))

    def test_h2snn_final_seed_run_opens_logs_and_summarises(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_h2snn_wandb_tuning",
                                                      PROJECT_ROOT / "scripts/run_h2snn_wandb_tuning.py")
        launcher = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = launcher
        spec.loader.exec_module(launcher)
        import yaml

        from src.configs.config import TrainConfig

        class FakeTrainer:
            def fit(self, data, epoch_callback):
                epoch_callback(0, {"loss": 1.0}, {"macro_f1": 0.6, "loss": 0.8}, {"macro_f1": 0.5})
                self._best_test_metrics = {"val_macro_f1": 0.6, "test_macro_f1": 0.5, "best_epoch": 0.0}

        class FakeLogger:
            def close(self) -> None:
                pass

        raw = yaml.safe_load((PROJECT_ROOT / "configs/experiments/acm_h2snn_v3b.yaml").read_text())
        wandb = fake_wandb()
        with tempfile.TemporaryDirectory() as d, only_wandb(wandb), \
                mock.patch("src.builders.build_experiment_components",
                           lambda cfg: (None, None, None, None, FakeLogger(), None, FakeTrainer())):
            raw["output_dir"] = d
            base_cfg = TrainConfig.from_dict({**raw, "exp_name": "h2snn_node_classification_acm"})
            summary_dir = Path(d) / "h2snn_node_classification_acm"
            summary_dir.mkdir(parents=True)
            (summary_dir / "best_tuned_run0.json").write_text(json.dumps({"sweep_params": {}}))
            args = types.SimpleNamespace(final_seeds=[0], monitor=None, wandb_project="hp", wandb_entity="",
                                         wandb_mode="offline")
            launcher.run_final_seeds(raw, args=args, base_cfg=base_cfg, base_exp_name="h2snn_node_classification_acm")
        run = wandb.runs[0]
        self.assertEqual((run.kwargs["group"], run.kwargs["tags"], run.kwargs["name"]),
                         ("h2snn_node_classification_acm_final", ["final_seed"], "h2snn_node_classification_acm_seed0"))
        self.assertTrue(run.finished)
        payload = wandb.logged[0][1]
        self.assertTrue({"train/loss", "val/macro_f1", "val/loss", "test/macro_f1", "best/macro_f1"} <= set(payload))
        for key in ("selected_epoch", "val_macro_f1", "test_macro_f1", "wall_time_s", "gpu_name", "git_head"):
            self.assertIn(key, run.summary)
        self.assertIn("resolved", run.config.updates[-1])


if __name__ == "__main__":
    unittest.main()
