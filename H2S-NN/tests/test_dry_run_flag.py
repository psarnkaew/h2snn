"""Card 2 D3: the explicit dry-run flag of the tuning and seeds launchers.

`--dry_run` sets the epoch budget to 1 (a check that the launcher works end to end, never a result). It is a
command-line flag only: it cannot be set in a tuning sweep config (neither `dry_run` nor `epochs` is a sweep
parameter or a base-config key), and the generated sweep config is the same with and without it.
No data, no W&B, no training.
"""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _args(**overrides) -> argparse.Namespace:
    values = dict(exp_name=None, output_dir=None, device=None, patience=None, epochs=None, decoder_type=None,
                  wandb_tags=None, dry_run=False)
    values.update(overrides)
    return argparse.Namespace(**values)


class DryRunFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = yaml.safe_load((REPO / "configs" / "experiments" / "acm_h2snn_v3b.yaml").read_text())

    def test_flag_sets_one_epoch_and_tags_the_runs(self) -> None:
        from scripts.run_h2snn_wandb_tuning import _apply_base_cli
        from src.configs.config import TrainConfig

        args = _args(dry_run=True)
        payload = _apply_base_cli(self.raw, args)
        self.assertEqual(payload["epochs"], 1)
        self.assertIn("dry-run", args.wandb_tags)
        self.assertEqual(TrainConfig.from_dict(payload).epochs, 1)
        plain = _apply_base_cli(self.raw, _args())
        self.assertEqual(TrainConfig.from_dict(plain).epochs, 500)  # the HGB NC budget (card 16), untouched
        with self.assertRaises(ValueError):
            _apply_base_cli(self.raw, _args(dry_run=True, epochs=5))

    def test_flag_cannot_be_set_in_a_sweep_config(self) -> None:
        from scripts.run_h2snn_wandb_tuning import (KNOWN_PARAMETER_NAMES, apply_sweep_overrides,
                                                    build_sweep_config, validate_trial_params)
        from src.configs.config import TrainConfig

        for name, value in (("dry_run", True), ("epochs", 1)):
            with self.subTest(parameter=name):
                self.assertNotIn(name, KNOWN_PARAMETER_NAMES)
                with self.assertRaisesRegex(ValueError, "not a tuning parameter"):
                    validate_trial_params({name: value})
                with self.assertRaisesRegex(ValueError, "not a tuning parameter"):
                    validate_trial_params({"d": 16}, allowed_parameters={"d", name})
                with self.assertRaisesRegex(ValueError, "not a tuning parameter"):
                    apply_sweep_overrides(self.raw, {name: value})
        with self.assertRaises(ValueError):  # not a base-config key either (strict config loading)
            TrainConfig.from_dict({**self.raw, "dry_run": True})
        for task, dataset in (("node_classification", "acm"), ("link_prediction", "acm_pa"),
                              ("graph_classification", "mutag")):
            sweep = build_sweep_config(task_type=task, metric="auroc", epochs=1, dataset_name=dataset)
            self.assertEqual(sweep, build_sweep_config(task_type=task, metric="auroc", epochs=300,
                                                       dataset_name=dataset))
            self.assertFalse({"dry_run", "epochs"} & set(sweep["parameters"]))

    def test_seeds_launcher_flag_sets_one_epoch(self) -> None:
        from scripts.run_h2snn_runs import _apply_common_overrides
        from src.configs.config import TrainConfig

        cfg = TrainConfig.from_dict(self.raw)
        args = _args(dry_run=True, monitor=None, train_ratio=None, val_ratio=None, test_ratio=None,
                     use_wandb=False, wandb_project=None, wandb_entity=None, wandb_mode=None, wandb_group=None,
                     wandb_log_artifacts=False)
        cfg = _apply_common_overrides(cfg, args)
        self.assertEqual(cfg.epochs, 1)
        self.assertIn("dry-run", cfg.wandb_tags)


class LauncherFixTests(unittest.TestCase):
    """Card 2 D3: two launcher bugs found by the dry run on real data (one synthetic test each)."""

    def test_baseline_worker_is_spawned_without_task(self) -> None:
        """`run_unit_sweep` re-spawns `run_baseline_tuning.py --worker_context <file>` only; with `--task` required
        argparse exited and no baseline trial ever ran (D3, GCN on ACM)."""
        import sys
        from unittest import mock

        import scripts.run_baseline_tuning as launcher

        with mock.patch.object(sys, "argv", ["run_baseline_tuning.py", "--worker_context", "ctx.json"]):
            args = launcher.parse_args()
        self.assertEqual(str(args.worker_context), "ctx.json")
        with mock.patch.object(sys, "argv", ["run_baseline_tuning.py"]), self.assertRaises(SystemExit):
            launcher.main()  # without a worker context, --task is still required

    def test_baseline_unit_summary_selects_by_the_monitor(self) -> None:
        """`run_unit_sweep` calls `summarize(records, monitor, path)`; the baseline launcher passed the monitor as the
        task of `summarize_cell` and every sweep raised after its trials (D3, GCN on ACM). Synthetic records."""
        import inspect
        import json
        import tempfile

        import scripts.run_baseline_tuning as launcher

        self.assertIn("summarize=summarize_unit", inspect.getsource(launcher.main))
        records = [{"trial_name": "a", "val_macro_f1": 0.2}, {"trial_name": "b", "val_macro_f1": 0.7},
                   {"trial_name": "c", "val_macro_f1": 0.5}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_tuned_run0.json"
            best = launcher.summarize_unit(records, "macro_f1", path)
            self.assertEqual(best["trial_name"], "b")
            self.assertEqual(json.loads(path.read_text())["trial_name"], "b")

    def test_final_seeds_run_without_a_wandb_run(self) -> None:
        """`--final_seeds` trains outside a sweep worker: no W&B run exists, and the epoch callback's wandb.log
        raised (D3, ACM / acm_pa / MUTAG). Real wandb module, no run, synthetic ACM-shaped graph."""
        import json
        import tempfile

        import wandb

        import scripts.run_h2snn_wandb_tuning as tuning
        from tests.partO_fixtures import acm_like_hetero, patched_acm_loader
        from tests.test_partO_reload import _raw_config

        self.assertIsNone(wandb.run)
        with tempfile.TemporaryDirectory() as tmp:
            raw = _raw_config(Path(tmp), "exp", seed=0)
            raw["epochs"] = 1
            summary = Path(tmp) / "exp"
            summary.mkdir(parents=True)
            (summary / "best_tuned_run0.json").write_text(json.dumps({"sweep_params": {"lr": 0.01}}))
            with patched_acm_loader(acm_like_hetero(seed=1)):
                records = tuning.run_final_seeds(raw, args=argparse.Namespace(final_seeds=[0], monitor=None),
                                                 base_cfg=tuning.TrainConfig.from_dict(raw), base_exp_name="exp")
            self.assertEqual(len(records), 1)
            self.assertTrue((summary / "final_seed0.json").is_file())


if __name__ == "__main__":
    unittest.main()
