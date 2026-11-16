"""Card 2 D6.8 (author decision 6): the baseline launcher's final-seeds step, as H2S-NN's `--final_seeds`.

The best trial of a finished seed-0 sweep is selected by the validation monitor, and its parameters are used unchanged
for each seed of H2S-NN's final-seed list, with the tuning budget. Synthetic trial records; `fit_cell` is replaced by a
recorder (no data, no training, no W&B). Mutation check (card): selecting the last trial instead fails this test.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class BaselineFinalSeedsTests(unittest.TestCase):
    def test_best_trial_by_monitor_and_its_params_unchanged_per_seed(self) -> None:
        import scripts.run_baseline_tuning as launcher
        from scripts.run_h2snn_wandb_tuning import FINAL_SEEDS
        from src.baselines.campaign import Cell

        # the best trial (by val_macro_f1) is neither the first nor the last
        trials = [
            {"trial_name": "t0", "val_macro_f1": 0.31, "sweep_params": {"lr": 0.01, "hidden_dim": 64, "dropout": 0.1}},
            {"trial_name": "t1", "val_macro_f1": 0.74, "sweep_params": {"lr": 0.002, "hidden_dim": 256, "dropout": 0.5}},
            {"trial_name": "t2", "val_macro_f1": 0.52, "sweep_params": {"lr": 0.0003, "hidden_dim": 128, "dropout": 0.7}},
        ]
        cell = Cell(model="gcn", task="node_classification", dataset="acm")
        calls: list[tuple[dict, dict, str]] = []

        def fake_fit(raw_base, params, *, source, device, report=None):
            calls.append((dict(raw_base), dict(params), source))
            return {"val_macro_f1": 0.0, "test_macro_f1": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / cell.token).mkdir(parents=True)
            (out / cell.token / "tuning_trials_run0.json").write_text(json.dumps(trials))
            with mock.patch.object(launcher, "fit_cell", fake_fit), \
                    mock.patch.object(launcher, "summarize_cell", lambda records, task, path: {}):
                records = launcher.run_final_seeds(cell, seeds=list(FINAL_SEEDS), output_dir=out,
                                                   monitor="macro_f1", device="cpu", epochs=7, patience=3)
            self.assertEqual(len(records), len(FINAL_SEEDS))
            self.assertEqual([c[0]["seed"] for c in calls], list(FINAL_SEEDS))  # H2S-NN's seed list (0-4)
            for raw_base, params, source in calls:
                self.assertEqual(params, trials[1]["sweep_params"])  # the best by the monitor, unchanged
                self.assertEqual(source, "tuned")
                self.assertEqual((raw_base["epochs"], raw_base["patience"]), (7, 3))  # the tuning budget passed on
                self.assertEqual((raw_base["model"], raw_base["task"], raw_base["dataset"]),
                                 ("gcn", "node_classification", "acm"))
            for seed in FINAL_SEEDS:
                saved = json.loads((out / cell.token / f"final_seed{seed}.json").read_text())
                self.assertEqual(saved["selected_trial"], "t1")
                self.assertEqual(saved["sweep_params"], trials[1]["sweep_params"])

    def test_trial_record_keeps_its_params(self) -> None:
        import torch

        import scripts.run_baseline_tuning as launcher

        params = {"lr": 0.001, "hidden_dim": 64}
        with mock.patch.object(launcher, "fit_cell", lambda *a, **k: {"val_macro_f1": 0.0}), \
                mock.patch.object(launcher, "WandbEpochReport", lambda monitor: None), \
                mock.patch.object(launcher, "resolve_device", lambda raw: torch.device("cpu")):
            record = launcher.train_trial({"model": "gcn"}, params=params, allowed_parameters=None, run_index=0,
                                          num_runs=1, seed=0, base_exp_name="x", output_dir=Path("."),
                                          monitor="macro_f1", trial_name="t", unit_label="run", unit_name=None,
                                          record_extra=None)
        self.assertEqual(record["sweep_params"], params)


if __name__ == "__main__":
    unittest.main()
