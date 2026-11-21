"""Card 10 L2 (Amendment 1 option A) and A2.2: the largest grid point of both launchers, the probe-line guard, the
peak line every trial prints, and `scripts/slurm/fit_check.py` (probe lines, 75 % headroom, fit_check.tsv).

(1) `--grid_point max` on both launchers, through their own sweep builders (`run_baseline_tuning.cell_sweep_config`
    on every runnable baseline cell of every task; `run_h2snn_wandb_tuning._build_sweep_for_run` on every
    `*_h2snn_v3b.yaml`): a one-point grid with every size axis at the maximum of its values and every other axis at
    its central value; `min` keeps card 8's smallest point. Mutation (`docs/card10_evidence/mutation_l2.py`): the
    size axes take their minimum -> fails.
(2) Guard: both launchers refuse `--grid_point` without `--dry_run --n_trials 1`.
(3) The peak line: a trial of either launcher prints `FIT_PEAK` with the three phases (a training epoch, the
    whole-split validation and test evaluation) read from that trial; a missing phase is reported as missing.
(4) fit_check: one probe line per cell (tuning and final-seeds lines share it; published lines skipped; options it
    would drop refused); status from the log: peak <= 75 % of GPU memory with all three phases -> fits, above -> oom,
    a missing phase -> error; `peak_fraction` written; `site_split.py` reads the tsv. Mutation
    (`docs/card10_evidence/mutation_a22.py`): threshold 100 % -> a 90 % peak passes -> fails.
No GPU, no real data: the trials are faked at the trainer / adapter boundary.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.slurm import fit_check as F  # noqa: E402  (the A2.2 mutation replaces F)
from scripts.slurm import grid_points as G  # noqa: E402  (the L2 mutation replaces G)
from scripts.tsubame.lines import parse_line, runnable_lines  # noqa: E402
from scripts.tsubame.make_cell_list import (final_seeds_line, h2snn_final_seeds_line,  # noqa: E402
                                            h2snn_tuning_line, tuning_line)

NC = "node_classification"
H2SNN_CONFIGS = sorted((REPO / "configs" / "experiments").glob("*_h2snn_v3b.yaml"))


def numeric_values(spec: dict) -> list:
    return [v for v in spec.get("values", []) if isinstance(v, (int, float)) and not isinstance(v, bool)]


class GridPointTests(unittest.TestCase):
    def check_point(self, space: dict, sweep: dict, where: str) -> int:
        """Every size axis at its maximum, every other at its central value; returns the size axes checked."""
        self.assertEqual(sweep["method"], "grid", where)
        self.assertEqual(set(sweep["parameters"]), set(space), where)
        sizes = 0
        for key, spec in space.items():
            value = sweep["parameters"][key]["value"]
            if key in G.SIZE_AXES:
                sizes += 1
                self.assertEqual(value, max(numeric_values(spec)) if "values" in spec else spec["value"],
                                 f"{where}: size axis {key}")
            else:
                self.assertEqual(value, G.central(spec), f"{where}: {key}")
                if "values" in spec:
                    self.assertIn(value, spec["values"], f"{where}: {key}")
        return sizes

    def test_baseline_max_point_every_cell(self) -> None:
        import scripts.run_baseline_tuning as rbt
        from src.baselines.campaign import build_sweep_config, smallest_grid_point

        with mock.patch.object(rbt, "grid_sweep", G.grid_sweep):
            cells = [c for task in sorted(rbt.DATASETS) for c in rbt.resolve_cells(task, ["all"], ["all"])]
            self.assertGreater(len(cells), 50)
            sizes = 0
            for cell in cells:
                space = build_sweep_config(cell, monitor="val_x", n_trials=1, epochs=1)["parameters"]
                sweep = rbt.cell_sweep_config(cell, monitor="val_x", n_trials=1, epochs=1, grid_point="max")
                sizes += self.check_point(space, sweep, cell.token)
                low = rbt.cell_sweep_config(cell, monitor="val_x", n_trials=1, epochs=1, grid_point="min")
                self.assertEqual({k: v["value"] for k, v in low["parameters"].items()},
                                 {k: smallest_grid_point(s) for k, s in space.items()}, cell.token)
            self.assertGreater(sizes, len(cells))          # most cells have several size axes

    def test_h2snn_max_point_every_config(self) -> None:
        import scripts.run_h2snn_wandb_tuning as launcher
        from src.configs.config import TrainConfig

        self.assertGreaterEqual(len(H2SNN_CONFIGS), 20)
        seen = set()
        with mock.patch.object(launcher, "grid_sweep", G.grid_sweep):
            for path in H2SNN_CONFIGS:
                cfg = TrainConfig.from_dict(launcher.load_raw_config(path))

                def sweep(point):
                    return launcher._build_sweep_for_run(
                        args=argparse.Namespace(grid_point=point, d_values=None), base_cfg=cfg,
                        base_exp_name="x", monitor="macro_f1", run_index=0)

                space = sweep(None)["parameters"]
                self.check_point(space, sweep("max"), path.name)
                seen |= set(space) & G.SIZE_AXES
                point = {k: v["value"] for k, v in sweep("max")["parameters"].items()}
                self.assertEqual((point["d"], point["wisp_max_length"], point["wisp_walks_per_anchor"]), (64, 5, 64))
                if "batch_size" in point:
                    self.assertEqual(point["batch_size"], 1024)
        self.assertEqual(seen, {"d", "wisp_max_length", "wisp_walks_per_anchor", "batch_size"})

    def test_unclassified_key_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not classified"):
            G.fixed_point({"hidden": {"values": [1, 2]}, "new_width": {"values": [8, 16]}}, "max")

    def test_central_values(self) -> None:
        self.assertEqual(G.central({"values": [0.0, 1e-05, 5e-05, 0.0001, 0.0005]}), 5e-05)
        self.assertEqual(G.central({"values": ["mean", "max", "add", "attention"]}), "max")
        self.assertAlmostEqual(G.central({"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2}), 1e-3)
        self.assertAlmostEqual(G.central({"distribution": "uniform", "min": 0.0, "max": 0.9}), 0.45)


class GuardTests(unittest.TestCase):
    def run_launcher(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, *argv], cwd=REPO, capture_output=True, text=True, timeout=600,
                              env={**os.environ, "WANDB_MODE": "disabled"})

    def test_both_launchers_refuse_grid_point_outside_probe_lines(self) -> None:
        h2 = ["scripts/run_h2snn_wandb_tuning.py", "--config", "configs/experiments/acm_h2snn_v3b.yaml"]
        bl = ["scripts/run_baseline_tuning.py", "--task", NC, "--models", "gcn", "--datasets", "acm"]
        for argv in ([*h2, "--grid_point", "max", "--n_trials", "1"],
                     [*h2, "--grid_point", "max", "--dry_run", "--n_trials", "2"],
                     [*bl, "--grid_point", "max", "--n_trials", "1"],
                     [*bl, "--grid_point", "max", "--dry_run", "--n_trials", "2"]):
            with self.subTest(argv=argv):
                proc = self.run_launcher(*argv)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("--grid_point is for probe lines", proc.stdout + proc.stderr)


class PeakLineTests(unittest.TestCase):
    def peak(self, text: str) -> dict:
        lines = [x for x in text.splitlines() if x.startswith("FIT_PEAK ")]
        self.assertEqual(len(lines), 1, text)
        return json.loads(lines[0][len("FIT_PEAK "):])

    def h2snn_trial(self, call_callback: bool, record: dict) -> dict:
        import scripts.run_h2snn_wandb_tuning as launcher
        import src.builders

        class Trainer:
            def fit(self, data, epoch_callback=None):
                if call_callback:   # B1a-like: the callback sees training and the validation subset only
                    epoch_callback(0, {"loss": 1.0}, {"macro_f1": 0.5}, {})
                self._best_test_metrics = dict(record)

        logger = mock.Mock()
        raw = launcher.load_raw_config(REPO / "configs" / "experiments" / "acm_h2snn_v3b.yaml")
        space = launcher.build_sweep_config(task_type=NC, metric="macro_f1", epochs=1, dataset_name="acm")
        params = {k: v["value"] for k, v in G.fixed_point(space["parameters"], "max").items()}
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out), \
                mock.patch.object(src.builders, "build_experiment_components",
                                  return_value=(None, None, None, None, logger, None, Trainer())), \
                mock.patch.object(launcher, "enrich_run_record", side_effect=lambda r, *a, **k: dict(r)):
            raw["device"] = "cpu"
            launcher.train_trial(raw, params=params, run_index=0, num_runs=1, seed=0, base_exp_name="x",
                                 output_dir=Path(d), monitor="macro_f1", trial_name="t0")
        return self.peak(out.getvalue())

    def test_h2snn_trial_prints_three_phases(self) -> None:
        full = {"val_macro_f1": 0.5, "test_macro_f1": 0.4, "val_loss": 1.0}
        self.assertEqual(self.h2snn_trial(True, full)["phases"], {"train": True, "val": True, "test": True})
        self.assertEqual(self.h2snn_trial(False, full)["phases"], {"train": False, "val": True, "test": True})
        self.assertEqual(self.h2snn_trial(True, {"val_macro_f1": 0.5})["phases"],
                         {"train": True, "val": True, "test": False})

    def test_baseline_trial_prints_three_phases(self) -> None:
        import scripts.run_baseline_tuning as rbt

        for record, phases in (({"val_macro_f1": 0.5, "test_macro_f1": 0.4}, {"train": True, "val": True, "test": True}),
                               ({"val_macro_f1": 0.5}, {"train": True, "val": True, "test": False})):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), mock.patch.object(rbt, "fit_cell", return_value=dict(record)), \
                    mock.patch.object(rbt, "resolve_device", return_value=mock.Mock(type="cpu")):
                rbt.train_trial({}, params={"hidden_dim": 64}, allowed_parameters=None, run_index=0, num_runs=1,
                                seed=0, base_exp_name="x", output_dir=Path("."), monitor="val_macro_f1",
                                trial_name="t0", unit_label="run", unit_name=None, record_extra=None)
            self.assertEqual(self.peak(out.getvalue())["phases"], phases)


def log(fraction: float | None, phases: dict | None = None, extra: str = "") -> str:
    total = 24 * 2 ** 30
    body = ["head=x", "agent: python ...", extra]
    if fraction is not None:
        body.append("FIT_PEAK " + json.dumps({"cuda": True, "peak_bytes": int(fraction * total), "total_bytes": total,
                                              "phases": phases or {"train": True, "val": True, "test": True}}))
    return "\n".join(body + ["exit=0"])


class FitCheckTests(unittest.TestCase):
    def test_headroom_and_phases(self) -> None:
        self.assertEqual(F.classify(log(0.70))[0], "fits")
        self.assertEqual(F.classify(log(0.75))[0], "fits")                 # 18 GB of 24 GB
        self.assertEqual(F.classify(log(0.90))[0], "oom")                  # fits the GPU, not the 75 % headroom
        status, mib, fraction = F.classify(log(0.90))
        self.assertAlmostEqual(fraction, 0.90, places=3)
        self.assertAlmostEqual(mib, 0.90 * 24 * 1024, delta=1)
        for phase in ("train", "val", "test"):                             # every phase must be measured
            phases = {"train": True, "val": True, "test": True, phase: False}
            self.assertEqual(F.classify(log(0.10, phases))[0], "error", phase)
        self.assertEqual(F.classify(log(None, extra="torch.OutOfMemoryError: CUDA out of memory."))[0], "oom")
        self.assertEqual(F.classify(log(None, extra="ValueError: x"))[0], "error")
        self.assertEqual(F.classify(None)[0], "error")

    def test_build_read_and_site_split(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out, fit_out = root / "out", root / "fit"
            lines = [tuning_line(NC, "gcn", "acm", out), final_seeds_line(NC, "gcn", "acm", out),
                     h2snn_tuning_line(NC, "acm", out), h2snn_final_seeds_line(NC, "dblp", out),
                     tuning_line(NC, "gat", "acm", out),
                     f"scripts/run_baseline_published.py --task {NC} --models han --datasets acm --output_dir "
                     f"{out}/experiments/baselines --final_seeds 0 1 2 3 4"]
            list_path = root / "cells_tuning.txt"
            list_path.write_text("# x\n" + "\n".join(lines) + "\n")
            probes, skipped = F.build(list_path, fit_out)
            self.assertEqual([parse_line(p).token for p in probes],
                             ["gcn_node_classification_acm", "h2snn_node_classification_acm",
                              "h2snn_node_classification_dblp", "gat_node_classification_acm"])
            self.assertEqual(len(skipped), 1)
            self.assertIn("han_node_classification_acm", skipped[0])
            for p in probes:
                self.assertIn("--n_trials 1 --dry_run --grid_point max", p)
                self.assertNotIn(str(out), p)                                  # writes only under fit_out
                self.assertIn(str(fit_out), p)
                self.assertIn("_fitcheck", p)
            self.assertIn("--config configs/experiments/dblp_h2snn_v3b.yaml", probes[2])
            self.assertEqual(runnable_lines((fit_out / F.PROBE_LIST).read_text()), probes)
            # logs as lab_slots.sbatch names them; line 4 has no log
            logs = fit_out / "logs" / "lines"
            logs.mkdir(parents=True)
            for number, text in ((1, log(0.5)), (2, log(0.9)), (3, log(None, extra="CUDA out of memory"))):
                (logs / f"fit_lines.line{number:03d}.700.{number}.0.log").write_text(text)
            rows = F.read(fit_out, "3090")
            self.assertEqual([r["status"] for r in rows], ["fits", "oom", "oom", "error"])
            with (fit_out / "fit_check.tsv").open(newline="") as handle:
                table = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(tuple(table[0]), F.FIT_COLUMNS)
            self.assertEqual(table[0]["peak_fraction"], "0.5000")
            # site_split reads it: only the fitting cell goes to the lab
            from scripts.slurm import site_split
            (root / "lab").mkdir()
            (root / "t4").mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                site_split.main(["--list", str(list_path), "--fit-check", str(fit_out / "fit_check.tsv"),
                                 "--gpu-type", "3090", "--lab-out", str(root / "lab"), "--t4-out", str(root / "t4")])
            lab = runnable_lines((root / "cells_tuning.lab.txt").read_text())
            self.assertEqual(lab, lines[:2])

    def test_dropped_option_refused(self) -> None:
        line = tuning_line(NC, "gcn", "acm", Path("/o")) + " --patience 3"
        with self.assertRaisesRegex(ValueError, "would drop"):
            F.probe_line(line, Path("/f"))


if __name__ == "__main__":
    unittest.main()
