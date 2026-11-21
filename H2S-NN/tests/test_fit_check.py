"""Card 10 L2 (Amendment 1 option A), A2.2 and A3: the largest grid point of both launchers, the probe-line guard,
the peak line every trial prints, and `scripts/slurm/fit_check.py` (probe lines, 75 % headroom, fit_check.tsv).

(1) `--grid_point max` on both launchers, through their own sweep builders (`run_baseline_tuning.cell_sweep_config`
    on every runnable baseline cell of every task; `run_h2snn_wandb_tuning._build_sweep_for_run` on every
    `*_h2snn_v3b.yaml`): a grid with every size axis at the maximum of its values, every memory-relevant key at its
    most memory-hungry value or, where the code does not show it, over its values (A3.1; EXPECTED_HUNGRY below,
    written independently of the script's table), every other axis at its central value; `min` keeps card 8's
    smallest point; the launchers run one trial per grid point and announce it (FIT_GRID). Mutations
    (`docs/card10_evidence/mutation_l2.py`, `mutation_a31.py`): the size axes take their minimum, or the
    memory-relevant keys their central value -> fails.
(2) Guard: both launchers refuse `--grid_point` without `--dry_run --n_trials 1`; the published runner refuses
    `--dry_run` without exactly one seed; it has no `--resume` (the slot runners do not pass it, A3.3).
(3) The peak line: a trial of either launcher, and a published seed, prints `FIT_PEAK` with the three phases (a
    training epoch, the whole-split validation and test evaluation) read from that trial; a missing phase is
    reported as missing.
(4) fit_check: one probe line per cell (tuning and final-seeds lines share it; options it would drop refused);
    status from the log: the largest peak over the announced trials <= 75 % of GPU memory with all three phases in
    every trial -> fits, above -> oom, a missing trial or phase -> error; `peak_fraction` written; `site_split.py`
    reads the tsv. Mutation (`docs/card10_evidence/mutation_a22.py`): threshold 100 % -> a 90 % peak passes -> fails.
(5) A3.2 published lines: a probe at the published configuration (`run_baseline_published.py --dry_run`, one seed,
    one epoch), a fit_check.tsv row, `fits` -> lab and otherwise TSUBAME, H2GB published cells always TSUBAME.
    Mutation (`docs/card10_evidence/mutation_a32.py`): published lines left out of the probe list -> fails.
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

from scripts.slurm import fit_check as F  # noqa: E402  (the A2.2 / A3.2 mutations replace F)
from scripts.slurm import grid_points as G  # noqa: E402  (the L2 / A3.1 mutations replace G)
from scripts.tsubame.lines import parse_line, runnable_lines  # noqa: E402
from scripts.tsubame.make_cell_list import (final_seeds_line, h2snn_final_seeds_line,  # noqa: E402
                                            h2snn_tuning_line, published_line, tuning_line)

NC = "node_classification"
LP = "link_prediction"
#: A3.1, the decisions as the ops review and the source code give them (a list = probe every value, in this order)
EXPECTED_HUNGRY = {
    "feats_type": [0, 2, 3], "label_feats": True, "residual": True, "sheaf_model": ["BundleSheaf", "GeneralSheaf"],
    "orth": "cayley", "add_lp": True, "add_hp": True, "conformal": True, "gnn_type": "GAT", "metapath_pe": True,
    "mask_lp": True, "r": [0.3, 0.5], "poly": ["GPR_GPR", "GPR_legendre", "GPR_jaco"], "attn_dropout": 0.2,
    "decoder_type": "distmult", "graph_pooling": "attention",
    "sheaf": [f"{f}/{x}" for f in ("bundle", "general") for x in (
        "local_concat", "type_concat", "type_ensemble", "edge_type_concat", "node_type_concat", "node_type",
        "edge_type", "types_only", "attention", "attention_type_concat", "attention_type_ensemble",
        "attention_edge_encoding", "attention_node_encoding", "attention_types_only", "attention_node_type",
        "attention_edge_type")],
}
H2SNN_CONFIGS = sorted((REPO / "configs" / "experiments").glob("*_h2snn_v3b.yaml"))


def numeric_values(spec: dict) -> list:
    return [v for v in spec.get("values", []) if isinstance(v, (int, float)) and not isinstance(v, bool)]


class GridPointTests(unittest.TestCase):
    def check_point(self, space: dict, sweep: dict, where: str) -> int:
        """Every size axis at its maximum, every memory-relevant key as EXPECTED_HUNGRY, every other at its central
        value, and the grid size; returns the size axes checked."""
        self.assertEqual(sweep["method"], "grid", where)
        self.assertEqual(set(sweep["parameters"]), set(space), where)
        sizes, points = 0, 1
        for key, spec in space.items():
            got = sweep["parameters"][key]
            if key in G.SIZE_AXES:
                sizes += 1
                self.assertEqual(got, {"value": max(numeric_values(spec)) if "values" in spec else spec["value"]},
                                 f"{where}: size axis {key}")
            elif key in EXPECTED_HUNGRY:
                want = EXPECTED_HUNGRY[key]
                offered = spec.get("values", [spec.get("value")])
                if isinstance(want, list):
                    grid = [v for v in want if v in offered]
                    self.assertEqual(got, {"value": grid[0]} if len(grid) == 1 else {"values": grid}, f"{where}: {key}")
                    points *= len(grid)
                else:
                    self.assertEqual(got, {"value": want}, f"{where}: memory-relevant {key}")
            else:
                self.assertEqual(got, {"value": G.central(spec)}, f"{where}: {key}")
                if "values" in spec:
                    self.assertIn(got["value"], spec["values"], f"{where}: {key}")
        self.assertEqual(G.grid_size(sweep), points, where)
        return sizes

    def test_baseline_max_point_every_cell(self) -> None:
        import scripts.run_baseline_tuning as rbt
        from src.baselines.campaign import build_sweep_config, smallest_grid_point

        with mock.patch.object(rbt, "grid_sweep", G.grid_sweep):
            cells = [c for task in sorted(rbt.DATASETS) for c in rbt.resolve_cells(task, ["all"], ["all"])]
            self.assertGreater(len(cells), 50)
            sizes, memory = 0, set()
            for cell in cells:
                space = build_sweep_config(cell, monitor="val_x", n_trials=1, epochs=1)["parameters"]
                sweep = rbt.cell_sweep_config(cell, monitor="val_x", n_trials=1, epochs=1, grid_point="max")
                sizes += self.check_point(space, sweep, cell.token)
                memory |= set(space) & set(EXPECTED_HUNGRY)
                low = rbt.cell_sweep_config(cell, monitor="val_x", n_trials=1, epochs=1, grid_point="min")
                self.assertEqual({k: v["value"] for k, v in low["parameters"].items()},
                                 {k: smallest_grid_point(s) for k, s in space.items()}, cell.token)
            self.assertGreater(sizes, len(cells))          # most cells have several size axes
            self.assertEqual(memory, set(EXPECTED_HUNGRY) - {"graph_pooling"})   # every baseline memory key seen

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
                seen |= set(space) & (G.SIZE_AXES | {"decoder_type", "graph_pooling"})
                point = {k: v["value"] for k, v in sweep("max")["parameters"].items()}   # H2S-NN: one point
                self.assertEqual((point["d"], point["wisp_max_length"], point["wisp_walks_per_anchor"]), (64, 5, 64))
                if "batch_size" in point:
                    self.assertEqual(point["batch_size"], 1024)
                if "decoder_type" in point:
                    self.assertEqual(point["decoder_type"], "distmult")
                if "graph_pooling" in point:
                    self.assertEqual(point["graph_pooling"], "attention")
        self.assertEqual(seen, {"d", "wisp_max_length", "wisp_walks_per_anchor", "batch_size", "decoder_type",
                                "graph_pooling"})

    def test_memory_table(self) -> None:
        """A3.1: the script's table holds exactly the expected decisions, each with a file:line reason; the three
        classes are disjoint."""
        self.assertEqual(set(G.MEMORY_AXES), set(EXPECTED_HUNGRY))
        for key, rule in G.MEMORY_AXES.items():
            self.assertEqual(rule.get("values", rule.get("value")), EXPECTED_HUNGRY[key], key)
            self.assertRegex(rule["why"], r"\.py:\d+", key)
        self.assertFalse(G.SIZE_AXES & set(G.MEMORY_AXES) or G.SIZE_AXES & G.OTHER_AXES
                         or G.OTHER_AXES & set(G.MEMORY_AXES))

    def test_launchers_run_one_trial_per_grid_point(self) -> None:
        """The baseline launcher on NSD ACM NC (sheaf_model 2 x feats_type 3 = 6 trials) and GCN ACM NC (feats_type:
        3); the H2S-NN launcher on ACM LP (decoder fixed: 1). Each announces FIT_GRID; the sweep is not started."""
        import scripts.run_baseline_tuning as rbt
        import scripts.run_h2snn_wandb_tuning as launcher

        class Stop(Exception):
            pass

        seen: dict = {}

        def capture(*args, campaign, sweep_config, **kwargs):
            seen.update(n=campaign.n_trials, sweep=sweep_config)
            raise Stop

        for model, points in (("nsd", 6), ("gcn", 3)):
            out = io.StringIO()
            argv = ["x", "--task", NC, "--models", model, "--datasets", "acm", "--num_runs", "1", "--n_trials", "1",
                    "--dry_run", "--grid_point", "max", "--wandb_mode", "disabled"]
            with self.subTest(model=model), mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(rbt, "run_unit_sweep", side_effect=capture), contextlib.redirect_stdout(out), \
                    mock.patch.object(rbt, "grid_sweep", G.grid_sweep), mock.patch.object(rbt, "grid_size", G.grid_size):
                with self.assertRaises(Stop):
                    rbt.main()
                self.assertEqual(seen["n"], points)
                self.assertIn(f'FIT_GRID {{"points": {points}}}', out.getvalue())
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            argv = ["x", "--config", "configs/experiments/acm_pa_h2snn_v3b.yaml", "--exp_name", "h2snn_x",
                    "--output_dir", d, "--n_trials", "1", "--dry_run", "--grid_point", "max", "--wandb_mode",
                    "disabled"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(launcher, "run_unit_sweep", side_effect=capture), \
                    contextlib.redirect_stdout(out), mock.patch.object(launcher, "grid_sweep", G.grid_sweep), \
                    mock.patch.object(launcher, "grid_size", G.grid_size):
                cwd = os.getcwd()
                os.chdir(REPO)
                try:
                    with self.assertRaises(Stop):
                        launcher.main()
                finally:
                    os.chdir(cwd)
            self.assertEqual(seen["n"], 1)
            self.assertEqual(seen["sweep"]["parameters"]["decoder_type"], {"value": "distmult"})
            self.assertIn('FIT_GRID {"points": 1}', out.getvalue())

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

    def test_published_runner_guard_and_no_resume(self) -> None:
        pub = ["scripts/run_baseline_published.py", "--task", LP, "--models", "gcn", "--datasets", "lastfm"]
        for argv in ([*pub, "--dry_run"], [*pub, "--dry_run", "--final_seeds", "0", "1"]):
            with self.subTest(argv=argv):
                proc = self.run_launcher(*argv)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("--dry_run is for fit-check probe lines", proc.stdout + proc.stderr)
        # A3.3: no dummy --resume in the runner; the slot scripts pass none to a published line (test_slot_resume)
        proc = self.run_launcher("scripts/run_baseline_published.py", "--task", LP, "--list", "--resume")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("unrecognized arguments: --resume", proc.stderr)


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

    def test_published_seed_dry_run(self) -> None:
        """A3.2: a published seed under --dry_run trains one epoch (raw_base epochs 1) at the published values and
        prints FIT_PEAK with its three phases."""
        import scripts.run_baseline_published as rbp
        from src.baselines.campaign import Cell, monitor_for

        monitor = monitor_for(LP).removeprefix("val_")
        seen: dict = {}

        def fit(raw_base, values, **kwargs):
            seen.update(raw_base=raw_base, values=values)
            return {f"val_{monitor}": 0.5, f"test_{monitor}": 0.4}

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out), \
                mock.patch.object(rbp, "fit_cell", side_effect=fit):
            args = argparse.Namespace(dry_run=True, wandb_project=None, wandb_entity="", wandb_mode="disabled",
                                      output_dir=Path(d))
            cell = Cell(model="gcn", task=LP, dataset="lastfm")
            unit = rbp.seed_units([0])[0]
            rbp.run_published_seed(cell, unit, {"hidden_dim": 64}, device=mock.Mock(type="cpu"), args=args)
        self.assertEqual(seen["raw_base"]["epochs"], 1)
        self.assertEqual(seen["values"], {"hidden_dim": 64})
        self.assertEqual(self.peak(out.getvalue())["phases"], {"train": True, "val": True, "test": True})


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

    def test_grid_probe_takes_the_largest_peak_of_every_trial(self) -> None:
        def grid(*fractions: float, points: int = 3, extra: str = "") -> str:
            return "\n".join([f'FIT_GRID {{"points": {points}}}', extra] + [log(f) for f in fractions])

        self.assertEqual(F.classify(grid(0.3, 0.5, 0.7))[0], "fits")
        status, _, fraction = F.classify(grid(0.3, 0.9, 0.5))
        self.assertEqual((status, round(fraction, 3)), ("oom", 0.9))            # the largest trial decides
        self.assertEqual(F.classify(grid(0.3, 0.5))[0], "error")                # a trial printed no peak
        self.assertEqual(F.classify(grid(0.3, 0.5, extra="CUDA out of memory"))[0], "oom")

    def test_build_read_and_site_split(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out, fit_out = root / "out", root / "fit"
            lines = [tuning_line(NC, "gcn", "acm", out), final_seeds_line(NC, "gcn", "acm", out),
                     h2snn_tuning_line(NC, "acm", out), h2snn_final_seeds_line(NC, "dblp", out),
                     tuning_line(NC, "gat", "acm", out)]
            list_path = root / "cells_tuning.txt"
            list_path.write_text("# x\n" + "\n".join(lines) + "\n")
            probes = F.build(list_path, fit_out)
            self.assertEqual([parse_line(p).token for p in probes],
                             ["gcn_node_classification_acm", "h2snn_node_classification_acm",
                              "h2snn_node_classification_dblp", "gat_node_classification_acm"])
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

    def test_published_lines_get_a_row_and_follow_it(self) -> None:
        """A3.2: every published line has a probe and a fit_check.tsv row; fits -> lab, oom / error -> TSUBAME; an H2GB
        published cell -> TSUBAME even when it fits (range rule)."""
        from scripts.slurm import site_split
        from src.configs.config import H2GB_NC_DATASETS

        self.assertEqual(tuple(site_split.H2GB_NC_DATASETS), tuple(H2GB_NC_DATASETS))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out, fit_out = root / "out", root / "fit"
            lines = [published_line(LP, "gcn", "lastfm", out), published_line(LP, "rgcn", "pubmed", out),
                     published_line(NC, "pshgcn", "imdb", out), published_line(NC, "sehgnn", "ogbn-mag", out),
                     tuning_line(NC, "gcn", "acm", out)]
            list_path = root / "cells_tuning.txt"
            list_path.write_text("\n".join(lines) + "\n")
            probes = F.build(list_path, fit_out)
            self.assertEqual([parse_line(p).token for p in probes], [parse_line(x).token for x in lines])
            for p in probes[:4]:
                self.assertTrue(p.startswith("scripts/run_baseline_published.py "))
                self.assertIn("--final_seeds 0 --dry_run", p)
                self.assertNotIn("--grid_point", p)
                self.assertIn(str(fit_out), p)
                self.assertNotIn(str(out), p)
            logs = fit_out / "logs" / "lines"
            logs.mkdir(parents=True)
            for number, fraction in ((1, 0.4), (2, 0.9), (4, 0.2), (5, 0.5)):    # line 3 (pshgcn imdb): no log
                (logs / f"fit_lines.line{number:03d}.700.{number}.0.log").write_text(log(fraction))
            rows = F.read(fit_out, "3090")
            self.assertEqual([(r["cell"], r["status"]) for r in rows],
                             [("gcn_link_prediction_lastfm", "fits"), ("rgcn_link_prediction_pubmed", "oom"),
                              ("pshgcn_node_classification_imdb", "error"),
                              ("sehgnn_node_classification_ogbn-mag", "fits"), ("gcn_node_classification_acm", "fits")])
            (root / "lab").mkdir()
            (root / "t4").mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                site_split.main(["--list", str(list_path), "--fit-check", str(fit_out / "fit_check.tsv"),
                                 "--gpu-type", "3090", "--lab-out", str(root / "lab"), "--t4-out", str(root / "t4")])
            lab = runnable_lines((root / "cells_tuning.lab.txt").read_text())
            t4 = runnable_lines((root / "cells_tuning.t4.txt").read_text())
            self.assertEqual(lab, [lines[0], lines[4]])
            self.assertEqual(t4, lines[1:4])

    def test_dropped_option_refused(self) -> None:
        line = tuning_line(NC, "gcn", "acm", Path("/o")) + " --patience 3"
        with self.assertRaisesRegex(ValueError, "would drop"):
            F.probe_line(line, Path("/f"))


if __name__ == "__main__":
    unittest.main()
