"""Card 7 T9 (Phase B, V8): the TSUBAME phases `ablation`, `perturbation` and `timing` of
`scripts/tsubame/make_cell_list.py` and the row wrapper `scripts/tsubame/run_row.py`.

- line counts equal the manifests (ablation: one tuning line per `tune` row, one seed line per `runs` row;
  perturbation: one line per runnable row, a comment per "needs tuned config" row; timing: 14 x 2 = 28, amendment 5);
- every line parses: the tuning lines with the H2S-NN launcher's parser, the wrapped launchers with their own parsers;
- the existing phases are byte-identical: `baselines-1` and `h2snn` give exactly the lists of `v3b` 365a757's script;
- `run_row.py`: the slot's trailing `--resume` is dropped, a finished row is not run again, the working directory,
  the trials copy, the (b) tuned file (and its wait), the optional tuned file; one perturbation line (GCN, HGB
  command, toy graph) runs through the wrapper as the slot runs it, and a second leg skips it.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.card7_fixtures import write_toy_hgb
from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "ablation"))
sys.path.insert(0, str(ROOT / "scripts" / "perturbation"))

import make_graphs as mg  # noqa: E402

from scripts.tsubame import make_cell_list as mcl  # noqa: E402
from scripts.tsubame import run_row  # noqa: E402

BASE_SHA = "365a75763f45dee4b8853cc85ae06f0084295c47"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _parse(module_path: str, words: list[str]):
    """The launcher's own `parse_args` on `words` (unknown flags exit)."""
    name = Path(module_path).stem
    spec = importlib.util.spec_from_file_location(f"_t9_{name}", ROOT / module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with mock.patch.object(sys, "argv", [module_path, *words]):
        return module.parse_args()


def _row_words(line: str) -> tuple[object, str, list[str]]:
    words = shlex.split(line)
    assert words[0] == mcl.ROW_RUNNER, words[0]
    return run_row.parse(words[1:])


class AblationPhaseTests(unittest.TestCase):
    def test_T9_ablation_lines_equal_the_manifests(self) -> None:
        import seed_runs

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifests = []
            for dataset in ("acm", "dblp"):
                cfg = ROOT / "configs" / "experiments" / f"{dataset}_h2snn_v3b.yaml"
                rows = seed_runs.build_manifest(dataset, cfg, cfg, tmp / "main", tmp / "abl", None)
                path = tmp / "abl" / dataset / "manifest.tsv"
                seed_runs.write_manifest(rows, path)
                manifests.append((path, rows))
            out = Path("/OUT")
            tuning, seeds = mcl.ablation_lists([p for p, _ in manifests], out)
            tune_rows = [r for _, rows in manifests for r in rows if r["kind"] == "tune"]
            run_rows = [r for _, rows in manifests for r in rows if r["kind"] == "runs"]
            from scripts.tsubame.lines import runnable_lines

            tuning, seeds = runnable_lines("\n".join(tuning)), runnable_lines("\n".join(seeds))
            self.assertEqual((len(tuning), len(seeds)), (len(tune_rows), len(run_rows)))
            self.assertEqual((len(tuning), len(seeds)), (2 * 4, 2 * 11))
            for line, row in zip(tuning, tune_rows):
                words = shlex.split(line)
                self.assertEqual(words[0], mcl.H2SNN_LAUNCHER)
                args = _parse(words[0], words[1:] + ["--resume"])
                self.assertEqual((args.config, args.exp_name, args.n_trials, args.base_seed, args.num_runs),
                                 (Path(row["config"]), row["exp_name"], 100, 0, 1))
                self.assertEqual((Path(args.state_dir), args.wandb_project),
                                 (out / "state" / row["exp_name"], mcl.ABLATION_WANDB_PROJECT))
            for line, row in zip(seeds, run_rows):
                args, launcher, words = _row_words(line + " --resume")
                self.assertEqual(launcher, "scripts/run_h2snn_runs.py")
                self.assertNotIn("--resume", words)
                parsed = _parse(launcher, words)
                self.assertEqual((parsed.config, parsed.base_seed, parsed.num_runs, parsed.exp_name),
                                 (Path(row["config"]), int(row["base_seed"]), int(row["num_runs"]), row["exp_name"]))
                self.assertEqual(args.done, out / "experiments" / "ablation" / row["exp_name"] / "runs_summary.json")
                if row["configuration"] == "b":
                    base = [r["config"] for r in tune_rows if (r["dataset"], r["arm"]) == (row["dataset"],
                                                                                         row["arm"])][0]
                    self.assertEqual(args.tuned_config, [Path(base), out / "experiments" / "ablation" /
                                                         f"{row['dataset']}__ablation_b_{row['arm']}" /
                                                         "best_tuned_run0.json", Path(row["config"])])
                else:
                    self.assertIsNone(args.tuned_config)
            self.assertFalse([r for r in run_rows if r["arm"] == "full" and int(r["base_seed"]) < 5])


class PerturbationPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "data"
        write_toy_hgb(cls.root, "dblp")
        mg.generate(cls.root, "dblp", 2.0, levels=(0.60,), log=lambda *_: None)
        mg.write_original(cls.root, "dblp", log=lambda *_: None)
        import manifest

        baselines = Path(cls.tmp.name) / "baselines"
        trials = baselines / "hetsheaf_node_classification_dblp" / "tuning_trials_run0.json"
        trials.parent.mkdir(parents=True)
        trials.write_text("[]", encoding="utf-8")
        full = ROOT / "configs" / "experiments" / "dblp_h2snn_v3b.yaml"
        cls.out = Path(cls.tmp.name) / "pert"
        rows, missing = manifest.build(cls.root / "NC", {"dblp": full}, baselines, cls.out)
        manifest.write(rows, missing, cls.out)
        cls.rows = rows
        cls.trials = trials

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_T9_perturbation_lines_equal_the_manifest(self) -> None:
        from scripts.tsubame.lines import runnable_lines

        text = "\n".join(mcl.perturbation_list(self.out / "manifest.tsv"))
        lines = runnable_lines(text)
        runnable = [r for r in self.rows if r["command"]]
        self.assertEqual(len(lines), len(runnable))
        self.assertEqual(len(lines), 2 * 8 + 1 - 2 + 6 - 1)            # 2 graphs + S2 + 6 reference, CSNN waits
        self.assertEqual(text.count("# needs tuned config:"), len(self.rows) - len(runnable))
        self.assertIn("# needs tuned config: dblp original csnn", text)
        for line, row in zip(lines, runnable):
            args, launcher, words = _row_words(line + " --resume")
            self.assertEqual((args.done, args.cwd), (Path(row["summary"]), Path(row["cwd"])))
            self.assertTrue(launcher.endswith("scripts/perturbation/run_checked.py"))
            self.assertEqual(shlex.join(["python", launcher, *words]), row["command"].split(" && ")[-1])
            inner = words[words.index("--") + 1:]
            module = Path(inner[0]).relative_to(ROOT).as_posix()
            if module != "scripts/perturbation/run_hgb_command.py":
                _parse(module, inner[1:])                                # the inner launcher's own parser
            if row["command"].startswith("mkdir -p "):                    # a tuned baseline: its sweep's trials
                self.assertEqual(row["configuration"], "tuned")
                self.assertEqual(args.copy, [self.trials, Path(row["summary"]).parent])
            else:
                self.assertIsNone(args.copy)

    def test_T9_a_line_runs_as_the_slot_runs_it(self) -> None:
        """GCN on the toy original graph through `python run_row.py ... --resume` from H2S-NN/, then a second leg."""
        row = [r for r in self.rows if (r["graph_kind"], r["method"]) == ("original", "gcn")][0]
        row = {**row, "command": row["command"].replace("--num_runs 5", "--num_runs 1")}
        cmd = [sys.executable, *shlex.split(mcl.perturbation_line(row)), "--resume"]
        first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stdout[-2000:] + first.stderr[-4000:])
        self.assertTrue(Path(row["summary"]).exists())
        checks = [json.loads(x) for x in (Path(row["summary"]).parent / "fingerprint_checks.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.assertTrue(checks and all(c["match"] for c in checks))
        self.assertEqual(Path(checks[0]["cwd"]).resolve(), Path(row["cwd"]).resolve())
        second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0)
        self.assertIn("row complete", second.stdout)


class TimingPhaseTests(unittest.TestCase):
    def test_T9_timing_lines(self) -> None:
        from scripts.tsubame.lines import runnable_lines

        sys.path.insert(0, str(ROOT / "scripts" / "timing"))
        from time_methods import main_table_methods

        out = Path("/OUT")
        lines = runnable_lines("\n".join(mcl.timing_list(out)))
        methods = {d: main_table_methods(d) for d in mcl.TIMING_DATASETS}
        self.assertEqual(len(lines), sum(len(m) for m in methods.values()))
        self.assertEqual(len(lines), 28)                                 # amendment 5: H2S-NN + 13 baselines, x 2
        expected = [(m, d) for d in mcl.TIMING_DATASETS for m in methods[d]]
        for line, (method, dataset) in zip(lines, expected):
            args, launcher, words = _row_words(line + " --resume")
            self.assertEqual(launcher, "scripts/timing/time_methods.py")
            with mock.patch.object(sys, "argv", [launcher, *words]):
                import argparse

                parser = argparse.ArgumentParser()               # time_methods builds its parser inside main()
                for flag in ("--dataset", "--method", "--h2snn_config", "--baseline_output", "--device", "--out"):
                    parser.add_argument(flag)
                parser.add_argument("--epochs", type=int)
                parsed = parser.parse_args(words)
            self.assertEqual((parsed.method, parsed.dataset, parsed.epochs), (method, dataset, 4))
            self.assertEqual(args.done, Path(parsed.out))
            if method == "h2snn":
                name, base, best, cfg = args.optional_tuned_config
                self.assertEqual(name, "h2snn_config")
                self.assertTrue((ROOT / base).exists(), base)
                self.assertTrue(best.endswith(f"h2snn_node_classification_{dataset}/best_tuned_run0.json"))
            else:
                self.assertIsNone(args.optional_tuned_config)
        source = (ROOT / "scripts" / "timing" / "time_methods.py").read_text(encoding="utf-8")
        for flag in ("--dataset", "--method", "--h2snn_config", "--baseline_output", "--device", "--out", "--epochs"):
            self.assertIn(f'parser.add_argument("{flag}"', source)


class ExistingPhasesTests(unittest.TestCase):
    def test_T9_existing_phases_byte_identical(self) -> None:
        text = subprocess.run(["git", "show", f"{BASE_SHA}:H2S-NN/scripts/tsubame/make_cell_list.py"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "make_cell_list_base.py"
            path.write_text(text, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("make_cell_list_base", path)
            base = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(base)
            out = Path("/OUT")
            self.assertEqual(mcl.tuning_list("baselines-1", out), base.tuning_list("baselines-1", out))
            self.assertEqual(mcl.h2snn_tuning_list(out), base.h2snn_tuning_list(out))
            self.assertEqual(mcl.h2snn_final_seeds_list(out), base.h2snn_final_seeds_list(out))
            for phase in ("baselines-1", "h2snn"):
                folder = Path(tmp) / phase                       # the lists embed the folder: one folder, twice
                base.main(["--phase", phase, "--out-dir", str(folder)])
                old = {p.name: p.read_bytes() for p in folder.iterdir()}
                for p in folder.iterdir():
                    p.unlink()
                mcl.main(["--phase", phase, "--out-dir", str(folder)])
                new = {p.name: p.read_bytes() for p in folder.iterdir()}
                self.assertEqual(sorted(new), sorted(old))
                for name in old:
                    self.assertEqual(new[name], old[name], f"{phase}/{name}")


class RunRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.launcher = self.dir / "launcher.py"
        self.launcher.write_text("import json, os, sys\n"
                                 "out = sys.argv[sys.argv.index('--write') + 1]\n"
                                 "open(out, 'w').write(json.dumps({'cwd': os.getcwd(), 'argv': sys.argv[1:]}))\n",
                                 encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_T9_resume_dropped_cwd_copy_done(self) -> None:
        work = self.dir / "work"
        work.mkdir()
        src = self.dir / "trials.json"
        src.write_text("[]", encoding="utf-8")
        record = self.dir / "record.json"
        done = self.dir / "summary.json"
        argv = ["--done", str(done), "--cwd", str(work), "--copy", str(src), str(self.dir / "copied"), "--",
                str(self.launcher), "--write", str(record), "--x", "1", "--resume"]
        self.assertEqual(run_row.main(argv), 0)
        got = json.loads(record.read_text(encoding="utf-8"))
        self.assertEqual(got["argv"], ["--write", str(record), "--x", "1"])
        self.assertEqual(Path(got["cwd"]).resolve(), work.resolve())
        self.assertTrue((self.dir / "copied" / "trials.json").exists())
        record.unlink()
        done.write_text("{}", encoding="utf-8")
        self.assertEqual(run_row.main(argv), 0)
        self.assertFalse(record.exists())                         # complete: not run again

    def test_T9_tuned_config_waits_then_materialises(self) -> None:
        import yaml

        base = ROOT / "configs" / "experiments" / "acm_h2snn_v3b.yaml"
        best = self.dir / "best_tuned_run0.json"
        out = self.dir / "tuned.yaml"
        record = self.dir / "record.json"
        argv = ["--done", str(self.dir / "summary.json"), "--tuned_config", str(base), str(best), str(out), "--",
                str(self.launcher), "--write", str(record)]
        self.assertEqual(run_row.main(argv), run_row.WAITING)
        self.assertFalse(record.exists() or out.exists())
        best.write_text(json.dumps({"sweep_params": {"d": 16}}), encoding="utf-8")
        self.assertEqual(run_row.main(argv), 0)
        self.assertEqual(yaml.safe_load(out.read_text(encoding="utf-8"))["model"]["d"], 16)
        optional = ["--done", str(self.dir / "s2.json"), "--optional_tuned_config", "h2snn_config", str(base),
                    str(self.dir / "absent.json"), str(self.dir / "t2.yaml"), "--", str(self.launcher), "--write",
                    str(record)]
        self.assertEqual(run_row.main(optional), 0)
        self.assertNotIn("--h2snn_config", json.loads(record.read_text(encoding="utf-8"))["argv"])
        optional[5] = str(best)
        self.assertEqual(run_row.main(optional), 0)
        self.assertEqual(json.loads(record.read_text(encoding="utf-8"))["argv"][-2:],
                         ["--h2snn_config", str(self.dir / "t2.yaml")])


if __name__ == "__main__":
    unittest.main()
