"""TSUBAME card T7.1: `scripts/tsubame/lines.py` on H2S-NN lines, checked against the H2S-NN launcher itself.

On all 48 `h2snn` lines (24 tuning, 24 final-seeds; `make_cell_list.h2snn_tuning_line` / `h2snn_final_seeds_line`):
- tuning: `run_h2snn_wandb_tuning.py <line> --validate_only` (in-process; no W&B, nothing trained) prints the state
  path it would use; it equals `parse_line(line).state_paths()`, with and without `--resume`;
- final seeds: the launcher's `run_final_seeds` (train_trial, write_run_record and write_summary_files replaced by
  recorders; nothing trained) reads `best_tuned_run0.json` from `cell_dir` and writes exactly
  `cell_dir/final_seed<s>.json` for s = 0..4; `complete()` is false before those files exist and true after.
Baseline lines keep their behaviour (`tests/test_tsubame_collect.py`, unchanged).
"""

from __future__ import annotations

import contextlib
import io
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.tsubame import lines as L
from scripts.tsubame import make_cell_list as M


def h2snn_lines(out: Path) -> tuple[list[str], list[str]]:
    tuning = [M.h2snn_tuning_line(task, dataset, out) for task, dataset in M.H2SNN_CELLS]
    finals = [M.h2snn_final_seeds_line(task, dataset, out) for task, dataset in M.H2SNN_CELLS]
    return tuning, finals


def printed_json(text: str) -> list[dict]:
    """The JSON objects `validate_tuning_setup` prints (indent=2, one per run index)."""
    decoder, objects, i = json.JSONDecoder(), [], 0
    while (start := text.find("{\n", i)) >= 0:
        obj, end = decoder.raw_decode(text, start)
        objects.append(obj)
        i = end
    return objects


class H2SNNLineTests(unittest.TestCase):
    def test_48_lines_parse(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tuning, finals = h2snn_lines(Path(d))
        self.assertEqual((len(tuning), len(finals)), (24, 24))
        for text in tuning + finals:
            line = L.parse_line(text)
            self.assertEqual((line.launcher, line.model), ("h2snn", "h2snn"))
            self.assertEqual(line.token, f"h2snn_{line.task}_{line.dataset}")
            self.assertIn((line.task, line.dataset), M.H2SNN_CELLS)
        self.assertEqual(len({L.parse_line(t).token for t in tuning}), 24)

    def test_tuning_state_paths_are_the_launchers(self) -> None:
        import scripts.run_h2snn_wandb_tuning as launcher

        with tempfile.TemporaryDirectory() as d:
            tuning, _ = h2snn_lines(Path(d))
            for text in tuning:
                for extra in ([], ["--resume"]):
                    with self.subTest(line=L.parse_line(text).token, extra=extra):
                        buffer = io.StringIO()
                        with mock.patch.object(sys, "argv", [*shlex.split(text), *extra, "--validate_only"]), \
                                contextlib.redirect_stdout(buffer):
                            launcher.main()
                        printed = [Path(o["state_path"]) for o in printed_json(buffer.getvalue())]
                        self.assertEqual(printed, L.parse_line(text).state_paths())
            self.assertFalse(any(Path(d).rglob("*.json")), "validate_only wrote a state")

    def test_final_seed_files_are_the_launchers(self) -> None:
        import scripts.run_h2snn_wandb_tuning as launcher

        with tempfile.TemporaryDirectory() as d:
            _, finals = h2snn_lines(Path(d))
            for text in finals:
                line = L.parse_line(text)
                with self.subTest(line=line.token):
                    line.cell_dir.mkdir(parents=True, exist_ok=True)
                    (line.cell_dir / "best_tuned_run0.json").write_text(json.dumps({"sweep_params": {}}))
                    self.assertFalse(L.complete(line))
                    written: list[Path] = []

                    def record(path, rec):
                        written.append(Path(path))
                        Path(path).write_text("{}")

                    with mock.patch.object(sys, "argv", shlex.split(text)), \
                            mock.patch.object(launcher, "train_trial", lambda *a, **k: {"seed": k["seed"]}), \
                            mock.patch.object(launcher, "write_run_record", record), \
                            mock.patch.object(launcher, "write_summary_files", lambda *a, **k: None):
                        launcher.main()
                    self.assertEqual(written, [line.cell_dir / f"final_seed{s}.json" for s in line.final_seeds])
                    self.assertEqual(line.final_seeds, (0, 1, 2, 3, 4))
                    self.assertTrue(L.complete(line))
                    written[-1].unlink()
                    self.assertFalse(L.complete(line))  # every seed's file is needed

    def test_baseline_lines_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            line = L.parse_line(M.tuning_line("node_classification", "gcn", "acm", out))
        self.assertEqual((line.launcher, line.token), ("baseline", "gcn_node_classification_acm"))
        self.assertEqual(line.state_paths(), [out / "state" / "gcn_node_classification_acm" / "tuning_state_run0.json"])


if __name__ == "__main__":
    unittest.main()
