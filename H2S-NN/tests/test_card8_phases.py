"""Card 8 C5: the TSUBAME lists built from the cell statuses (`scripts/tsubame/make_cell_list.py`).

* Phases `baselines-early`, `baselines-1`, `baselines-2`, `published`, `probe` are pairwise disjoint (no cell, no line,
  no state file in two lists), and their union is every NC and LP cell whose status is not `filled` or `N/A`, each
  exactly once; no `filled` cell has a line.
* `h2snn` and `baselines-early` are byte-identical to the base commit's generator (`git show fc0ebaa`).
* A published line is parsed by `lines.py` and is complete once its summary and seed record exist; a probe line pins
  every axis to its smallest grid point (`campaign.build_sweep_config(grid_point="min")`), one trial, one epoch.

Card mutation: drop the `filled` exclusion (`make_cell_list.not_tuned`) -> filled cells get tuning lines -> the overlap
with the filled set fails `test_phases_disjoint_and_cover_every_runnable_cell_once`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

from scripts.tsubame import make_cell_list as M
from scripts.tsubame.lines import complete, parse_line, runnable_lines

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASES = ("baselines-early", "baselines-1", "baselines-2", "published", "probe")


def build(out: Path) -> dict[str, list[str]]:
    lists = {}
    for phase in PHASES:
        text = M.status_list(phase, out) if phase in M.STATUS_PHASES else M.tuning_list(phase, out)
        lists[phase] = runnable_lines("\n".join(text))
    return lists


class PhaseTests(unittest.TestCase):
    def test_phases_disjoint_and_cover_every_runnable_cell_once(self) -> None:
        from src.baselines.cell_status import cell_status
        from src.baselines.registry import iter_cells

        with tempfile.TemporaryDirectory() as d:
            lists = build(Path(d))
        cells = {k: [(p.model, p.task, p.dataset) for p in map(parse_line, v)] for k, v in lists.items()}
        states = {k: {str(s) for p in map(parse_line, v) for s in p.state_paths()} for k, v in lists.items()}
        for a, b in combinations(PHASES, 2):
            with self.subTest(pair=(a, b)):
                self.assertEqual(set(cells[a]) & set(cells[b]), set())
                self.assertEqual(set(lists[a]) & set(lists[b]), set())
                self.assertEqual(states[a] & states[b], set())
        listed = [c for v in cells.values() for c in v]
        self.assertEqual(len(listed), len(set(listed)), "a cell is listed twice")
        status = {(r["model"], r["task"], r["dataset"]): cell_status(r["model"], r["task"], r["dataset"])
                  for r in iter_cells(tasks=("node_classification", "link_prediction"))}
        filled = {c for c, s in status.items() if s == "filled"}
        self.assertEqual(set(listed) & filled, set(), "a filled cell has a line")
        self.assertEqual(set(listed), {c for c, s in status.items() if s in ("tune", "published", "probe")})
        for phase, wanted in (("published", "published"), ("probe", "probe")):
            self.assertTrue(all(status[c] == wanted for c in cells[phase]), phase)
        for phase in ("baselines-early", "baselines-1", "baselines-2"):
            self.assertTrue(all(status[c] == "tune" for c in cells[phase]), phase)
        counts = {k: len(v) for k, v in cells.items()}
        print("card8 C5 line counts:", json.dumps(counts), "node_f tasks:",
              json.dumps({k: -(-v // 4) for k, v in counts.items()}))
        self.assertEqual(counts["published"], 43)  # NC 38 + LP 5 (card 17: + the five IEEE-CIS-G H2GB cells)
        self.assertEqual(counts["probe"], 10)      # MAGNN: NC 8 + LP 2

    def test_h2snn_and_early_are_byte_identical_to_the_base(self) -> None:
        source = subprocess.run(["git", "show", "fc0ebaa605280f2d987c4848cc3109c7d3b66680:"
                                 "H2S-NN/scripts/tsubame/make_cell_list.py"], cwd=PROJECT_ROOT, check=True,
                                capture_output=True, text=True).stdout
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "base_make_cell_list.py"
            path.write_text(source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("base_make_cell_list", path)
            base = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = base
            spec.loader.exec_module(base)
            out = Path(d) / "out"
            # Card 15 (author 2026-11-22): LP acm_pa is replaced by acm_pp, the only intended difference from the base.
            self.assertEqual(M.h2snn_tuning_list(out),
                             [line.replace("acm_pa", "acm_pp") for line in base.h2snn_tuning_list(out)])
            self.assertEqual(M.tuning_list("baselines-early", out), base.tuning_list("baselines-early", out))


class LineTests(unittest.TestCase):
    def test_published_line_parses_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            line = parse_line(M.published_line("node_classification", "gcn", "mag-year", out))
            self.assertEqual((line.launcher, line.model, line.dataset, line.final_seeds),
                             ("published", "gcn", "mag-year", (0, 1, 2, 3, 4)))
            self.assertFalse(complete(line))
            (line.cell_dir).mkdir(parents=True)
            (line.cell_dir / "published_summary.json").write_text("{}")
            state = line.state_dir / line.token / "published_state.json"
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({"seeds": [0, 1, 2]}))
            self.assertFalse(complete(line))
            state.write_text(json.dumps({"seeds": [0, 1, 2, 3, 4]}))
            self.assertTrue(complete(line))

    def test_probe_line_is_one_trial_at_the_smallest_point(self) -> None:
        from src.baselines.campaign import Cell, build_sweep_config, smallest_grid_point
        from src.baselines.registry import get_adapter

        with tempfile.TemporaryDirectory() as d:
            words = M.probe_line("node_classification", "magnn", "ogbn-mag", Path(d)).split()
        for flag, value in (("--n_trials", "1"), ("--grid_point", "min")):
            self.assertEqual(words[words.index(flag) + 1], value)
        self.assertIn("--dry_run", words)
        cell = Cell("magnn", "node_classification", "ogbn-mag")
        sweep = build_sweep_config(cell, monitor="macro_f1", n_trials=1, epochs=1, grid_point="min")
        space = get_adapter("magnn").search_space("node_classification", "ogbn-mag")
        self.assertEqual(sweep["parameters"], {k: {"value": smallest_grid_point(v)} for k, v in space.items()})
        self.assertEqual(sweep["parameters"]["hidden_dim"], {"value": 32})
        self.assertEqual(sweep["parameters"]["batch_size"], {"value": 32})
        self.assertEqual(sweep["method"], "grid")
        # the launcher accepts the probe line (in-process, nothing run) and refuses --grid_point outside a probe
        import contextlib
        import io
        from unittest import mock

        import scripts.run_baseline_tuning as launcher

        buffer = io.StringIO()
        with mock.patch.object(sys, "argv", [*words, "--validate_only"]), contextlib.redirect_stdout(buffer):
            launcher.main()
        self.assertIn("validate_only: nothing was run.", buffer.getvalue())
        with mock.patch.object(sys, "argv", [w for w in words if w != "--dry_run"] + ["--validate_only"]):
            with self.assertRaises(SystemExit):
                launcher.main()


if __name__ == "__main__":
    unittest.main()
