"""TSUBAME card T3: the phase-1 cell list.

The set of its lines equals the registry's runnable cells (`iter_cells`) restricted to HGB NC acm / dblp / imdb and
LP acm_pa / dblp_ap / imdb_ma / pubmed / lastfm, less the `baselines-early` cells (card T7.4); no line is duplicated; every unsupported cell is a comment; every
line passes `run_baseline_tuning.py ... --validate_only` (called in-process; nothing is trained).
Mutation (`docs/tsubame_evidence/t3_mutation.py`): the runnable filter dropped -> this test fails.
"""

from __future__ import annotations

import contextlib
import io
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# the phase-1 datasets, written here independently of make_cell_list.PHASES
PHASE1 = {"node_classification": {"acm", "dblp", "imdb"},
          "link_prediction": {"acm_pa", "dblp_ap", "imdb_ma", "pubmed", "lastfm"}}


# card T7.4: the HetSheaf / CSNN ACM and DBLP NC cells run in phase baselines-early, not in baselines-1
EARLY = {("node_classification", d, m) for d in ("acm", "dblp") for m in ("hetsheaf", "csnn")}


def expected_cells() -> set[tuple[str, str, str]]:
    from src.baselines.registry import iter_cells

    return {(r["task"], r["dataset"], r["model"]) for r in iter_cells()
            if r["runnable"] and r["dataset"] in PHASE1.get(r["task"], set())} - EARLY


class CellListTests(unittest.TestCase):
    def test_phase1_lines_are_the_registry_runnable_cells(self) -> None:
        from scripts.tsubame import make_cell_list
        from scripts.tsubame.lines import parse_line, runnable_lines

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            make_cell_list.main(["--phase", "baselines-1", "--out", str(out / "cells_tuning.txt")])
            text = (out / "cells_tuning.txt").read_text(encoding="utf-8")
        lines = runnable_lines(text)
        cells = [(p.task, p.dataset, p.model) for p in map(parse_line, lines)]
        self.assertEqual(len(cells), len(set(cells)), "duplicate line")
        self.assertEqual(len(lines), len(set(lines)))
        self.assertEqual(set(cells), expected_cells())
        self.assertEqual(cells, sorted(cells))  # sorted by (task, dataset, method)
        for line in lines:  # card 2 D5 form + TSUBAME output / state / project
            words = shlex.split(line)
            self.assertEqual(words[0], "scripts/run_baseline_tuning.py")
            for flag, value in (("--num_runs", "1"), ("--base_seed", "0"), ("--n_trials", "100"),
                                ("--wandb_project", "h2snn_baseline_t4"),
                                ("--output_dir", f"{out.resolve()}/experiments/baselines"),
                                ("--state_dir", f"{out.resolve()}/state")):
                self.assertEqual(words[words.index(flag) + 1], value, (line, flag))
            self.assertNotIn("--resume", words)  # added by sweep_slots.sh
        # every unsupported phase-1 cell is a comment with its reason
        from src.baselines.registry import iter_cells

        unsupported = [r for r in iter_cells() if not r["runnable"] and r["dataset"] in PHASE1.get(r["task"], set())]
        for row in unsupported:
            self.assertIn(f"# unsupported {row['task']} {row['dataset']} {row['model']}: {row['reason']}", text)

    def test_every_line_passes_validate_only(self) -> None:
        import scripts.run_baseline_tuning as launcher
        from scripts.tsubame import make_cell_list
        from scripts.tsubame.lines import runnable_lines

        with tempfile.TemporaryDirectory() as d:
            lines = runnable_lines("\n".join(make_cell_list.tuning_list("baselines-1", Path(d))))
            self.assertTrue(lines)
            for line in lines:
                with self.subTest(line=line):
                    buffer = io.StringIO()
                    with mock.patch.object(sys, "argv", [*shlex.split(line), "--validate_only"]), \
                            contextlib.redirect_stdout(buffer):
                        launcher.main()
                    self.assertIn("validate_only: nothing was run.", buffer.getvalue())
                    self.assertIn("1 cell(s) x 1 unit(s) x 100 trials", buffer.getvalue())
            self.assertFalse(any(Path(d).iterdir()), "validate_only wrote a file")


if __name__ == "__main__":
    unittest.main()
