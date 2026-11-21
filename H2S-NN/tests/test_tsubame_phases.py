"""TSUBAME card T7.4: the three phase lists are disjoint and cover their cells.

`h2snn` (24 H2S-NN cells), `baselines-early` (HetSheaf and CSNN, ACM and DBLP NC) and `baselines-1` (the other
runnable phase-1 baseline cells): pairwise disjoint (no cell, no state file, no line in two lists: two chains
holding one line would resume one state from two processes), and together equal to the runnable cells they
cover (the registry's runnable cells on the 8 phase-1 datasets plus the 24 H2S-NN cells). Every early line passes
`--validate_only` (in-process; nothing trained).
Mutation (`docs/tsubame_evidence/t7_4_mutation.py`): no exclusion in `baselines-1` -> overlap -> this test fails.
"""

from __future__ import annotations

import contextlib
import io
import shlex
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest import mock

from scripts.tsubame import make_cell_list as M
from scripts.tsubame.lines import parse_line, runnable_lines

PHASE1 = {"node_classification": {"acm", "dblp", "imdb"},
          "link_prediction": {"acm_pa", "dblp_ap", "imdb_ma", "pubmed", "lastfm"}}


def phase_lists(out: Path) -> dict[str, list[str]]:
    return {"h2snn": runnable_lines("\n".join(M.h2snn_tuning_list(out))),
            "baselines-early": runnable_lines("\n".join(M.tuning_list("baselines-early", out))),
            "baselines-1": runnable_lines("\n".join(M.tuning_list("baselines-1", out)))}


class PhaseTests(unittest.TestCase):
    def test_phases_pairwise_disjoint_and_cover_their_cells(self) -> None:
        from src.baselines.registry import iter_cells

        with tempfile.TemporaryDirectory() as d:
            lists = phase_lists(Path(d))
        cells = {name: [(p.model, p.task, p.dataset) for p in map(parse_line, ls)] for name, ls in lists.items()}
        states = {name: {str(s) for p in map(parse_line, ls) for s in p.state_paths()} for name, ls in lists.items()}
        for a, b in combinations(lists, 2):
            with self.subTest(pair=(a, b)):
                self.assertEqual(set(cells[a]) & set(cells[b]), set())
                self.assertEqual(states[a] & states[b], set())
                self.assertEqual(set(lists[a]) & set(lists[b]), set())
        for name, cs in cells.items():
            self.assertEqual(len(cs), len(set(cs)), f"duplicate in {name}")
        from src.baselines.cell_status import cell_status

        runnable = {(r["model"], r["task"], r["dataset"]) for r in iter_cells()  # card 8 C5: `tune` cells only
                    if r["runnable"] and r["dataset"] in PHASE1.get(r["task"], set())
                    and cell_status(r["model"], r["task"], r["dataset"]) == "tune"}
        h2snn = {("h2snn", t, ds) for t, ds in M.H2SNN_CELLS}
        self.assertEqual(set().union(*map(set, cells.values())), runnable | h2snn)
        self.assertEqual(len(cells["h2snn"]), 24)
        self.assertEqual(sorted(cells["baselines-early"]),
                         sorted((m, "node_classification", ds) for ds in ("acm", "dblp") for m in ("csnn", "hetsheaf")))

    def test_early_lines_pass_validate_only(self) -> None:
        import scripts.run_baseline_tuning as launcher

        with tempfile.TemporaryDirectory() as d:
            for line in phase_lists(Path(d))["baselines-early"]:
                with self.subTest(line=line):
                    buffer = io.StringIO()
                    with mock.patch.object(sys, "argv", [*shlex.split(line), "--validate_only"]), \
                            contextlib.redirect_stdout(buffer):
                        launcher.main()
                    self.assertIn("validate_only: nothing was run.", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
