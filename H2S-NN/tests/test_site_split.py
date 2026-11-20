"""Card 10 L3: `scripts/slurm/site_split.py` splits a `make_cell_list.py` list between the lab and TSUBAME.

The two outputs are disjoint and together exactly the list's runnable lines, in order; a line goes to the lab only
if its cell `fits` on the chosen GPU type (rows of other GPU types are ignored; oom / error / no row -> TSUBAME);
a cell with state at one site stays there; lab state without `fits`, or state at both sites, is refused with
nothing written. Lines are the real list builders' (`make_cell_list.py`, baselines and H2S-NN, tuning and final
seeds); the fit check is joined by cell, so lab-written and TSUBAME-written lists split the same way.
Mutation (`docs/card10_evidence/mutation_l3.py`): a lab line is also written to the TSUBAME list -> fails.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.slurm import site_split  # noqa: E402
from scripts.tsubame.lines import parse_line, runnable_lines  # noqa: E402
from scripts.tsubame.make_cell_list import (final_seeds_line, h2snn_final_seeds_line,  # noqa: E402
                                            h2snn_tuning_line, tuning_line)

NC = "node_classification"
LP = "link_prediction"


class SiteSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.lab_out, self.t4_out = self.root / "lab", self.root / "t4"
        self.lab_out.mkdir()
        self.t4_out.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_list(self, out: Path) -> list[str]:
        return [tuning_line(NC, "gcn", "acm", out),            # fits                  -> lab
                tuning_line(NC, "gat", "acm", out),            # oom                   -> t4
                h2snn_tuning_line(NC, "acm", out),             # fits                  -> lab
                h2snn_tuning_line(LP, "amazon", out),          # error                 -> t4
                tuning_line(NC, "hgt", "dblp", out),           # no row on 3090        -> t4
                final_seeds_line(NC, "rgcn", "imdb", out),     # fits, state at t4     -> t4
                h2snn_final_seeds_line(NC, "dblp", out),       # fits, state at lab    -> lab
                tuning_line(NC, "han", "imdb", out)]           # fits, lab state       -> lab

    def write(self, out: Path, lines: list[str], rows: list[tuple[str, str, str]]) -> tuple[Path, Path]:
        path = self.root / "cells_tuning.txt"
        text = ["# phase list", ""]
        for i, line in enumerate(lines):
            text += [line, "# comment"] if i == 2 else [line]
        path.write_text("\n".join(text) + "\n")
        fit = self.root / "fit_check.tsv"
        body = ["\t".join(site_split.FIT_COLUMNS)]
        body += [f"{i}\t{cell}\t{gpu}\t1000\t{status}\tx" for i, (cell, gpu, status) in enumerate(rows, start=1)]
        fit.write_text("\n".join(body) + "\n")
        return path, fit

    def rows(self) -> list[tuple[str, str, str]]:
        return [("gcn_node_classification_acm", "3090", "fits"), ("gat_node_classification_acm", "3090", "oom"),
                ("gat_node_classification_acm", "a6000", "fits"),                  # other GPU type: ignored
                ("h2snn_node_classification_acm", "3090", "fits"),
                ("h2snn_link_prediction_amazon", "3090", "error"),
                ("hgt_node_classification_dblp", "a6000", "fits"),
                ("rgcn_node_classification_imdb", "3090", "fits"), ("h2snn_node_classification_dblp", "3090", "fits"),
                ("han_node_classification_imdb", "3090", "fits")]

    def run_split(self, list_path: Path, fit: Path) -> tuple[int, str]:
        err = io.StringIO()
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                site_split.main(["--list", str(list_path), "--fit-check", str(fit), "--gpu-type", "3090",
                                 "--lab-out", str(self.lab_out), "--t4-out", str(self.t4_out)])
        except SystemExit as stop:
            return int(stop.code) if isinstance(stop.code, int) else 1, err.getvalue() + str(stop.code)
        return 0, err.getvalue()

    def seed_state(self) -> None:
        (self.t4_out / "experiments" / "baselines" / "rgcn_node_classification_imdb").mkdir(parents=True)
        (self.t4_out / "experiments" / "baselines" / "rgcn_node_classification_imdb" / "final_seed0.json").write_text("{}")
        (self.lab_out / "state" / "h2snn_node_classification_dblp").mkdir(parents=True)
        (self.lab_out / "state" / "h2snn_node_classification_dblp" / "tuning_state_run0.json").write_text("{}")
        (self.lab_out / "state" / "han_node_classification_imdb").mkdir(parents=True)
        (self.lab_out / "state" / "han_node_classification_imdb" / "tuning_state_run0.json").write_text("{}")

    def test_split_disjoint_union_and_rules(self) -> None:
        self.seed_state()
        for site_out in (self.lab_out, self.t4_out):            # a list written with either site's paths
            with self.subTest(list_paths=site_out.name):
                lines = self.make_list(site_out)
                list_path, fit = self.write(site_out, lines, self.rows())
                code, err = self.run_split(list_path, fit)
                self.assertEqual(code, 0, err)
                lab = runnable_lines((self.root / "cells_tuning.lab.txt").read_text())
                t4 = runnable_lines((self.root / "cells_tuning.t4.txt").read_text())
                self.assertFalse(set(lab) & set(t4))             # disjoint
                self.assertEqual(sorted(lab + t4), sorted(lines))  # union = input
                self.assertEqual(len(lab) + len(t4), len(lines))
                self.assertEqual(lab, [lines[i] for i in (0, 2, 6, 7)])  # list order kept
                self.assertEqual(t4, [lines[i] for i in (1, 3, 4, 5)])
                self.assertEqual(runnable_lines(list_path.read_text()), lines)  # input unchanged

    def test_refusals_write_nothing(self) -> None:
        self.seed_state()
        lines = self.make_list(self.lab_out)
        # lab state for a cell that does not fit
        rows = [r if r[0] != "han_node_classification_imdb" else (r[0], "3090", "oom") for r in self.rows()]
        list_path, fit = self.write(self.lab_out, lines, rows)
        code, err = self.run_split(list_path, fit)
        self.assertEqual(code, 3)
        self.assertIn("han_node_classification_imdb has lab state but its fit check is oom", err)
        # state at both sites
        (self.t4_out / "state" / "han_node_classification_imdb").mkdir(parents=True)
        (self.t4_out / "state" / "han_node_classification_imdb" / "tuning_state_run0.json").write_text("{}")
        list_path, fit = self.write(self.lab_out, lines, self.rows())
        code, err = self.run_split(list_path, fit)
        self.assertEqual(code, 3)
        self.assertIn("state at both sites", err)
        # two statuses for one cell on one GPU type
        (self.t4_out / "state" / "han_node_classification_imdb" / "tuning_state_run0.json").unlink()
        list_path, fit = self.write(self.lab_out, lines, self.rows() + [("gcn_node_classification_acm", "3090", "oom")])
        code, err = self.run_split(list_path, fit)
        self.assertEqual(code, 3)
        self.assertIn("two statuses", err)
        self.assertFalse((self.root / "cells_tuning.lab.txt").exists())
        self.assertFalse((self.root / "cells_tuning.t4.txt").exists())
        # a missing site directory is refused (missing state is never taken for no state)
        self.t4_out.rename(self.root / "gone")
        code, err = self.run_split(list_path, fit)
        self.assertNotEqual(code, 0)
        self.assertIn("is not a directory", err)

    def test_tokens_match_lines_py(self) -> None:
        for line in self.make_list(self.lab_out):
            self.assertIn(parse_line(line).token, {r[0] for r in self.rows()})


if __name__ == "__main__":
    unittest.main()
