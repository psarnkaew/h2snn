"""TSUBAME card T7.3: H2S-NN final seeds only after tuning.

- `make_cell_list.py --final-seeds` writes a final-seeds line for an `h2snn` (or baseline) cell only when its sweep
  has 100 finished trials and no `exceeds_24h`; an unfinished or stopped cell is a comment with its count;
- `--phase h2snn` writes the 24 tuning lines and no final-seeds list (nothing to submit before tuning ends).
Mutation (`docs/tsubame_evidence/t7_3_mutation.py`): the gate removed -> a line for an unfinished cell -> fails.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.tsubame import make_cell_list as M
from scripts.tsubame.lines import parse_line, runnable_lines

NC = "node_classification"


def write_state(text: str, attempted: int, **extra) -> None:
    path = parse_line(text).state_paths()[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"attempted_trials": attempted, "target_trials": 100,
                                "records": [{"i": i} for i in range(attempted)], **extra}))


class FinalSeedsGateTests(unittest.TestCase):
    def test_final_seeds_only_for_finished_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            acm, dblp, imdb = (M.h2snn_tuning_line(NC, x, out) for x in ("acm", "dblp", "imdb"))
            mag = M.h2snn_tuning_line("link_prediction", "ogbn-mag_pp", out)       # no state yet
            gcn = M.tuning_line(NC, "gcn", "acm", out)
            write_state(acm, 100)
            write_state(dblp, 40)
            write_state(imdb, 100, line_status="exceeds_24h")
            write_state(gcn, 100)
            tuning = out / "cells_tuning_h2snn.txt"
            tuning.write_text("# h2snn\n" + "\n".join([acm, dblp, imdb, mag, gcn]) + "\n")
            M.main(["--final-seeds", "--out-dir", str(out), "--tuning-list", str(tuning),
                    "--out", str(out / "cells_final_seeds_h2snn.txt")])
            text = (out / "cells_final_seeds_h2snn.txt").read_text()
        self.assertEqual(runnable_lines(text), [M.h2snn_final_seeds_line(NC, "acm", out),
                                                M.final_seeds_line(NC, "gcn", "acm", out)])
        self.assertIn("# not finished h2snn_node_classification_dblp: 40/100 finished trials", text)
        self.assertIn("# not finished h2snn_node_classification_imdb: 100/100 finished trials, exceeds_24h", text)
        self.assertIn("# not finished h2snn_link_prediction_ogbn-mag_pp: 0/100 finished trials", text)
        self.assertEqual(parse_line(runnable_lines(text)[0]).final_seeds, (0, 1, 2, 3, 4))

    def test_phase_h2snn_writes_no_final_seeds_up_front(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            M.main(["--phase", "h2snn", "--out", str(out / "cells_tuning_h2snn.txt")])
            files = sorted(p.name for p in out.iterdir())
            lines = runnable_lines((out / "cells_tuning_h2snn.txt").read_text())
        self.assertEqual(files, ["cells_tuning_h2snn.txt"])
        self.assertEqual(len(lines), 24)
        self.assertTrue(all(parse_line(x).kind == "tuning" for x in lines))


if __name__ == "__main__":
    unittest.main()
