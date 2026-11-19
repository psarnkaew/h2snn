"""TSUBAME card T7.5: host memory per `node_f` task of the `h2snn` list.

Every H2S-NN cell has a row in the peaks table (a value or an explicit "unknown"); the grouping of lines into
tasks is the one `sweep_slots.sh` uses (line i on task ceil(i / 4), comment lines skipped); no known sum exceeds
600 GB, so the list keeps card 5's order (no reorder, nothing else changed).
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from scripts.tsubame import host_memory as H
from scripts.tsubame import make_cell_list as M
from scripts.tsubame.lines import parse_line, runnable_lines


class HostMemoryTests(unittest.TestCase):
    def test_table_covers_every_cell_and_no_known_sum_exceeds_600_gb(self) -> None:
        self.assertEqual(set(H.PEAKS), set(M.H2SNN_CELLS))
        with tempfile.TemporaryDirectory() as d:
            text = "\n".join(M.h2snn_tuning_list(Path(d)))
        lines = runnable_lines(text)
        tasks = H.tasks_of(text)
        self.assertEqual(len(tasks), 6)                                   # 24 lines / 4 slots
        for i, line in enumerate(lines, start=1):                          # sweep_slots.sh: slot g = i - 1
            self.assertIn(line, tasks[math.ceil(i / 4) - 1])
        self.assertEqual(sum(map(len, tasks)), 24)
        rows = H.table(text)
        self.assertTrue(all(r["known_sum_gb"] <= H.THRESHOLD_GB for r in rows))
        self.assertEqual([parse_line(x).token for x in lines],
                         [M.h2snn_token(t, ds) for t, ds in M.H2SNN_CELLS])  # card 5's order kept
        self.assertIn("no reorder", H.render(rows))


if __name__ == "__main__":
    unittest.main()
