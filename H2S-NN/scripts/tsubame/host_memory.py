"""TSUBAME card T7.5: recorded host-memory peaks of the `h2snn` lines per `node_f` task.

    python scripts/tsubame/host_memory.py [--list <out>/cells_tuning_h2snn.txt] [--slots 4]

Four slots of one `node_f` task share the node's 768 GB of host memory. Runnable line i (1-based; comment lines
skipped as in `sweep_slots.sh`) runs on task ceil(i / S). For each task this prints its lines and the sum of their
recorded host peaks. A peak is the largest one recorded for that H2S-NN cell (GiB; peak RSS of the process or sacct
MaxRSS) in the sources below. None was measured at the model of v3b 365a757 (spec 1.25 TD-1) or at the largest
point of the search space, except H-Pokec (card 2 D6.6, largest point). A run that hit a CUDA OOM before its peak
gives a lower bound (">="). "unknown" = no record. The card's threshold: reorder only if a known sum > 600 GB.

Sources (docs/): dryrun_evidence/cells.csv (card 2 D2, jobs 16795 / 16797, head e419370),
dryrun_evidence/d6_central.csv (card 2 D6 central point, jobs 16923 / 16924 / 16960 / 16961),
dryrun_evidence/d6_runs.md (D6.6 H-Pokec largest point, job 16966, sacct MaxRSS),
H2S-NN_card5_report.md (card 5 U3 Freebase NC probe, host peak RSS 13.4 GiB, OOM).
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tsubame.lines import parse_line, runnable_lines  # noqa: E402

THRESHOLD_GB = 600.0
NODE_F_SLOTS = 4
#: (task, dataset) -> (peak GiB or None, lower bound?, source)
PEAKS: dict[tuple[str, str], tuple[float | None, bool, str]] = {
    ("node_classification", "acm"): (1.68, False, "cells.csv D2 16797_1"),
    ("node_classification", "dblp"): (2.02, False, "cells.csv D2 16797_2"),
    ("node_classification", "imdb"): (1.88, False, "cells.csv D2 16797_3"),
    ("node_classification", "freebase"): (13.91, True, "cells.csv D2 16797_4 (CUDA OOM); card 5 U3 13.4 (OOM)"),
    ("node_classification", "ogbn-mag"): (9.75, False, "d6_central.csv 16923_13"),
    ("node_classification", "mag-year"): (9.89, False, "d6_central.csv 16923_14"),
    ("node_classification", "oag-cs"): (10.05, False, "d6_central.csv 16960_15"),
    ("node_classification", "oag-eng"): (6.95, False, "d6_central.csv 16960_16"),
    ("node_classification", "oag-chem"): (14.47, False, "d6_central.csv 16960_17"),
    ("node_classification", "ieee-cis-g"): (6.38, False, "d6_central.csv 16960_18"),
    ("node_classification", "h-pokec"): (39.79, False, "d6_runs.md D6.6 largest point 16966 (MaxRSS; central 15.40)"),
    ("node_classification", "pdns"): (14.32, False, "d6_central.csv 16960_19"),
    ("link_prediction", "acm_pa"): (1.87, False, "cells.csv D2 16797_5"),
    ("link_prediction", "dblp_ap"): (4.34, True, "cells.csv D2 16797_6 (CUDA OOM; step-down 3.90)"),
    ("link_prediction", "imdb_ma"): (2.52, False, "cells.csv D2 16797_7"),
    ("link_prediction", "freebase_bb"): (14.11, True, "cells.csv D2 16797_8 (CUDA OOM)"),
    ("link_prediction", "lastfm"): (2.16, True, "cells.csv D2 16797_10 (CUDA OOM; step-down 1.62)"),
    ("link_prediction", "pubmed"): (3.60, True, "cells.csv D2 16797_9 (CUDA OOM; step-down 2.47)"),
    ("link_prediction", "ogbn-mag_pp"): (None, False, "no record (D4b: no real-data run)"),
    ("link_prediction", "h-pokec_uu"): (None, False, "no record (D4b: no real-data run)"),
    ("graph_classification", "mutag"): (1.54, False, "cells.csv D2 16797_11 (central 1.52)"),
    ("graph_classification", "enzymes"): (1.53, False, "d6_central.csv 16961_22 (D2 1.48)"),
    ("graph_classification", "nci1"): (1.51, False, "cells.csv D2 16797_14 (central 1.50)"),
    ("graph_classification", "proteins"): (1.60, False, "cells.csv D2 16797_12 (central 1.57)"),
}


def tasks_of(text: str, slots: int = NODE_F_SLOTS) -> list[list[str]]:
    """The runnable lines of each array task, as `sweep_slots.sh` assigns them (G = K * S: one line per slot)."""
    lines = runnable_lines(text)
    k = math.ceil(len(lines) / slots)
    return [lines[t * slots:(t + 1) * slots] for t in range(k)]


def table(text: str, slots: int = NODE_F_SLOTS) -> list[dict]:
    rows = []
    for t, lines in enumerate(tasks_of(text, slots), start=1):
        cells = []
        for line in lines:
            parsed = parse_line(line)
            peak, lower, source = PEAKS[(parsed.task, parsed.dataset)]
            cells.append({"token": parsed.token, "peak": peak, "lower_bound": lower, "source": source})
        known = [c["peak"] for c in cells if c["peak"] is not None]
        rows.append({"task": t, "cells": cells, "known_sum_gib": round(sum(known), 2),
                     "unknown": sum(c["peak"] is None for c in cells),
                     "known_sum_gb": round(sum(known) * 1.073741824, 2)})
    return rows


def render(rows: list[dict]) -> str:
    out = ["| node_f task | lines (host peak GiB) | known sum GiB (GB) | unknown |", "|---|---|---|---|"]
    for r in rows:
        cells = "; ".join(f"{c['token']} {'unknown' if c['peak'] is None else ('>= ' if c['lower_bound'] else '') + str(c['peak'])}"
                          for c in r["cells"])
        out.append(f"| {r['task']} | {cells} | {r['known_sum_gib']} ({r['known_sum_gb']}) | {r['unknown']} |")
    worst = max(r["known_sum_gb"] for r in rows)
    out.append(f"\nLargest known sum: {worst} GB; threshold {THRESHOLD_GB:g} GB -> "
               + ("REORDER NEEDED" if worst > THRESHOLD_GB else "no reorder (card T7.5)"))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    from scripts.tsubame.make_cell_list import h2snn_tuning_list

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=None, help="default: the phase h2snn list as make_cell_list writes it")
    ap.add_argument("--slots", type=int, default=NODE_F_SLOTS)
    args = ap.parse_args(argv)
    text = (args.list.read_text(encoding="utf-8") if args.list
            else "\n".join(h2snn_tuning_list(Path(tempfile.gettempdir()) / "out")))
    print(render(table(text, args.slots)))


if __name__ == "__main__":
    main()
