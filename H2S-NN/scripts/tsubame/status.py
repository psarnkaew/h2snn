"""TSUBAME card T4: progress of every line of a run directory (guide section 6.3). Never calls `qstat`.

    python scripts/tsubame/status.py --out $WORK/h2snn-out/<sha> [--list cells_tuning.txt]

Per tuning line: finished trials / target (attempts whose bookkeeping completed: successful + pruned + failed),
successful records, failed attempts (the worker failed, the job lived), unfinished starts of the open attempt (the job
was killed during it; T2 guard), the line status (`exceeds_24h` / `done` / `-`), final seeds written / 5, and the
last write time of the line's log files. It reads only the files of `docs/tsubame_evidence/t0_inventory.md` section
4 (through `scripts/tsubame/lines.py`): the sweep states, the final-seed records, the log files' modification time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tsubame.lines import parse_line, read_json, runnable_lines, snapshot  # noqa: E402

FINAL_SEEDS = 5


def line_status(text: str, number: int, out: Path, list_name: str) -> dict[str, Any]:
    line = parse_line(text)
    snap = snapshot(line)
    unit = snap["units"][0] if snap["units"] else {"exists": False}
    state = read_json(Path(unit["state"])) if unit.get("exists") else None
    state = state if isinstance(state, dict) else {}
    finished = int(state.get("attempted_trials", 0))
    target = int(state.get("target_trials") or line.n_trials)
    open_attempt = state.get("open_attempt") if isinstance(state.get("open_attempt"), dict) else {}
    if state.get("line_status"):
        status = str(state["line_status"])
    elif unit.get("exists") and finished >= target and not open_attempt:
        status = "done"
    else:
        status = "-"
    logs = sorted((out / "logs" / "lines").glob(f"{list_name}.line{number:03d}.*.log"))
    last = max((p.stat().st_mtime for p in logs), default=None)
    return {"line": number, "token": line.token, "finished": finished, "target": target,
            "records": int(unit.get("records", 0)), "failed": int(state.get("failed_trials", 0)),
            "killed_starts": int(open_attempt.get("starts", 0)), "status": status,
            "final_seeds": len(snap["final_seed_files"]),
            "last_log": dt.datetime.fromtimestamp(last).isoformat(timespec="seconds") if last else "-"}


def collect_status(out: Path, list_path: Path) -> list[dict[str, Any]]:
    name = list_path.stem
    return [line_status(text, i, out, name)
            for i, text in enumerate(runnable_lines(list_path.read_text(encoding="utf-8")), start=1)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=None, help="default <out>/cells_tuning.txt")
    args = ap.parse_args(argv)
    rows = collect_status(args.out, args.list or args.out / "cells_tuning.txt")
    print(f"{'line':>4}  {'cell':<44} {'finished':>9} {'records':>7} {'failed':>6} {'killed':>6} "
          f"{'status':<11} {'seeds':>5}  last log")
    for r in rows:
        print(f"{r['line']:>4}  {r['token']:<44} {r['finished']:>4}/{r['target']:<4} {r['records']:>7} "
              f"{r['failed']:>6} {r['killed_starts']:>6} {r['status']:<11} {r['final_seeds']:>2}/{FINAL_SEEDS}  "
              f"{r['last_log']}")
    done = sum(r["status"] == "done" for r in rows)
    over = sum(r["status"] == "exceeds_24h" for r in rows)
    seeds = sum(r["final_seeds"] >= FINAL_SEEDS for r in rows)
    print(f"lines {len(rows)}: sweeps done {done}, exceeds_24h {over}, final seeds complete {seeds}")


if __name__ == "__main__":
    main()
