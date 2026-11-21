"""Card 10 A2.3: every failed trial of a run directory, reported (never moved or rerun; the author decides).

    python scripts/tsubame/failures.py --out <out> --list <out>/cells_tuning.txt

Works on lab (`scripts/slurm/lab_slots.sbatch`) and TSUBAME (`scripts/tsubame/sweep_slots.sh`) outputs: both write
the same line logs and provenance. Reads only the sweep states' `failures[]` (`src/utils/wandb_campaign.py`
`record_failed_attempt`: attempt_index, return_code, error "<Type>: <message>", failure {condition, value, limit,
epoch} when a named condition fired) and the line logs, joined by the provenance (`<out>/logs/prov/*.jsonl`): the
line run that made attempt `a` of a state is the last run that started with at most `a` attempted trials.

Writes `<out>/failures.tsv` (one row per failed trial: line, cell, unit, attempt, condition, the named condition,
the error, the error's last lines in that run's line log, job, task, host, GPU, log, state) and prints a summary.
Condition: `PCGConvergenceError` (T_max), `StopConditionError` (a spec §6 stop condition), `OOM` (out of memory in
the error or its log lines), `other`. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tsubame.lines import parse_line, read_json, runnable_lines  # noqa: E402

COLUMNS = ("line_no", "cell", "unit", "attempt", "condition", "named_condition", "error", "error_tail", "job_id",
           "task_id", "host", "gpu_name", "log", "state")
TAIL = 5
OOM_MARKERS = ("out of memory", "outofmemoryerror")


def load_starts(out: Path) -> list[dict[str, Any]]:
    starts = []
    for path in sorted((out / "logs" / "prov").glob("*.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                event = json.loads(raw)
                if event.get("event") == "start":
                    starts.append(event)
    return sorted(starts, key=lambda e: e.get("start", ""))


def run_for_attempt(state: Path, attempt: int, starts: list[dict[str, Any]]) -> dict[str, Any] | None:
    chosen = None
    for event in starts:
        before = event.get("before")
        for unit in (before or {}).get("units", []) if isinstance(before, dict) else []:
            if Path(unit.get("state", "")) in (state, state.resolve()):
                if (int(unit.get("attempted", 0)) if unit.get("exists") else 0) <= attempt:
                    chosen = event
    return chosen


def classify(entry: dict[str, Any], tail: list[str]) -> tuple[str, str]:
    error = str(entry.get("error", ""))
    failure = entry.get("failure") if isinstance(entry.get("failure"), dict) else None
    named = str(failure.get("condition", "")) if failure else ""
    if error.startswith("PCGConvergenceError") or named == "T_max":
        return "PCGConvergenceError", named
    if failure or error.startswith("StopConditionError"):
        return "StopConditionError", named
    if any(m in text.lower() for text in [error, *tail] for m in OOM_MARKERS):
        return "OOM", named
    return "other", named


def error_tail(log: Path | None, error: str, occurrence: int) -> list[str]:
    """The TAIL lines of the run's log that end at the `occurrence`-th line holding the error's first line."""
    if log is None or not log.is_file():
        return []
    first = error.splitlines()[0].strip() if error.strip() else ""
    if not first:
        return []
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    # the worker's traceback ends with "<Type>: <message>", the error's first line (stored cut at 1000 characters)
    hits = [i for i, text in enumerate(lines) if first[:200] in text]
    if not hits:
        return []
    end = hits[min(occurrence, len(hits) - 1)]
    return [text.rstrip() for text in lines[max(0, end - TAIL + 1):end + 1]]


def collect_failures(out: Path, list_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    starts = load_starts(out)
    rows, skipped, seen = [], [], set()
    for number, text in enumerate(runnable_lines(list_path.read_text(encoding="utf-8")), start=1):
        try:
            line = parse_line(text)
        except ValueError:
            skipped.append(f"line {number}: not a tuning launcher line")
            continue
        for state_path in line.state_paths():
            if state_path in seen:          # a cell's tuning and final-seeds lines share one state
                continue
            seen.add(state_path)
            state = read_json(state_path)
            if not isinstance(state, dict) or not isinstance(state.get("failures"), list):
                continue
            used: Counter = Counter()
            for entry in state["failures"]:
                if not isinstance(entry, dict):
                    continue
                attempt = int(entry.get("attempt_index", -1))
                run = run_for_attempt(state_path, attempt, starts)
                log = Path(run["log"]) if run and run.get("log") else None
                error = str(entry.get("error", ""))
                key = (run.get("log") if run else None, error.splitlines()[0] if error else "")
                tail = error_tail(log, error, used[key])
                used[key] += 1
                condition, named = classify(entry, tail)
                rows.append({"line_no": number, "cell": line.token, "unit": state_path.stem.removeprefix(
                    "tuning_state_"), "attempt": attempt, "condition": condition, "named_condition": named,
                    "error": " ".join(error.split())[:500], "error_tail": " | ".join(tail),
                    "job_id": run.get("job_id", "") if run else "", "task_id": run.get("task_id", "") if run else "",
                    "host": run.get("host", "") if run else "", "gpu_name": run.get("gpu_name", "") if run else "",
                    "log": str(log or ""), "state": str(state_path)})
    return rows, skipped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True)
    args = ap.parse_args(argv)
    rows, skipped = collect_failures(args.out, args.list)
    path = args.out / "failures.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    by_condition = Counter(r["condition"] for r in rows)
    print(f"failed trials {len(rows)} -> {path}")
    for condition, n in sorted(by_condition.items()):
        print(f"  {condition}: {n}")
    for cell, n in sorted(Counter(r["cell"] for r in rows).items()):
        print(f"  {cell}: {n}")
    for note in skipped:
        print(f"  skipped {note}")
    print("Failures are reported to the author; no line is moved or rerun by this script.")


if __name__ == "__main__":
    main()
