"""TSUBAME card T4: collect a run directory's records with their provenance (guide section 6.5).

    python scripts/tsubame/collect.py --out $WORK/h2snn-out/<sha>

Writes into `<out>/collected/`:
- `results.csv`: one row per trial record (the sweep states' `records`) and per final-seed record
  (`final_seed<s>.json`). Values are copied from the records unchanged; no metric is computed.
- `cells/<token>.tar`: one archive per cell (its state directory, its output directory, its line logs).
- `SHA256SUMS` of `results.csv` and every archive.

Provenance (v3b SHA, host, job id, task id, GPU name, wall time) comes from the slot script's events
(`<out>/logs/prov/*.jsonl`, `scripts/tsubame/sweep_slots.sh`): a line run's start event holds the line's state
before it, so the records `[records before run r, records before run r+1)` of a state were written by run `r`.
The records carry no timing of their own, so `wall_time_s` is the wall time of that line run (empty when its job
was killed before the end event).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tsubame.lines import read_json  # noqa: E402

COLUMNS = ("task", "dataset", "method", "kind", "trial_or_seed", "unit", "record_index", "sweep_params",
           "monitor", "val_monitor", "test_metrics", "wandb_run_id", "v3b_sha", "host", "job_id", "task_id",
           "gpu_name", "wall_time_s", "source")
STATE_NAME = re.compile(r"^tuning_state_(?P<unit>run\d+)(_continuation\d+)?\.json$")


def trial_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    """The successful trial records of one sweep state, in the order they were written."""
    records = state.get("records")
    return [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []


def load_events(out: Path) -> tuple[list[dict[str, Any]], dict[tuple, dict[str, Any]]]:
    starts: list[dict[str, Any]] = []
    ends: dict[tuple, dict[str, Any]] = {}
    for path in sorted((out / "logs" / "prov").glob("*.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            event = json.loads(raw)
            key = (event.get("job_id"), event.get("task_id"), event.get("slot"), event.get("start"))
            if event.get("event") == "start":
                starts.append(event)
            elif event.get("event") == "end":
                ends[key] = event
    starts.sort(key=lambda e: e.get("start", ""))
    return starts, ends


def provenance(start: dict[str, Any] | None, ends: dict[tuple, dict[str, Any]]) -> dict[str, Any]:
    if start is None:
        return {"v3b_sha": "", "host": "", "job_id": "", "task_id": "", "gpu_name": "", "wall_time_s": ""}
    end = ends.get((start.get("job_id"), start.get("task_id"), start.get("slot"), start.get("start")))
    return {"v3b_sha": start.get("sha", ""), "host": start.get("host", ""), "job_id": start.get("job_id", ""),
            "task_id": start.get("task_id", ""), "gpu_name": start.get("gpu_name", ""),
            "wall_time_s": end.get("wall_s", "") if end else ""}


def runs_of_state(state_path: Path, starts: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """(records before the run, start event) of every line run that read this state, by start time."""
    runs = []
    for event in starts:
        before = event.get("before")
        for unit in (before or {}).get("units", []) if isinstance(before, dict) else []:
            if Path(unit.get("state", "")) == state_path:
                runs.append((int(unit.get("records", 0)) if unit.get("exists") else 0, event))
    return runs


def run_for_record(index: int, runs: list[tuple[int, dict[str, Any]]]) -> dict[str, Any] | None:
    """The run that wrote record `index`: the last run that started with at most `index` records."""
    chosen = None
    for before, event in runs:
        if before <= index:
            chosen = event
    return chosen


def row(record: dict[str, Any], *, kind: str, trial_or_seed: Any, unit: str, index: int, monitor: str,
        prov: dict[str, Any], source: Path) -> dict[str, Any]:
    key = f"val_{monitor.removeprefix('val_')}"
    return {"task": record.get("task", ""), "dataset": record.get("dataset", ""), "method": record.get("model", ""),
            "kind": kind, "trial_or_seed": trial_or_seed, "unit": unit, "record_index": index,
            "sweep_params": json.dumps(record.get("sweep_params", {}), sort_keys=True),
            "monitor": key, "val_monitor": record.get(key, ""),
            "test_metrics": json.dumps({k: v for k, v in sorted(record.items()) if k.startswith("test_")}),
            "wandb_run_id": record.get("wandb_run_id", ""), **prov, "source": str(source)}


def iter_rows(out: Path) -> Iterator[dict[str, Any]]:
    starts, ends = load_events(out)
    for state_path in sorted((out / "state").glob("*/tuning_state_*.json")):
        match = STATE_NAME.match(state_path.name)
        if not match:  # worker context / status files
            continue
        state = read_json(state_path)
        if not isinstance(state, dict):
            continue
        runs = runs_of_state(state_path.resolve(), starts) or runs_of_state(state_path, starts)
        monitor = str(state.get("monitor", ""))
        for index, record in enumerate(trial_records(state)):
            yield row(record, kind="trial", trial_or_seed=record.get("trial_name", ""), unit=match["unit"],
                      index=index, monitor=monitor, prov=provenance(run_for_record(index, runs), ends),
                      source=state_path)
    for path in sorted((out / "experiments" / "baselines").glob("*/final_seed*.json")):
        seed = path.stem.removeprefix("final_seed")
        if not seed.isdigit():
            continue
        record = read_json(path)
        if not isinstance(record, dict):
            continue
        token = path.parent.name
        # the last final-seeds run of this cell that ended with the seed written, else its last run
        candidates = [e for e in starts if "--final_seeds" in e.get("line", "") and isinstance(e.get("before"), dict)
                      and e["before"].get("token") == token]
        done = [e for e in candidates
                if int(seed) in ((ends.get((e.get("job_id"), e.get("task_id"), e.get("slot"), e.get("start")))
                                  or {}).get("after") or {}).get("final_seed_files", [])]
        start = (done or candidates or [None])[-1]
        yield row(record, kind="final_seed", trial_or_seed=int(seed), unit=f"final_seed{seed}", index=int(seed),
                  monitor=str(record.get("monitor", "")), prov=provenance(start, ends), source=path)


def write_tars(out: Path, target: Path) -> list[Path]:
    starts, _ = load_events(out)
    tokens = sorted({p.name for p in (out / "state").glob("*") if p.is_dir()}
                    | {p.name for p in (out / "experiments" / "baselines").glob("*") if p.is_dir()})
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for token in tokens:
        tar_path = target / f"{token}.tar"
        logs = sorted({e["log"] for e in starts if isinstance(e.get("before"), dict)
                       and e["before"].get("token") == token and e.get("log")})
        with tarfile.open(tar_path, "w") as tar:
            for directory in (out / "state" / token, out / "experiments" / "baselines" / token):
                if directory.is_dir():
                    tar.add(directory, arcname=str(directory.relative_to(out)))
            for log in logs:
                path = Path(log)
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(out)) if path.is_relative_to(out) else path.name)
        written.append(tar_path)
    return written


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(out: Path) -> dict[str, Any]:
    target = out / "collected"
    target.mkdir(parents=True, exist_ok=True)
    rows = list(iter_rows(out))
    csv_path = target / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tars = write_tars(out, target / "cells")
    sums = [f"{sha256(p)}  {p.relative_to(target)}" for p in [csv_path, *tars]]
    (target / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    missing = sum(1 for r in rows if not r["job_id"])
    return {"rows": len(rows), "trials": sum(r["kind"] == "trial" for r in rows),
            "final_seeds": sum(r["kind"] == "final_seed" for r in rows), "cells": len(tars),
            "rows_without_provenance": missing}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    summary = collect(args.out.resolve())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
