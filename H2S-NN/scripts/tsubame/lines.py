"""Cell-list lines and the files they write (TSUBAME card T1/T4). Standard library only: the slot script calls it
before and after every line run, and status / collect read through it.

A cell list has one launcher command per line (no leading ``python``), as `scripts/slurm/run_sweep_agents.sbatch`.
Blank lines and lines starting with ``#`` are skipped in the numbering: runnable line ``i`` (1-based) is the
``i``-th non-blank, non-comment line.

The paths a line writes are the launcher's (`docs/tsubame_evidence/t0_inventory.md`):
``<state_dir>/<model>_<task>_<dataset>/tuning_state_run<u>.json`` (`src/baselines/campaign.py: state_path_for`,
`Cell.token`, `build_units`) and ``<output_dir>/<token>/...``. `tests/test_tsubame_collect.py` checks this module's
paths against `state_path_for` on every phase-1 line, so the two cannot drift.

    python scripts/tsubame/lines.py snapshot -- <line words...>     (prints one JSON object)
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def runnable_lines(text: str) -> list[str]:
    """The runnable lines of a cell list, in order (blank and ``#`` lines skipped)."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


@dataclass(frozen=True)
class Line:
    text: str
    task: str
    model: str
    dataset: str
    output_dir: Path
    state_dir: Path
    num_runs: int
    final_seeds: tuple[int, ...] | None
    n_trials: int

    @property
    def token(self) -> str:
        return f"{self.model}_{self.task}_{self.dataset}"

    @property
    def kind(self) -> str:
        return "final_seeds" if self.final_seeds is not None else "tuning"

    def state_paths(self) -> list[Path]:
        return [self.state_dir / self.token / f"tuning_state_run{u}.json" for u in range(self.num_runs)]

    @property
    def cell_dir(self) -> Path:
        return self.output_dir / self.token


def parse_line(text: str) -> Line:
    """One launcher line of a phase-1 list: one model, one dataset (the launcher's own defaults otherwise)."""
    words = shlex.split(text)
    if not words or not words[0].endswith("run_baseline_tuning.py"):
        raise ValueError(f"not a baseline tuning line: {text!r}")
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--task", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--output_dir", type=Path, default=Path("experiments/baselines"))
    ap.add_argument("--state_dir", type=Path, default=None)
    ap.add_argument("--num_runs", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--final_seeds", type=int, nargs="+", default=None)
    args, _ = ap.parse_known_args(words[1:])
    if len(args.models) != 1 or len(args.datasets) != 1 or "all" in args.models + args.datasets:
        raise ValueError(f"a cell-list line names one model and one dataset: {text!r}")
    return Line(text=text, task=args.task, model=args.models[0], dataset=args.datasets[0],
                output_dir=args.output_dir, state_dir=args.state_dir or args.output_dir / "tuning_state",
                num_runs=args.num_runs, final_seeds=None if args.final_seeds is None else tuple(args.final_seeds),
                n_trials=args.n_trials)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def snapshot(line: Line) -> dict[str, Any]:
    """The line's progress as its files show it: per unit state, attempted / records / guard; final-seed files."""
    units = []
    for path in line.state_paths():
        state = read_json(path)
        if not isinstance(state, dict):
            units.append({"state": str(path), "exists": False})
            continue
        records = state.get("records") if isinstance(state.get("records"), list) else []
        units.append({"state": str(path), "exists": True, "attempted": int(state.get("attempted_trials", 0)),
                      "records": len(records), "target": state.get("target_trials"),
                      "open_attempt": state.get("open_attempt"), "line_status": state.get("line_status")})
    seeds = sorted(int(p.stem.removeprefix("final_seed")) for p in line.cell_dir.glob("final_seed*.json")
                   if p.stem.removeprefix("final_seed").isdigit())
    return {"token": line.token, "kind": line.kind, "units": units, "final_seed_files": seeds}


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "snapshot" and argv[1] == "--":
        print(json.dumps(snapshot(parse_line(shlex.join(argv[2:])))))
        return
    raise SystemExit("usage: lines.py snapshot -- <line words...>")


if __name__ == "__main__":
    main(sys.argv[1:])
