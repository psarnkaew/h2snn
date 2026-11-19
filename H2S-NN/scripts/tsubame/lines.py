"""Cell-list lines and the files they write (TSUBAME card T1/T4). Standard library only: the slot script calls it
before and after every line run, and status / collect read through it.

A cell list has one launcher command per line (no leading ``python``), as `scripts/slurm/run_sweep_agents.sbatch`.
Blank lines and lines starting with ``#`` are skipped in the numbering: runnable line ``i`` (1-based) is the
``i``-th non-blank, non-comment line.

The paths a line writes are the launcher's (`docs/tsubame_evidence/t0_inventory.md`):
``<state_dir>/<model>_<task>_<dataset>/tuning_state_run<u>.json`` (`src/baselines/campaign.py: state_path_for`,
`Cell.token`, `build_units`) and ``<output_dir>/<token>/...``. `tests/test_tsubame_collect.py` checks this module's
paths against `state_path_for` on every phase-1 line, so the two cannot drift.
H2S-NN lines (`scripts/run_h2snn_wandb_tuning.py`, card T7.1) follow that launcher's main(): the state is
``<state_dir or output_dir/exp_name>/tuning_state_run<u>.json`` and the final seeds
``<output_dir>/<exp_name>/final_seed<s>.json``; `tests/test_tsubame_h2snn_lines.py` checks them against the
launcher called in-process on all 48 `h2snn` lines.

    python scripts/tsubame/lines.py snapshot -- <line words...>     (prints one JSON object)
    python scripts/tsubame/lines.py complete -- <line words...>     (exit 0: a final-seeds line with every seed file)
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


TASKS = ("node_classification", "link_prediction", "graph_classification")
BASELINE_LAUNCHER = "run_baseline_tuning.py"
H2SNN_LAUNCHER = "run_h2snn_wandb_tuning.py"


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
    launcher: str = "baseline"          # "baseline" | "h2snn"
    exp_name: str = ""                  # h2snn lines: the launcher's base_exp_name
    run_start: int = 0                  # h2snn lines: --run_start
    run_count: int | None = None        # h2snn lines: --run_count

    @property
    def token(self) -> str:
        if self.launcher == "h2snn":
            return self.exp_name
        return f"{self.model}_{self.task}_{self.dataset}"

    @property
    def kind(self) -> str:
        return "final_seeds" if self.final_seeds is not None else "tuning"

    def state_paths(self) -> list[Path]:
        if self.launcher == "h2snn":
            # scripts/run_h2snn_wandb_tuning.py main(): state_dir / f"tuning_state_run{run_index}.json" for
            # run_index in range(run_start, run_start + run_count); run_count defaults to num_runs
            count = self.num_runs if self.run_count is None else self.run_count
            return [self.state_dir / f"tuning_state_run{u}.json" for u in range(self.run_start, self.run_start + count)]
        return [self.state_dir / self.token / f"tuning_state_run{u}.json" for u in range(self.num_runs)]

    @property
    def cell_dir(self) -> Path:
        # baselines: <output_dir>/<Cell.token>; h2snn: summary_dir = base_cfg.output_dir / base_exp_name
        # (run_final_seeds writes summary_dir / f"final_seed{seed}.json")
        return self.output_dir / self.token


def _h2snn_task_dataset(exp_name: str, config: str) -> tuple[str, str]:
    """`h2snn_<task>_<dataset>` (make_cell_list.h2snn_token); else the dataset of `<dataset>_h2snn_v3b.yaml`."""
    for task in TASKS:
        prefix = f"h2snn_{task}_"
        if exp_name.startswith(prefix):
            return task, exp_name[len(prefix):]
    return "", Path(config).name.split("_h2snn")[0]


def _parse_h2snn(text: str, words: list[str]) -> Line:
    """A `scripts/run_h2snn_wandb_tuning.py` line. Paths as the launcher's main(): `base_exp_name` = `--exp_name`,
    `output_dir` = `--output_dir` (both required here: the launcher's fallback is the YAML config, which this
    stdlib-only module does not read), `state_dir` = `--state_dir` or `output_dir / exp_name`,
    `num_runs` = `--num_runs` or 1, `--n_trials` default 100."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_name", default=None)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--state_dir", type=Path, default=None)
    ap.add_argument("--num_runs", type=int, default=None)
    ap.add_argument("--run_start", type=int, default=0)
    ap.add_argument("--run_count", type=int, default=None)
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--final_seeds", type=int, nargs="+", default=None)
    args, _ = ap.parse_known_args(words[1:])
    if args.exp_name is None or args.output_dir is None:
        raise ValueError(f"an h2snn cell-list line names --exp_name and --output_dir: {text!r}")
    task, dataset = _h2snn_task_dataset(args.exp_name, args.config)
    return Line(text=text, task=task, model="h2snn", dataset=dataset, output_dir=args.output_dir,
                state_dir=args.state_dir or args.output_dir / args.exp_name, num_runs=args.num_runs or 1,
                final_seeds=None if args.final_seeds is None else tuple(args.final_seeds), n_trials=args.n_trials,
                launcher="h2snn", exp_name=args.exp_name, run_start=args.run_start, run_count=args.run_count)


def parse_line(text: str) -> Line:
    """One launcher line: a baseline line (one model, one dataset; the launcher's own defaults otherwise) or an
    H2S-NN line (`scripts/run_h2snn_wandb_tuning.py`, TSUBAME card T7.1)."""
    words = shlex.split(text)
    if words and words[0].endswith(H2SNN_LAUNCHER):
        return _parse_h2snn(text, words)
    if not words or not words[0].endswith(BASELINE_LAUNCHER):
        raise ValueError(f"not a baseline or H2S-NN tuning line: {text!r}")
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


def complete(line: Line) -> bool:
    """A `--final_seeds` line whose seed files all exist (the launcher has no resume for final seeds)."""
    if line.final_seeds is None:
        return False
    return set(line.final_seeds) <= set(snapshot(line)["final_seed_files"])


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] in ("snapshot", "complete") and argv[1] == "--":
        line = parse_line(shlex.join(argv[2:]))
        if argv[0] == "snapshot":
            print(json.dumps(snapshot(line)))
            return
        raise SystemExit(0 if complete(line) else 1)
    raise SystemExit("usage: lines.py {snapshot,complete} -- <line words...>")


if __name__ == "__main__":
    main(sys.argv[1:])
