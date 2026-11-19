"""Card 7 V8: one manifest row as one cell-list line (phases `ablation`, `perturbation`, `timing` of
`make_cell_list.py`). The slot script runs every line as `python <line> --resume` from `H2S-NN/`
(`sweep_slots.sh`, unchanged); the row launchers (`run_h2snn_runs.py`, `run_baseline_published.py`,
`perturbation/run_checked.py`, `perturbation/run_hgb_command.py`, `timing/time_methods.py`) have no `--resume`, the
perturbation baselines must start in their graph's folder (the `./data` route), and a tuned-baseline row first copies
its sweep's trials file. This wrapper does exactly that, and nothing else:

1. drops the `--resume` the slot script appends (the last word only);
2. `--done FILE`: the row's summary file; if it exists the row is complete and exits 0 at once (a later leg does not
   re-run a finished row; an unfinished row runs again from the start);
3. `--tuned_config BASE BEST OUT` (an ablation (b) seed row): writes OUT from the arm's tuning base and its sweep's
   `best_tuned_run0.json` with `scripts/ablation/seed_runs.py: tuned_config` (the launcher's own merge) when OUT is
   missing; without BEST the sweep has not finished: exit 3, the row waits for a later leg;
4. `--optional_tuned_config NAME BASE BEST OUT` (timing, H2S-NN): the same when BEST exists, and `--NAME OUT` is
   added to the launcher's words (NAME without dashes); without BEST nothing is added (the launcher's central point,
   flagged);
5. `--copy SRC DIR`: copies SRC into DIR (created);
6. `--cwd DIR`: the launcher's working directory;
7. runs `python LAUNCHER WORDS...` (LAUNCHER resolved from the starting directory) and exits with its status.

    python scripts/tsubame/run_row.py --done <summary> [--cwd <dir>] [--copy <src> <dir>] \
        [--tuned_config <base> <best> <out>] [--optional_tuned_config <name> <base> <best> <out>] \
        -- <launcher.py> <launcher words ...> [--resume]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAITING = 3


def parse(argv: list[str]) -> tuple[argparse.Namespace, str, list[str]]:
    if "--" not in argv:
        raise SystemExit("usage: run_row.py --done FILE [...] -- launcher.py words...")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--done", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--copy", nargs=2, type=Path, default=None, metavar=("SRC", "DIR"))
    parser.add_argument("--tuned_config", nargs=3, type=Path, default=None, metavar=("BASE", "BEST", "OUT"))
    parser.add_argument("--optional_tuned_config", nargs=4, default=None, metavar=("NAME", "BASE", "BEST", "OUT"))
    args = parser.parse_args(argv[:split])
    rest = argv[split + 1:]
    if not rest:
        raise SystemExit("no launcher after --")
    words = rest[1:]
    if words and words[-1] == "--resume":
        words = words[:-1]
    return args, rest[0], words


def materialise(base: Path, best: Path, out: Path) -> bool:
    """Writes `out` (if missing) from the tuning base and the sweep's best trial; False when the sweep has no best yet."""
    if out.exists():
        return True
    if not best.exists():
        return False
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "ablation"))
    from seed_runs import tuned_config

    out.parent.mkdir(parents=True, exist_ok=True)
    tuned_config(base, best, out)
    return True


def main(argv: list[str] | None = None) -> int:
    args, launcher, words = parse(list(sys.argv[1:] if argv is None else argv))
    if args.done.exists():
        print(f"row complete: {args.done} exists")
        return 0
    if args.tuned_config is not None and not materialise(*args.tuned_config):
        print(f"waiting: {args.tuned_config[1]} does not exist yet (the arm's sweep has not finished)")
        return WAITING
    if args.optional_tuned_config is not None:
        name, base, best, out = args.optional_tuned_config
        if name.startswith("-"):
            raise SystemExit("--optional_tuned_config NAME is given without dashes")
        if materialise(Path(base), Path(best), Path(out)):
            words = [*words, f"--{name}", out]
    if args.copy is not None:
        src, folder = args.copy
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, folder / src.name)
    command = [sys.executable, str(Path(launcher).resolve()), *words]
    print(f"run_row: cwd={args.cwd or Path.cwd()} {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=args.cwd).returncode


if __name__ == "__main__":
    sys.exit(main())
