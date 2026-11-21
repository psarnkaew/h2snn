"""Card 10 L2 (Amendment 1 option A, A2.2, A3): the largest-point fit check of a cell list on one lab GPU type.
Standard library only. Not a result: it only decides which cells may run on the lab (condition 2).

    # 1. one probe line per cell (tuned cells: the launcher's own `--grid_point max`, `--n_trials 1 --dry_run`;
    #    published cells: the published configuration, one seed, `--dry_run`)
    python scripts/slurm/fit_check.py build  --list <out>/cells_tuning.txt --fit-out <out>/fit_3090
    # 2. submit them on the GPU type the cells would use (scripts/slurm/submit_lab.sh, one leg)
    python scripts/slurm/fit_check.py submit --fit-out <out>/fit_3090 --gpu-type 3090 --mem 48G --time 6:00:00
    # 3. after the jobs: <fit-out>/fit_check.tsv (read by scripts/slurm/site_split.py)
    python scripts/slurm/fit_check.py read   --fit-out <out>/fit_3090 --gpu-type 3090

`--grid_point max` (scripts/slurm/grid_points.py) sets every size axis to its largest value, every memory-relevant
categorical key to its most memory-hungry value, and runs one trial per value of a memory-relevant key whose largest
value the code does not show (A3.1); it announces the number of trials (`FIT_GRID {"points": k}`). A published
probe runs the published configuration as the line states it (`scripts/run_baseline_published.py --dry_run`, one
seed, one epoch, then the whole-split evaluation; A3.2).

The probe lines write only under `<fit-out>` (state, outputs, logs, W&B project `<project>_fitcheck`), never into a
run directory. Each trial prints one `FIT_PEAK {json}` line: the peak GPU memory of the trial process after one
training epoch and the evaluation of the whole validation and test splits, and which of the three phases ran. The
cell's peak is the largest over its trials. Status per cell (A2.2):
- `fits`: every announced trial printed FIT_PEAK with all three phases, and peak / GPU memory <= 0.75 (24 GB -> 18 GB,
  48 GB -> 36 GB);
- `oom`: out of memory in the line log, or every trial measured and the peak above 0.75 (no headroom);
- `error`: anything else (no log, fewer FIT_PEAK lines than trials, a phase not measured, another failure).
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tsubame.lines import (BASELINE_LAUNCHER, H2SNN_LAUNCHER, PUBLISHED_LAUNCHER,  # noqa: E402
                                   parse_line, runnable_lines)

HEADROOM = 0.75
PEAK_TAG = "FIT_PEAK"
GRID_TAG = "FIT_GRID"
FIT_COLUMNS = ("line_no", "cell", "gpu_type", "peak_mem_mib", "peak_fraction", "status", "line")
PROBE_LIST = "fit_lines.txt"
OOM_MARKERS = ("out of memory", "outofmemoryerror")
#: options a list line may carry, per launcher; the probe line rebuilds the cell from the kept ones (anything not
#: kept or replaced is refused, so no memory-relevant option is dropped silently)
KEEP = {"baseline": ("--task", "--models", "--datasets"), "h2snn": ("--config", "--exp_name"),
        "published": ("--task", "--models", "--datasets")}
REPLACED = ("--num_runs", "--base_seed", "--n_trials", "--output_dir", "--state_dir", "--wandb_project",
            "--final_seeds", "--run_start", "--run_count")


def _options(words: list[str]) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    key = None
    for word in words[1:]:
        if word.startswith("--"):
            key = word
            options[key] = []
        elif key is None:
            raise ValueError(f"unexpected argument {word!r}")
        else:
            options[key].append(word)
    return options


def probe_line(text: str, fit_out: Path) -> tuple[str, str]:
    """(cell token, the cell's probe line). The cell keeps its token (state and output directory names)."""
    line = parse_line(text)
    options = _options(shlex.split(text))
    keep = KEEP[line.launcher]
    unknown = sorted(set(options) - set(keep) - set(REPLACED))
    if unknown:
        raise ValueError(f"{line.token}: options the fit check would drop: {unknown}")
    project = (options.get("--wandb_project") or ["h2snn"])[0] + "_fitcheck"
    kept = " ".join(f"{key} {shlex.join(options[key])}" for key in keep if key in options)
    if line.launcher == "published":  # A3.2: the published configuration, one seed, one epoch
        probe = (f"scripts/{PUBLISHED_LAUNCHER} {kept} --final_seeds 0 --dry_run --wandb_project {project} "
                 f"--output_dir {fit_out}/experiments/baselines --state_dir {fit_out}/state")
    else:
        common = f"--num_runs 1 --base_seed 0 --n_trials 1 --dry_run --grid_point max --wandb_project {project}"
        if line.launcher == "h2snn":
            probe = (f"scripts/{H2SNN_LAUNCHER} {kept} {common} --output_dir {fit_out}/experiments/h2snn "
                     f"--state_dir {fit_out}/state/{line.token}")
        else:
            probe = (f"scripts/{BASELINE_LAUNCHER} {kept} {common} --output_dir {fit_out}/experiments/baselines "
                     f"--state_dir {fit_out}/state")
    if parse_line(probe).token != line.token:
        raise ValueError(f"probe line of {line.token} names another cell")
    return line.token, probe


def build(list_path: Path, fit_out: Path) -> list[str]:
    """One probe line per cell of the list (a cell's tuning and final-seeds lines share one probe), in list order."""
    probes, seen = [], set()
    for text in runnable_lines(list_path.read_text(encoding="utf-8")):
        token, probe = probe_line(text, fit_out)
        if token not in seen:
            seen.add(token)
            probes.append(probe)
    fit_out.mkdir(parents=True, exist_ok=True)
    (fit_out / PROBE_LIST).write_text(
        f"# fit check (card 10 L2/A3) of {list_path}\n" + "\n".join(probes) + "\n", encoding="utf-8")
    return probes


def tagged(text: str, tag: str) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    for raw in text.splitlines():
        if raw.startswith(tag + " "):
            try:
                out.append(json.loads(raw[len(tag) + 1:]))
            except json.JSONDecodeError:
                out.append(None)
    return out


def classify(log_text: str | None, threshold: float = HEADROOM) -> tuple[str, float | None, float | None]:
    """(status, peak MiB, peak fraction) of one probe line's log: the largest peak over the probe's trials."""
    if log_text is None:
        return "error", None, None
    grids = [g for g in tagged(log_text, GRID_TAG) if isinstance(g, dict)]
    points = int(grids[-1].get("points", 1)) if grids else 1   # a published probe or a one-point grid: 1 trial
    peaks = [p for p in tagged(log_text, PEAK_TAG)
             if isinstance(p, dict) and p.get("cuda") and p.get("peak_bytes") and p.get("total_bytes")]
    mib = max((p["peak_bytes"] / 2 ** 20 for p in peaks), default=None)
    fraction = max((p["peak_bytes"] / p["total_bytes"] for p in peaks), default=None)
    if any(m in log_text.lower() for m in OOM_MARKERS):
        return "oom", mib, fraction
    if len(peaks) < points:
        return "error", mib, fraction            # a trial of the probe printed no peak: no decision
    if not all(all((p.get("phases") or {}).get(k) for k in ("train", "val", "test")) for p in peaks):
        return "error", mib, fraction            # a phase was not measured: no decision
    return ("fits" if fraction <= threshold else "oom"), mib, fraction


def read(fit_out: Path, gpu_type: str, threshold: float = HEADROOM) -> list[dict[str, Any]]:
    rows = []
    stem = Path(PROBE_LIST).stem
    for number, text in enumerate(runnable_lines((fit_out / PROBE_LIST).read_text(encoding="utf-8")), start=1):
        logs = sorted((fit_out / "logs" / "lines").glob(f"{stem}.line{number:03d}.*.log"),
                      key=lambda p: p.stat().st_mtime)
        log_text = logs[-1].read_text(encoding="utf-8", errors="replace") if logs else None
        status, mib, fraction = classify(log_text, threshold)
        rows.append({"line_no": number, "cell": parse_line(text).token, "gpu_type": gpu_type,
                     "peak_mem_mib": "" if mib is None else f"{mib:.0f}",
                     "peak_fraction": "" if fraction is None else f"{fraction:.4f}", "status": status, "line": text})
    with (fit_out / "fit_check.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--list", type=Path, required=True)
    b.add_argument("--fit-out", type=Path, required=True)
    s = sub.add_parser("submit")
    s.add_argument("--fit-out", type=Path, required=True)
    for name in ("--gpu-type", "--mem", "--time"):
        s.add_argument(name, required=True)
    r = sub.add_parser("read")
    r.add_argument("--fit-out", type=Path, required=True)
    r.add_argument("--gpu-type", required=True)
    args = ap.parse_args(argv)
    fit_out = args.fit_out.resolve()
    if args.cmd == "build":
        print(f"{len(build(args.list, fit_out))} probe lines -> {fit_out / PROBE_LIST}")
    elif args.cmd == "submit":
        raise SystemExit(subprocess.call(
            ["bash", "scripts/slurm/submit_lab.sh", "--list", str(fit_out / PROBE_LIST), "--out", str(fit_out),
             "--gpu-type", args.gpu_type, "--mem", args.mem, "--time", args.time, "--legs", "1"]))
    else:
        rows = read(fit_out, args.gpu_type)
        counts = {s: sum(r["status"] == s for r in rows) for s in ("fits", "oom", "error")}
        print(f"{len(rows)} cells on {args.gpu_type}: {counts} -> {fit_out / 'fit_check.tsv'}")


if __name__ == "__main__":
    main()
