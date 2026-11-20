"""Card 10 L3: split one cell list between the lab cluster and TSUBAME. Standard library only.

    python scripts/slurm/site_split.py --list <out>/cells_tuning.txt --fit-check <out>/fit_check.tsv \
        --gpu-type 3090 --lab-out <lab out> --t4-out <t4 out (or a local mirror of its state/ and experiments/)>

Writes `<list stem>.lab.txt` and `<list stem>.t4.txt` next to the list: disjoint, and together exactly the runnable
lines of the list (`scripts/tsubame/lines.py` numbering), each in the list's order. The list itself is unchanged
(a `make_cell_list.py` list).

A line's cell is its `lines.py` token (the state / output directory it writes); `fit_check.tsv` (card 10 L2) is joined
by cell, so a list written with either site's `--out-dir` splits the same way. Only the rows of `--gpu-type` count:
one lab list = one GPU type (`submit_lab.sh`, condition 1).
- A line goes to the lab only if its cell's fit-check status on `--gpu-type` is `fits` (condition 2); `oom`, `error`
  and a cell with no fit-check row go to TSUBAME.
- A line is never moved away from a site where its cell already has state (`<site out>/state/<token>/tuning_state_*`
  or `<site out>/experiments/*/<token>/final_seed*.json`, the layout `collect.py` reads): state at TSUBAME -> TSUBAME;
  state at the lab -> the lab, and an error if the cell does not `fits`; state at both sites -> an error.
Both output directories must exist (pass a mirror of TSUBAME's `state/` and `experiments/` when run on the lab), so
missing state is never mistaken for no state. Any error: nothing is written, exit 3.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tsubame.lines import parse_line, runnable_lines  # noqa: E402

FIT_COLUMNS = ("line_no", "cell", "gpu_type", "peak_mem_mib", "status", "line")
STATUSES = ("fits", "oom", "error")


def read_fit_check(path: Path, gpu_type: str) -> dict[str, str]:
    """cell -> status on `gpu_type`. Two rows of one cell and GPU type with different statuses are an error."""
    status: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(FIT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            if row["gpu_type"] != gpu_type:
                continue
            if row["status"] not in STATUSES:
                raise ValueError(f"{path}: status {row['status']!r} of {row['cell']} is not one of {STATUSES}")
            if status.setdefault(row["cell"], row["status"]) != row["status"]:
                raise ValueError(f"{path}: cell {row['cell']} has two statuses on {gpu_type}")
    return status


def has_state(out: Path, token: str) -> bool:
    return any((out / "state" / token).glob("tuning_state_*.json")) or \
        any((out / "experiments").glob(f"*/{token}/final_seed*.json"))


def split(lines: list[str], fit: dict[str, str], lab_out: Path, t4_out: Path) -> tuple[list[str], list[str], list[str]]:
    lab, t4, errors = [], [], []
    for number, text in enumerate(lines, start=1):
        token = parse_line(text).token
        status = fit.get(token, "missing")
        at_lab, at_t4 = has_state(lab_out, token), has_state(t4_out, token)
        if at_lab and at_t4:
            errors.append(f"line {number}: cell {token} has state at both sites")
        elif at_t4:
            t4.append(text)
        elif at_lab and status != "fits":
            errors.append(f"line {number}: cell {token} has lab state but its fit check is {status}")
        elif status == "fits":
            lab.append(text)
        else:
            t4.append(text)
    return lab, t4, errors


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--fit-check", type=Path, required=True)
    ap.add_argument("--gpu-type", required=True, choices=["3090", "titan_rtx", "a6000", "2080ti"])
    ap.add_argument("--lab-out", type=Path, required=True)
    ap.add_argument("--t4-out", type=Path, required=True)
    args = ap.parse_args(argv)
    for name in ("lab_out", "t4_out"):
        if not getattr(args, name).is_dir():
            raise SystemExit(f"--{name.replace('_', '-')} {getattr(args, name)} is not a directory")
    lines = runnable_lines(args.list.read_text(encoding="utf-8"))
    try:
        fit = read_fit_check(args.fit_check, args.gpu_type)
        lab, t4, errors = split(lines, fit, args.lab_out, args.t4_out)
    except ValueError as err:
        errors = [str(err)]
    if errors:
        print("refused, nothing written:\n  " + "\n  ".join(errors), file=sys.stderr)
        raise SystemExit(3)
    stem = args.list.with_suffix("")
    head = f"# site_split of {args.list.name} (fit check {args.fit_check.name}, gpu {args.gpu_type})"
    for suffix, part in ((".lab.txt", lab), (".t4.txt", t4)):
        Path(f"{stem}{suffix}").write_text("\n".join([head, *part]) + "\n", encoding="utf-8")
    print(f"lines {len(lines)}: lab {len(lab)} ({stem}.lab.txt), t4 {len(t4)} ({stem}.t4.txt)")


if __name__ == "__main__":
    main()
