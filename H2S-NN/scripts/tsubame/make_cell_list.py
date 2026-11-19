"""TSUBAME card T3: the cell lists of phase 1 (baselines on HGB NC and the small LP tasks; guide section 6).

    python scripts/tsubame/make_cell_list.py --phase baselines-1 --out <out>/cells_tuning.txt
    python scripts/tsubame/make_cell_list.py --final-seeds --out-dir <out>      (-> <out>/cells_final_seeds.txt)

Tuning list: one line per **runnable** registry cell (`src/baselines/registry.py: iter_cells`, the source of
`campaign_summary`) with task NC on acm / dblp / imdb or LP on acm_pa / dblp_ap / imdb_ma / pubmed / lastfm, every
baseline method, in card 2 D5's form plus the TSUBAME output, state and W&B project; sorted by (task, dataset, method).
A cell the registry marks unsupported is a comment line with its reason and is never run.
Held out of phase 1: Freebase NC and freebase_bb, H2GB NC, the large LP tasks, GC, H2S-NN.

Phase `h2snn` (card 5 U4): `--phase h2snn --out <out>/cells_tuning.txt` writes the 24 H2S-NN tuning lines and
`<out>/cells_final_seeds_h2snn.txt` (seeds 0-4 per cell). `lines.py`/`status.py` read baseline lines only.

Final-seeds list: one `--final_seeds 0 1 2 3 4` line per tuning line whose sweep has `n_trials` finished attempts and
no `exceeds_24h` (read with `scripts/tsubame/lines.py: snapshot`, the reader of `status.py`); every other line is a
comment with its count. `<out>` is the directory of the list (`--out-dir`, default: the directory of `--out`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tsubame.lines import parse_line, runnable_lines, snapshot  # noqa: E402

PHASES = {
    "baselines-1": {
        "node_classification": ("acm", "dblp", "imdb"),
        "link_prediction": ("acm_pa", "dblp_ap", "imdb_ma", "pubmed", "lastfm"),
    },
}
N_TRIALS = 100
FINAL_SEEDS = (0, 1, 2, 3, 4)
WANDB_PROJECT = "h2snn_baseline_t4"
LAUNCHER = "scripts/run_baseline_tuning.py"

# Card 5 U4: phase `h2snn`, one tuning line per H2S-NN main-table cell (spec 1.26 §6: 100 trials, seed 0, then
# seeds 0-4), with H2S-NN's own launcher and experiment configs. Order: the 12 NC datasets (HGB 4, then the 8 H2GB),
# the 8 LP tasks, the 4 GC datasets. Each cell has its own exp_name and state directory (the launcher writes
# `<state_dir>/tuning_state_run0.json` and `<output_dir>/<exp_name>/best_tuned_run0.json`, which `--final_seeds`
# reads). Freebase NC is full batch unless card 5 U3 decides otherwise (see FREEBASE_NC_NOTE).
H2SNN_LAUNCHER = "scripts/run_h2snn_wandb_tuning.py"
H2SNN_WANDB_PROJECT = "h2snn_h2snn_t4"
H2SNN_CELLS: tuple[tuple[str, str], ...] = (
    *(("node_classification", d) for d in ("acm", "dblp", "imdb", "freebase")),
    *(("node_classification", d) for d in ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem",
                                          "ieee-cis-g", "h-pokec", "pdns")),
    *(("link_prediction", d) for d in ("acm_pa", "dblp_ap", "imdb_ma", "freebase_bb", "lastfm", "pubmed",
                                       "ogbn-mag_pp", "h-pokec_uu")),
    *(("graph_classification", d) for d in ("mutag", "enzymes", "nci1", "proteins")),
)
FREEBASE_NC_NOTE = ("# freebase NC: full batch (configs/experiments/freebase_h2snn_v3b.yaml resolves to "
                    "full_batch); card 5 U3 probe: see docs/card5_evidence/u3_freebase_probe.json")


def is_listed(row: dict) -> bool:
    """A registry row becomes a line only when it is runnable (unsupported cells are comments)."""
    return bool(row["runnable"])


def tuning_line(task: str, model: str, dataset: str, out: Path) -> str:
    return (f"{LAUNCHER} --task {task} --models {model} --datasets {dataset} --num_runs 1 --base_seed 0 "
            f"--n_trials {N_TRIALS} --output_dir {out}/experiments/baselines --state_dir {out}/state "
            f"--wandb_project {WANDB_PROJECT}")


def final_seeds_line(task: str, model: str, dataset: str, out: Path) -> str:
    seeds = " ".join(str(s) for s in FINAL_SEEDS)
    return (f"{LAUNCHER} --task {task} --models {model} --datasets {dataset} --num_runs 1 --base_seed 0 "
            f"--output_dir {out}/experiments/baselines --final_seeds {seeds}")


def h2snn_token(task: str, dataset: str) -> str:
    return f"h2snn_{task}_{dataset}"


def h2snn_tuning_line(task: str, dataset: str, out: Path) -> str:
    token = h2snn_token(task, dataset)
    return (f"{H2SNN_LAUNCHER} --config configs/experiments/{dataset}_h2snn_v3b.yaml --exp_name {token} "
            f"--num_runs 1 --base_seed 0 --n_trials {N_TRIALS} --output_dir {out}/experiments/h2snn "
            f"--state_dir {out}/state/{token} --wandb_project {H2SNN_WANDB_PROJECT}")


def h2snn_final_seeds_line(task: str, dataset: str, out: Path) -> str:
    seeds = " ".join(str(s) for s in FINAL_SEEDS)
    return (f"{H2SNN_LAUNCHER} --config configs/experiments/{dataset}_h2snn_v3b.yaml "
            f"--exp_name {h2snn_token(task, dataset)} --num_runs 1 --base_seed 0 "
            f"--output_dir {out}/experiments/h2snn --final_seeds {seeds}")


def h2snn_tuning_list(out: Path) -> list[str]:
    lines = ["# TSUBAME phase h2snn: one tuning line per H2S-NN main-table cell (make_cell_list.py, card 5 U4)"]
    for task, dataset in H2SNN_CELLS:
        if (task, dataset) == ("node_classification", "freebase"):
            lines.append(FREEBASE_NC_NOTE)
        lines.append(h2snn_tuning_line(task, dataset, out))
    return lines


def h2snn_final_seeds_list(out: Path) -> list[str]:
    lines = ["# TSUBAME phase h2snn final seeds: run a line only after its cell's tuning has "
             f"{N_TRIALS} finished trials (scripts/tsubame/lines.py reads baseline lines only)"]
    return lines + [h2snn_final_seeds_line(task, dataset, out) for task, dataset in H2SNN_CELLS]


def phase_rows(phase: str) -> list[dict]:
    from src.baselines.registry import iter_cells

    datasets = PHASES[phase]
    rows = [row for row in iter_cells() if row["dataset"] in datasets.get(row["task"], ())]
    return sorted(rows, key=lambda row: (row["task"], row["dataset"], row["model"]))


def tuning_list(phase: str, out: Path) -> list[str]:
    lines = [f"# TSUBAME phase {phase}: one tuning line per runnable registry cell (make_cell_list.py)"]
    for row in phase_rows(phase):
        if is_listed(row):
            lines.append(tuning_line(row["task"], row["model"], row["dataset"], out))
        else:
            lines.append(f"# unsupported {row['task']} {row['dataset']} {row['model']}: {row['reason']}")
    return lines


def final_seeds_list(tuning_text: str, out: Path) -> list[str]:
    lines = ["# TSUBAME final seeds: lines whose sweep finished (make_cell_list.py --final-seeds)"]
    for text in runnable_lines(tuning_text):
        line = parse_line(text)
        snap = snapshot(line)
        unit = snap["units"][0]
        finished = int(unit.get("attempted", 0)) if unit.get("exists") else 0
        status = unit.get("line_status")
        if line.num_runs == 1 and finished >= line.n_trials and status is None:
            lines.append(final_seeds_line(line.task, line.model, line.dataset, out))
        else:
            lines.append(f"# not finished {line.token}: {finished}/{line.n_trials} finished trials"
                         + (f", {status}" if status else ""))
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", choices=sorted([*PHASES, "h2snn"]))
    mode.add_argument("--final-seeds", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="the list file to write")
    ap.add_argument("--out-dir", type=Path, default=None, help="<out> (default: the directory of --out)")
    ap.add_argument("--tuning-list", type=Path, default=None, help="--final-seeds: default <out>/cells_tuning.txt")
    args = ap.parse_args(argv)
    if args.out is None and args.out_dir is None:
        ap.error("give --out or --out-dir")
    out_dir = (args.out_dir or args.out.parent).resolve()
    if args.phase == "h2snn":  # card 5 U4: the tuning list and its final-seeds list
        target = args.out or out_dir / "cells_tuning.txt"
        lines = h2snn_tuning_list(out_dir)
        seeds_target = out_dir / "cells_final_seeds_h2snn.txt"
        seeds_target.parent.mkdir(parents=True, exist_ok=True)
        seeds_target.write_text("\n".join(h2snn_final_seeds_list(out_dir)) + "\n", encoding="utf-8")
    elif args.phase:
        target = args.out or out_dir / "cells_tuning.txt"
        lines = tuning_list(args.phase, out_dir)
    else:
        target = args.out or out_dir / "cells_final_seeds.txt"
        source = args.tuning_list or out_dir / "cells_tuning.txt"
        lines = final_seeds_list(source.read_text(encoding="utf-8"), out_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runnable = len(runnable_lines("\n".join(lines)))
    print(f"{target}: {runnable} runnable line(s), {len(lines) - 1 - runnable} comment line(s)")


if __name__ == "__main__":
    main()
