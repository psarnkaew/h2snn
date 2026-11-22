"""TSUBAME card T3: the cell lists of phase 1 (baselines on HGB NC and the small LP tasks; guide section 6).

    python scripts/tsubame/make_cell_list.py --phase baselines-1 --out <out>/cells_tuning.txt
    python scripts/tsubame/make_cell_list.py --final-seeds --out-dir <out>      (-> <out>/cells_final_seeds.txt)

Tuning list: one line per **runnable** registry cell (`src/baselines/registry.py: iter_cells`, the source of
`campaign_summary`) with task NC on acm / dblp / imdb or LP on acm_pp / dblp_ap / imdb_ma / pubmed / lastfm, every
baseline method, in card 2 D5's form plus the TSUBAME output, state and W&B project; sorted by (task, dataset, method).
A cell the registry marks unsupported is a comment line with its reason and is never run.
Held out of phase 1: Freebase NC and freebase_bb, H2GB NC, the large LP tasks, GC, H2S-NN.
Phase `baselines-early` (card T7.4): HetSheaf and CSNN on ACM and DBLP NC (card 7 needs their tuned configurations);
`baselines-1` lists those cells as comments ("# in phase baselines-early"), so no line is in two chains.

Phase `h2snn` (card 5 U4): `--phase h2snn --out <out>/cells_tuning_h2snn.txt` writes the 24 H2S-NN tuning lines.
TSUBAME card T7.3: no final-seeds list is written up front any more (it could be submitted before tuning ends);
`h2snn_final_seeds_list` stays as a function (card 5's test reads it), but only `--final-seeds` writes final seeds.

Phases of card 7 V8 (`ablation`, `perturbation`, `timing`; the ops session owns the TSUBAME card): built from the
card 7 manifests, written on TSUBAME after the `h2snn` phase has tuned `full` (the manifests need its tuned files):
- `--phase ablation --manifest <out>/<dataset>/manifest.tsv ... [--out <out>/cells_ablation_seeds.txt]` (card 9 W8,
  pre-registration v3: configuration (a) only): no tuning list (a v3 manifest has no `tune` row); one line per `runs`
  row (`scripts/run_h2snn_runs.py` through `run_row.py`, done file = the row's `runs_summary.json`). A `tune` row or a
  configuration (b) row (a v2 manifest) raises. `full`'s seeds 1-4 are the main table's (no line).
- `--phase perturbation --manifest <pert>/manifest.tsv --out <out>/cells_perturbation.txt`: one line per runnable
  manifest row (perturbed graphs and the original-graph reference rows), the row's checked command through
  `run_row.py` (working directory, trials copy, done file = the row's summary); a "needs tuned config" row is a comment.
- `--phase timing --out <out>/cells_timing.txt`: one line per (main-table method, dataset) of
  `scripts/timing/time_methods.py` (ogbn-mag, IMDB), through `run_row.py`; H2S-NN gets its tuned file from the `h2snn`
  phase when the sweep has finished, else the central point (flagged by the launcher).
The `run_row.py` lines (ablation seeds, perturbation, timing) are not read by `lines.py` / `status.py` (a row is done when its summary
file exists). All of them run under `sweep_slots.sh` unchanged (`run_row.py`).

Card 8 C5 (author decisions 2026-11-20; cell statuses `src/baselines/cell_status.py`: filled / published / tune /
probe / N/A): `baselines-1` lists only `tune` cells of its datasets (a `filled`, `published` or `probe` cell, or one of
`baselines-early`, is a comment line); phase `baselines-2` = every other `tune` cell (Freebase NC, freebase_bb, the H2GB
NC datasets, the two large LP tasks); phase `published` = one `run_baseline_published.py ... --final_seeds 0 1 2 3 4`
line per `published` cell (5 seeds at the source configuration, no tuning); phase `probe` = one 1-trial
`--dry_run --grid_point min` tuning line per `probe` cell (MAGNN on the H2GB NC datasets and the large LP tasks: N/A
when the smallest grid point does not fit one H100; never a result). GC: no baseline line (option (a)).

Final-seeds list: one `--final_seeds 0 1 2 3 4` line per tuning line (baseline or H2S-NN) whose sweep has `n_trials`
finished attempts and no `exceeds_24h` (`sweep_finished`, read with `scripts/tsubame/lines.py: snapshot`, the reader
of `status.py`); every other line is a comment with its count. `<out>` is the directory of the list (`--out-dir`,
default: the directory of `--out`); `--tuning-list` names the tuning list (default `<out>/cells_tuning.txt`).
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
        "link_prediction": ("acm_pp", "dblp_ap", "imdb_ma", "pubmed", "lastfm"),
    },
    # TSUBAME card T7.4: HetSheaf and CSNN tuning on ACM and DBLP NC first; card 7's perturbation study needs their
    # tuned configurations. Card 7 perturbs NC graphs only (data/NC/<DATASET>_pert), so no LP cell is early.
    "baselines-early": {
        "node_classification": ("acm", "dblp"),
    },
}
#: Card 8 C5: the phases built from the cell statuses (`cell_status`), over NC and LP (GC: no baseline line).
STATUS_PHASES = ("baselines-2", "published", "probe")
STATUS_PHASE_TASKS = ("node_classification", "link_prediction")
PUBLISHED_LAUNCHER = "scripts/run_baseline_published.py"
#: TSUBAME card T7.4: methods of a phase (all registry methods when absent).
PHASE_MODELS = {"baselines-early": ("hetsheaf", "csnn")}
#: TSUBAME card T7.4: `baselines-1` leaves out every `baselines-early` cell (two chains never hold one line).
PHASE_EXCLUDES = {"baselines-1": ("baselines-early",)}
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
    *(("link_prediction", d) for d in ("acm_pp", "dblp_ap", "imdb_ma", "freebase_bb", "lastfm", "pubmed",
                                       "ogbn-mag_pp", "h-pokec_uu")),
    *(("graph_classification", d) for d in ("mutag", "enzymes", "nci1", "proteins")),
)
FREEBASE_NC_NOTE = ("# freebase NC: full batch (configs/experiments/freebase_h2snn_v3b.yaml resolves to "
                    "full_batch); card 5 U3 probe: see docs/card5_evidence/u3_freebase_probe.json")


# Card 7 V8: the analysis phases (rows of the card 7 manifests as cell-list lines through `run_row.py`).
ROW_RUNNER = "scripts/tsubame/run_row.py"
CARD7_PHASES = ("ablation", "perturbation", "timing")
TIMING_DATASETS = ("ogbn-mag", "imdb")
TIMING_EPOCHS = 4


def _tsv(path: Path) -> list[dict]:
    import csv

    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def ablation_lists(manifests: list[Path], out: Path) -> list[str]:
    """The seed lines of the ablation manifests (`scripts/ablation/seed_runs.py`), in manifest order: one per `runs`
    row of configuration (a) (card 9 W8, pre-registration v3); a `tune` or (b) row raises."""
    seeds = ["# TSUBAME phase ablation seeds: one line per `runs` row, configuration (a) only (card 9 W8, "
             "pre-registration v3); no tuning list"]
    exp_out = out / "experiments" / "ablation"
    for manifest in manifests:
        for r in _tsv(manifest):
            if r["kind"] != "runs" or r["configuration"] != "a":
                raise ValueError(f"{manifest}: row kind {r['kind']!r}, configuration {r['configuration']!r}: "
                                 "pre-registration v3 has configuration (a) `runs` rows only (rebuild the manifest "
                                 "with scripts/ablation/seed_runs.py)")
            seeds.append(f"{ROW_RUNNER} --done {exp_out}/{r['exp_name']}/runs_summary.json -- "
                         f"scripts/run_h2snn_runs.py --config {r['config']} --base_seed {r['base_seed']} "
                         f"--num_runs {r['num_runs']} --exp_name {r['exp_name']} --output_dir {exp_out}")
    return seeds


def perturbation_line(row: dict) -> str:
    """A runnable perturbation manifest row as one line: its checked command (`python <checker> ...`, optionally
    after `mkdir -p <dir> && cp <trials> <dir>/`) through `run_row.py` with the row's working directory."""
    import shlex

    parts = [p.strip() for p in row["command"].split(" && ")]
    copy = ""
    if len(parts) == 3:
        mkdir, cp, command = parts
        cp_words = shlex.split(cp)
        if shlex.split(mkdir)[:2] != ["mkdir", "-p"] or cp_words[0] != "cp" or len(cp_words) != 3:
            raise ValueError(f"unexpected row command {row['command']!r}")
        copy = f"--copy {cp_words[1]} {cp_words[2].rstrip('/')} "
    elif len(parts) == 1:
        command = parts[0]
    else:
        raise ValueError(f"unexpected row command {row['command']!r}")
    words = shlex.split(command)
    if words[0] != "python" or not words[1].endswith("run_checked.py"):
        raise ValueError(f"not a checked command: {command!r}")
    return f"{ROW_RUNNER} --done {row['summary']} --cwd {row['cwd']} {copy}-- {shlex.join(words[1:])}"


def perturbation_list(manifest: Path) -> list[str]:
    lines = ["# TSUBAME phase perturbation: one line per runnable manifest row (card 7 V8, amendments 1-8)"]
    for row in _tsv(manifest):
        tag = row["graph_kind"] if row["graph_kind"] == "original" else f"{row['graph_kind']}_h{row['level']}"
        if not row["command"]:
            lines.append(f"# {row['configuration']}: {row['dataset']} {tag} {row['method']}")
            continue
        lines.append(perturbation_line(row))
    return lines


def timing_list(out: Path, methods: dict[str, list[str]] | None = None) -> list[str]:
    """One line per (main-table method, dataset) of `scripts/timing/time_methods.py` (card 7 V7, amendment 5)."""
    if methods is None:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "timing"))
        from time_methods import main_table_methods

        methods = {d: main_table_methods(d) for d in TIMING_DATASETS}
    lines = ["# TSUBAME phase timing: one line per (main-table method, dataset) (card 7 V7/V8); one H100 each"]
    for dataset in TIMING_DATASETS:
        for method in methods[dataset]:
            result = out / "timing" / f"{method}_{dataset}.json"
            tuned = ""
            if method == "h2snn":
                best = (out / "experiments" / "h2snn" / h2snn_token("node_classification", dataset)
                        / "best_tuned_run0.json")
                tuned = (f"--optional_tuned_config h2snn_config configs/experiments/{dataset}_h2snn_v3b.yaml "
                         f"{best} {out}/timing/configs/h2snn_{dataset}.yaml ")
            lines.append(f"{ROW_RUNNER} --done {result} {tuned}-- scripts/timing/time_methods.py --method {method} "
                         f"--dataset {dataset} --epochs {TIMING_EPOCHS} --out {result} "
                         f"--baseline_output {out}/experiments/baselines --device cuda")
    return lines


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
            f"--output_dir {out}/experiments/baselines --final_seeds {seeds} "
            f"--wandb_project {WANDB_PROJECT}")  # card 8 C7: the tuning lines' project


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


def in_phase(phase: str, row: dict) -> bool:
    """A registry row's cell belongs to `phase` (its datasets and, where given, its methods)."""
    models = PHASE_MODELS.get(phase)
    return row["dataset"] in PHASES[phase].get(row["task"], ()) and (models is None or row["model"] in models)


def excluded(phase: str, row: dict) -> str | None:
    """The earlier phase that already holds this row's cell, or None (card T7.4)."""
    for other in PHASE_EXCLUDES.get(phase, ()):
        if in_phase(other, row):
            return other
    return None


def phase_rows(phase: str) -> list[dict]:
    from src.baselines.registry import iter_cells

    rows = [row for row in iter_cells() if in_phase(phase, row)]
    return sorted(rows, key=lambda row: (row["task"], row["dataset"], row["model"]))


def not_tuned(row: dict) -> str | None:
    """Why a registry row gets no tuning line (card 8 C5), or None for a `tune` cell."""
    from src.baselines.cell_status import cell_status

    status = cell_status(row["model"], row["task"], row["dataset"])
    if status == "filled":
        return "filled (published number, register)"
    if status == "N/A":
        return f"unsupported: {row['reason']}"
    if status in ("published", "probe"):
        return f"in phase {status}"
    return None


def tuning_list(phase: str, out: Path) -> list[str]:
    lines = [f"# TSUBAME phase {phase}: one tuning line per runnable registry cell (make_cell_list.py)"]
    for row in phase_rows(phase):
        other = excluded(phase, row)
        if other:
            lines.append(f"# in phase {other}: {row['task']} {row['dataset']} {row['model']}")
        elif not is_listed(row):
            lines.append(f"# unsupported {row['task']} {row['dataset']} {row['model']}: {row['reason']}")
        elif (why := not_tuned(row)) is not None:  # card 8 C5
            lines.append(f"# {why}: {row['task']} {row['dataset']} {row['model']}")
        else:
            lines.append(tuning_line(row["task"], row["model"], row["dataset"], out))
    return lines


def published_line(task: str, model: str, dataset: str, out: Path) -> str:
    seeds = " ".join(str(s) for s in FINAL_SEEDS)
    return (f"{PUBLISHED_LAUNCHER} --task {task} --models {model} --datasets {dataset} --final_seeds {seeds} "
            f"--output_dir {out}/experiments/baselines --state_dir {out}/state "
            f"--wandb_project {WANDB_PROJECT}")  # card 8 C7


def probe_line(task: str, model: str, dataset: str, out: Path) -> str:
    return (f"{LAUNCHER} --task {task} --models {model} --datasets {dataset} --num_runs 1 --base_seed 0 "
            f"--n_trials 1 --dry_run --grid_point min --output_dir {out}/experiments/probe "
            f"--state_dir {out}/state/probe --wandb_project {WANDB_PROJECT}")


def status_rows() -> list[dict]:
    from src.baselines.registry import iter_cells

    rows = [row for row in iter_cells(tasks=STATUS_PHASE_TASKS)]
    return sorted(rows, key=lambda row: (row["task"], row["dataset"], row["model"]))


def in_tuning_phase(row: dict) -> str | None:
    """The earlier tuning phase (`baselines-early`, `baselines-1`) that holds this row's datasets, or None."""
    for phase in ("baselines-early", "baselines-1"):
        if in_phase(phase, row):
            return phase
    return None


def status_list(phase: str, out: Path) -> list[str]:
    """Card 8 C5: phase `baselines-2`, `published` or `probe` from the cell statuses."""
    from src.baselines.cell_status import cell_status

    header = {"baselines-2": "one tuning line per `tune` cell not in baselines-early / baselines-1",
              "published": "one 5-seed line per `published` cell (source configuration, no tuning)",
              "probe": "one 1-trial dry-run line per `probe` cell at the smallest grid point (never a result)"}[phase]
    lines = [f"# TSUBAME phase {phase}: {header} (make_cell_list.py, card 8 C5)"]
    for row in status_rows():
        status = cell_status(row["model"], row["task"], row["dataset"])
        cell = (row["task"], row["model"], row["dataset"])
        if phase == "published" and status == "published":
            lines.append(published_line(*cell, out))
        elif phase == "probe" and status == "probe":
            lines.append(probe_line(*cell, out))
        elif phase == "baselines-2" and status == "tune" and in_tuning_phase(row) is None:
            lines.append(tuning_line(*cell, out))
    return lines


def sweep_finished(line) -> tuple[bool, int, str | None]:
    """The gate of a final-seeds line (baselines and H2S-NN alike): one unit, `n_trials` finished attempts, no
    `exceeds_24h`. Returns (finished?, finished attempts, line status)."""
    unit = snapshot(line)["units"][0]
    finished = int(unit.get("attempted", 0)) if unit.get("exists") else 0
    status = unit.get("line_status")
    return line.num_runs == 1 and finished >= line.n_trials and status is None, finished, status


def final_seeds_list(tuning_text: str, out: Path) -> list[str]:
    lines = ["# TSUBAME final seeds: lines whose sweep finished (make_cell_list.py --final-seeds)"]
    for text in runnable_lines(tuning_text):
        line = parse_line(text)
        ok, finished, status = sweep_finished(line)
        if ok and line.launcher == "h2snn":  # card T7.3: H2S-NN final seeds only after its tuning
            lines.append(h2snn_final_seeds_line(line.task, line.dataset, out))
        elif ok:
            lines.append(final_seeds_line(line.task, line.model, line.dataset, out))
        else:
            lines.append(f"# not finished {line.token}: {finished}/{line.n_trials} finished trials"
                         + (f", {status}" if status else ""))
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase", choices=sorted([*PHASES, "h2snn", *CARD7_PHASES, *STATUS_PHASES]))
    mode.add_argument("--final-seeds", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="the list file to write")
    ap.add_argument("--out-dir", type=Path, default=None, help="<out> (default: the directory of --out)")
    ap.add_argument("--tuning-list", type=Path, default=None, help="--final-seeds: default <out>/cells_tuning.txt")
    ap.add_argument("--manifest", type=Path, nargs="+", default=None,
                    help="card 7 phases ablation (one per dataset) / perturbation (one): the manifest(s)")
    args = ap.parse_args(argv)
    if args.out is None and args.out_dir is None:
        ap.error("give --out or --out-dir")
    out_dir = (args.out_dir or args.out.parent).resolve()
    if args.phase in ("ablation", "perturbation") and not args.manifest:
        ap.error(f"--phase {args.phase} needs --manifest")
    if args.phase == "ablation":  # card 9 W8: the seed rows only (configuration (a); no tuning list)
        target = args.out or out_dir / "cells_ablation_seeds.txt"
        lines = ablation_lists(args.manifest, out_dir)
    elif args.phase == "perturbation":  # card 7 V8
        if len(args.manifest) != 1:
            ap.error("--phase perturbation takes one manifest")
        target = args.out or out_dir / "cells_perturbation.txt"
        lines = perturbation_list(args.manifest[0])
    elif args.phase == "timing":  # card 7 V8
        target = args.out or out_dir / "cells_timing.txt"
        lines = timing_list(out_dir)
    elif args.phase == "h2snn":  # card 5 U4: the tuning list (card T7.3: no final-seeds list up front)
        target = args.out or out_dir / "cells_tuning.txt"
        lines = h2snn_tuning_list(out_dir)
    elif args.phase in STATUS_PHASES:  # card 8 C5
        target = args.out or out_dir / f"cells_{args.phase.replace('-', '_')}.txt"
        lines = status_list(args.phase, out_dir)
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
