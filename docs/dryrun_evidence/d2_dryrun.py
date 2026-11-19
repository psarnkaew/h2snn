"""Card 2 D2: dry run of one cell on real data (one epoch at the largest memory point, then one validation pass).

    python ../docs/dryrun_evidence/d2_dryrun.py --cell 'h2snn|node_classification|acm' --out <dir>   (from H2S-NN/)

Records status (OK / OOM / error / non-finite / stop condition), the error with its first file:line, peak GPU and
host memory, time of the training epoch and of one validation pass, GPU and node, PCG iterations (H2S-NN) and fired
stop conditions. **No metric value is read, stored or printed by this script** (card 2 Rule): the trainers' metric
dicts are never inspected; the epoch report of the baselines is used as a timestamp only. W&B project: h2snn-dryrun
(the H2S-NN logger logs as in tuning; this script adds only the resource record to the run summary).

Nothing in src/ is changed: timing and PCG counts come from wrappers installed on the trainer instance and on
`src.models.anchored_solve.pcg` (looked up as a module global by `solve_anchored` and `AnchoredSolve.backward`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

import torch  # noqa: E402
import yaml  # noqa: E402

WANDB_PROJECT = "h2snn-dryrun"
REPO = Path.cwd().resolve()

# ---------------------------------------------------------------------------------------- memory and time
STATE: dict[str, Any] = {"phase": "setup", "pcg": {}, "phase_gpu_peak": {}, "times": {}, "calls": {}}


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def host_peak_gib() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(own, children) / 2**20  # ru_maxrss is KiB on Linux


def gpu_peak_gib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


class Phase:
    """Times one phase (CUDA synchronised) and keeps its GPU peak; nested calls add to the outer phase."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "Phase":
        self.outer = STATE["phase"]
        STATE["phase"] = self.name
        _sync()
        if torch.cuda.is_available():
            self.before_peak = gpu_peak_gib()
            torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        _sync()
        elapsed = time.perf_counter() - self.start
        STATE["times"][self.name] = STATE["times"].get(self.name, 0.0) + elapsed
        STATE["calls"][self.name] = STATE["calls"].get(self.name, 0) + 1
        if torch.cuda.is_available():
            peak = gpu_peak_gib()
            STATE["phase_gpu_peak"][self.name] = max(STATE["phase_gpu_peak"].get(self.name, (0, 0))[0], peak[0]), \
                max(STATE["phase_gpu_peak"].get(self.name, (0, 0))[1], peak[1])
            STATE["overall_gpu_peak"] = tuple(max(a, b, c) for a, b, c in zip(
                STATE.get("overall_gpu_peak", (0.0, 0.0)), self.before_peak, peak))
        STATE["phase"] = self.outer


def wrap(obj: Any, name: str, phase_of) -> None:
    """Replace `obj.name` by a timed call; `phase_of(args, kwargs, call_index)` names the phase."""
    original = getattr(obj, name)
    counter = {"n": 0}

    def timed(*args, **kwargs):
        phase = phase_of(args, kwargs, counter["n"])
        counter["n"] += 1
        if phase is None:
            return original(*args, **kwargs)
        with Phase(phase):
            return original(*args, **kwargs)

    setattr(obj, name, timed)


def install_pcg_counter() -> None:
    import src.models.anchored_solve as solve

    original = solve.pcg

    def counted(*args, **kwargs):
        out = original(*args, **kwargs)
        caller = sys._getframe(1).f_code.co_name
        kind = "bwd" if caller == "backward" else "fwd"
        iterations, converged = out[1], out[2]
        key = f"{STATE['phase']}/{kind}"
        rec = STATE["pcg"].setdefault(key, {"solves": 0, "max_iters": 0, "sum_max_iters": 0, "not_converged": 0})
        top = int(iterations.max()) if iterations.numel() else 0
        rec["solves"] += 1
        rec["max_iters"] = max(rec["max_iters"], top)
        rec["sum_max_iters"] += top
        rec["not_converged"] += int((~converged).sum())
        return out

    solve.pcg = counted


# ---------------------------------------------------------------------------------------- error classification
def repo_frame(tb_frames) -> str:
    """The innermost frame inside this repository (src/, scripts/, vendor/), else the innermost frame."""
    inner = ""
    for frame in tb_frames:
        path = Path(frame.filename)
        where = f"{frame.filename}:{frame.lineno} ({frame.name})"
        try:
            rel = path.resolve().relative_to(REPO)
            where = f"{rel}:{frame.lineno} ({frame.name})"
            if not str(rel).startswith("../"):
                inner = where
        except ValueError:
            pass
    return inner or (f"{tb_frames[-1].filename}:{tb_frames[-1].lineno}" if tb_frames else "")


def first_raise_frame(tb_frames) -> str:
    """The innermost frame of the traceback (where the exception was raised)."""
    if not tb_frames:
        return ""
    f = tb_frames[-1]
    try:
        return f"{Path(f.filename).resolve().relative_to(REPO)}:{f.lineno} ({f.name})"
    except ValueError:
        return f"{f.filename}:{f.lineno} ({f.name})"


def classify(exc: BaseException) -> dict[str, Any]:
    frames = traceback.extract_tb(exc.__traceback__)
    record: dict[str, Any] = {"error_type": type(exc).__name__, "error": str(exc)[:600],
                              "raised_at": first_raise_frame(frames), "repo_frame": repo_frame(frames),
                              "traceback_tail": "".join(traceback.format_exception(exc)[-12:])[-4000:]}
    message = str(exc)
    from src.models.anchored_solve import NonFiniteLossError, PCGConvergenceError, StopConditionError

    if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in message \
            or "out of memory" in message.lower() and "cuda" in message.lower():
        record["status"] = "OOM"
        record["oom_tried_to_allocate"] = message.split("Tried to allocate")[1].split(".")[0].strip() \
            if "Tried to allocate" in message else ""
    elif isinstance(exc, NonFiniteLossError):
        record["status"] = "non-finite"
    elif isinstance(exc, (StopConditionError, PCGConvergenceError)):
        record["status"] = "stop condition"
        record["stop_condition"] = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))
                                    for k in ("condition", "value", "limit", "epoch")
                                    if (v := getattr(exc, k, None)) is not None}
        record["stop_condition"]["class"] = type(exc).__name__
    elif isinstance(exc, MemoryError):
        record["status"] = "host OOM"
    else:
        record["status"] = "error"
    return record


# ---------------------------------------------------------------------------------------- H2S-NN
H2GB_NC = ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns")
#: Card 2 Rule: H2S-NN's largest memory point: d 64, l_max 5, r 64; batch 1024 on mini-batch cells; GC pooling
#: attention (the only pooling with parameters). One epoch.
H2SNN_POINT = {"model.d": 64, "data.wisp_max_length": 5, "data.wisp_walks_per_anchor": 64}
#: H2S-NN tuning grids (scripts/run_h2snn_wandb_tuning.py:68-73), for the step-down check.
H2SNN_GRID = {"model.d": [16, 32, 64], "data.wisp_max_length": [2, 3, 4, 5], "data.wisp_walks_per_anchor": [8, 16, 32, 64]}
#: Card 2 step-down (author, 2026-11-16): after an OOM at 24 GB, ONE memory axis one grid step lower, all else the
#: largest point. Set from --step_down axis=value; a value that is not the next lower grid value is refused.
STEP_DOWN: dict[str, Any] = {}


def check_step_down(name: str, current: Any, grid: list[Any]) -> Any:
    value = STEP_DOWN[name]
    if isinstance(current, bool):  # a switch (HEAVY_SWITCH): the other value of the grid is the lighter one
        if value == current or value not in grid:
            raise ValueError(f"step-down {name}={value} is not the other value of {grid} (max point {current})")
        return value
    lower = sorted(v for v in grid if v < current)
    if not lower or value != lower[-1]:
        raise ValueError(f"step-down {name}={value} is not one grid step below {current} in {grid}")
    return value


def h2snn_raw(dataset: str, task: str, out: Path) -> dict[str, Any]:
    raw = yaml.safe_load((REPO / "configs" / "experiments" / f"{dataset}_h2snn_v3b.yaml").read_text())
    raw["model"]["d"] = 64
    raw["data"]["wisp_max_length"] = 5
    raw["data"]["wisp_walks_per_anchor"] = 64
    point = dict(H2SNN_POINT)
    if dataset in H2GB_NC:
        raw["data"]["batch_size"] = 1024
        point["data.batch_size"] = 1024
    for name in STEP_DOWN:
        if name not in H2SNN_GRID:
            raise ValueError(f"step-down axis {name} is not an H2S-NN memory axis")
        section, key = name.split(".")
        raw[section][key] = point[name] = check_step_down(name, point[name], H2SNN_GRID[name])
    if task == "graph_classification":
        raw["model"]["graph_pooling"] = "attention"
        point["model.graph_pooling"] = "attention"
    raw["epochs"] = 1
    raw["device"] = "cuda"
    raw["exp_name"] = f"dryrun_h2snn_{dataset}"
    raw["output_dir"] = str(out / "runs")
    raw["use_wandb"] = True
    raw["wandb_project"] = WANDB_PROJECT
    raw["wandb_group"] = "d2"
    raw["wandb_tags"] = ["dry-run", "d2", task]
    return raw, point


def run_h2snn(task: str, dataset: str, out: Path, record: dict[str, Any]) -> None:
    from src.configs.config import TrainConfig
    from src.utils.misc import save_config, seed_everything

    raw, point = h2snn_raw(dataset, task, out)
    record["point"] = point
    cfg = TrainConfig.from_dict(raw)
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.exp_dir / "config.yaml")
    seed_everything(cfg.seed, cfg.device)
    record["resolved"] = {"epochs": cfg.epochs, "training_mode": cfg.data.training_mode,
                          "batch_size": cfg.data.batch_size, "epoch_targets": cfg.data.epoch_targets,
                          "eval_period": cfg.data.eval_period, "wisp_max_length": cfg.data.wisp_max_length,
                          "wisp_walks_per_anchor": cfg.data.wisp_walks_per_anchor, "d": cfg.model.d,
                          "solver_max_iterations": cfg.model.solver_max_iterations}
    install_pcg_counter()
    from src.builders import build_experiment_components

    with Phase("build"):
        data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    record["host_peak_after_build_gib"] = host_peak_gib()
    record["parameters"] = int(sum(p.numel() for p in model.parameters()))
    kind = type(trainer).__name__
    record["trainer"] = kind
    if kind == "MiniBatchNodeTrainer":
        wrap(trainer, "train_epoch", lambda a, k, i: "train_epoch")
        wrap(trainer, "evaluate_split", lambda a, k, i: "val_pass" if i == 0 else "test_pass")
    elif kind == "GraphClassificationTrainer":
        wrap(trainer, "_run_epoch", lambda a, k, i: "train_epoch" if k.get("train", a[2] if len(a) > 2 else None)
             else ("val_pass" if i == 1 else "test_pass"))
    elif kind == "LinkPredictionTrainer":
        wrap(trainer, "training_draw", lambda a, k, i: "train_epoch")
        wrap(trainer, "_train_epoch", lambda a, k, i: "train_epoch")
        wrap(trainer, "_eval_epoch", lambda a, k, i: "val_pass")
        wrap(trainer, "_eval_test_sets", lambda a, k, i: "test_pass")
    else:  # full-batch node classification Trainer
        wrap(trainer, "training_draw", lambda a, k, i: "train_epoch")
        wrap(trainer, "_train_epoch", lambda a, k, i: "train_epoch")
        wrap(trainer, "_eval_epoch", lambda a, k, i: "val_pass" if i == 0 else "test_pass")
    try:
        with Phase("fit"):
            trainer.fit(data)
    finally:
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.summary.update({f"dryrun/{k}": v for k, v in resource_summary(record).items()})
        except Exception as exc:  # the W&B summary is a convenience; the record below is the evidence
            record["wandb_summary_error"] = repr(exc)
        logger.close()


# ---------------------------------------------------------------------------------------- baselines
#: Size axes: the largest value is the largest memory point (card 2 Rule: hidden size, layers, heads, hops).
SIZE_AXES = {"hidden_dim", "hidden", "hidden_channels", "emb_dim", "embed_dim", "embed_size", "gnn_hidden",
             "num_layers", "layers", "gt_layers", "gnn_layers", "n_task_layers", "n_layers_2", "n_fp_layers",
             "num_heads", "attn_heads", "K", "num_hops", "num_label_hops", "num_gnns", "len_seq", "d", "batch_size"}
#: Switches that add memory: the heavier value.
HEAVY_SWITCH = {"sheaf_model": "GeneralSheaf", "gnn_type": "GAT", "label_feats": True, "metapath_pe": True,
                "feats_type": 3, "add_hp": True, "add_lp": True, "residual": True, "conformal": True,
                "batch_norm": True, "layer_norm": True}


def baseline_point(space: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    values, rule = {}, {}
    for name, spec in space.items():
        if "values" in spec:
            options = list(spec["values"])
            if name in SIZE_AXES:
                values[name], rule[name] = max(options), "max (size axis)"
            elif name in HEAVY_SWITCH and HEAVY_SWITCH[name] in options:
                values[name], rule[name] = HEAVY_SWITCH[name], "heavier switch"
            else:
                values[name], rule[name] = options[0], "first value (not a memory axis)"
        elif "value" in spec:
            values[name], rule[name] = spec["value"], "fixed"
        elif spec.get("distribution") == "log_uniform_values":
            values[name], rule[name] = math.sqrt(spec["min"] * spec["max"]), "geometric midpoint"
        elif spec.get("distribution") in ("uniform", "q_uniform"):
            values[name], rule[name] = (spec["min"] + spec["max"]) / 2, "midpoint"
        else:
            raise ValueError(f"unknown search-space entry {name}: {spec}")
    return values, rule


def run_baseline(model: str, task: str, dataset: str, out: Path, record: dict[str, Any]) -> None:
    import src.baselines.bundles as bundles
    from src.baselines.campaign import Cell, build_raw_base, fit_cell
    from src.baselines.registry import get_adapter
    from src.utils.wandb_campaign import CampaignUnit

    space = get_adapter(model).search_space(task, dataset)
    params, rule = baseline_point(space)
    for name in STEP_DOWN:
        if name not in space or "values" not in space[name]:
            raise ValueError(f"step-down axis {name} is not a grid axis of this cell")
        params[name] = check_step_down(name, params[name], list(space[name]["values"]))
        rule[name] = "one grid step below the max (card 2 step-down)"
    record["point"], record["point_rule"] = params, rule
    unit = CampaignUnit(run_index=0, num_runs=1, seed=0, label="run")
    raw_base = build_raw_base(Cell(model, task, dataset), unit, device="cuda", epochs=1)

    original_build = bundles.build_bundle

    def timed_build(*a, **k):
        with Phase("build"):
            return original_build(*a, **k)

    bundles.build_bundle = timed_build
    marks: dict[str, float] = {}
    original_eval = torch.nn.Module.eval

    def marked_eval(self):
        marks.setdefault("first_eval", time.perf_counter())
        return original_eval(self)

    torch.nn.Module.eval = marked_eval

    def report(*, epoch, monitor, running_best, metrics):  # timestamp only; values are never read
        marks.setdefault("first_report", time.perf_counter())
        marks["reports"] = marks.get("reports", 0) + 1

    import wandb

    run = wandb.init(project=WANDB_PROJECT, group="d2", name=f"dryrun_{model}_{task}_{dataset}",
                     tags=["dry-run", "d2", task], config={"cell": record["cell"], "point": params},
                     reinit=True, dir=str(out))
    try:
        with Phase("fit"):
            marks["fit_start"] = time.perf_counter()
            fit_cell(raw_base, params, source="tuned", device=torch.device("cuda"), report=report)
    finally:
        torch.nn.Module.eval = original_eval
        bundles.build_bundle = original_build
        build = STATE["times"].get("build", 0.0)
        if "first_eval" in marks:
            # from the end of the bundle build to the first `.eval()` call: model construction + the training epoch
            record["t_until_first_eval_s"] = marks["first_eval"] - marks["fit_start"] - build
        if "first_report" in marks and "first_eval" in marks:
            # first `.eval()` to the epoch report: validation (and, where the loop evaluates it there, test)
            record["t_first_eval_to_report_s"] = marks["first_report"] - marks["first_eval"]
        record["epoch_reports"] = int(marks.get("reports", 0))
        run.summary.update({f"dryrun/{k}": v for k, v in resource_summary(record).items()})
        run.finish()


# ---------------------------------------------------------------------------------------- main
def resource_summary(record: dict[str, Any]) -> dict[str, Any]:
    gpu = STATE.get("overall_gpu_peak", (0.0, 0.0))
    return {"status": record.get("status", "running"), "gpu_peak_alloc_gib": round(gpu[0], 3),
            "gpu_peak_reserved_gib": round(gpu[1], 3), "host_peak_gib": round(host_peak_gib(), 3),
            **{f"t_{k}_s": round(v, 2) for k, v in STATE["times"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempt", default="24GB")
    parser.add_argument("--step_down", default="", help="axis=value: one memory axis one grid step lower")
    args = parser.parse_args()
    if args.step_down:
        name, value = args.step_down.split("=")
        STEP_DOWN[name] = {"True": True, "False": False}.get(value, None)
        if STEP_DOWN[name] is None:
            STEP_DOWN[name] = int(value)
    model, task, dataset = args.cell.split("|")
    args.out.mkdir(parents=True, exist_ok=True)
    # Two workers may share a cell list (one from each end): the first to create the claim runs the cell.
    claims = args.out / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(claims / (args.cell.replace("|", "__") + f"__{args.attempt}.claim"),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        print(f"RESULT {args.cell} SKIP (claimed by another worker)", flush=True)
        return
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    record: dict[str, Any] = {
        "cell": args.cell, "model": model, "task": task, "dataset": dataset, "attempt": args.attempt,
        "step_down": dict(STEP_DOWN),
        "head": head, "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", ""),
        "array_job": os.environ.get("SLURM_ARRAY_JOB_ID", ""), "array_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "gpu": gpu, "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available() else 0, "python": platform.python_version(), "torch": torch.__version__}
    print(f"head={head} host={record['host']} job={record['job']} array={record['array_job']}_{record['array_task']} "
          f"gpu={gpu} cell={args.cell}", flush=True)
    start = time.perf_counter()
    try:
        if model == "h2snn":
            run_h2snn(task, dataset, args.out, record)
        else:
            run_baseline(model, task, dataset, args.out, record)
        record["status"] = "OK"
    except BaseException as exc:  # recorded as the cell's status (card 2 Rule), never hidden: see traceback_tail
        record.update(classify(exc))
    record["wall_s"] = time.perf_counter() - start
    gpu_peak = STATE.get("overall_gpu_peak", gpu_peak_gib())
    record["gpu_peak_alloc_gib"], record["gpu_peak_reserved_gib"] = gpu_peak
    record["host_peak_gib"] = host_peak_gib()
    record["times_s"] = STATE["times"]
    record["phase_calls"] = STATE["calls"]
    record["phase_gpu_peak_gib"] = STATE["phase_gpu_peak"]
    record["pcg"] = STATE["pcg"]
    name = args.cell.replace("|", "__") + f"__{args.attempt}.json"
    (args.out / "records").mkdir(parents=True, exist_ok=True)
    (args.out / "records" / name).write_text(json.dumps(record, indent=1, default=str))
    print(f"RESULT {args.cell} {record['status']} gpu_peak={record['gpu_peak_alloc_gib']:.2f}GiB "
          f"host_peak={record['host_peak_gib']:.2f}GiB times={ {k: round(v, 1) for k, v in STATE['times'].items()} }",
          flush=True)


if __name__ == "__main__":
    main()
