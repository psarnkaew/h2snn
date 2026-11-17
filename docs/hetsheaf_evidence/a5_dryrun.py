"""Card 3 A5 (added 2026-11-17 after the A4 review): HetSheaf OOM ladder (A5.2) and central-point timing (A5.3).
No metric: the adapter's record is discarded unread (A4's Rule). One cell per SLURM job.

    python a5_dryrun.py --ladder 3 --out a5_ladder/cell_3.jsonl     # LADDER_CELLS[3]: points in order, stop at first OK
    python a5_dryrun.py --central 5 --out a5_central/cell_5.jsonl   # CENTRAL_CELLS[5]: one epoch + one validation pass
    python a5_dryrun.py --cell <task:dataset> --values '<json>' --kind ladder|central   # one point (used by the above)

A5.2 ladder (decision 1, as card 2 D6.3): from A4's point (`d` 5, `layers` 8, `hidden_channels` 64, general maps,
learner `attention_type_ensemble`, LP decoder `distmult`, ...) one grid step at a time on 24 GB, axis order
`hidden_channels` 32 -> 16 -> 8, then `d` 5 -> 4 -> 3 -> 2, then `layers` 8 -> 7 -> ... -> 2. The 7 cells were OOM at
`hidden_channels` 64 and 32 in A4 (`a4_cells/`), so the ladder starts at `hidden_channels` 16. Stop at the first point
that fits (status OK); a non-OOM failure also ends the ladder (recorded, not stepped past).

A5.3 central point (decision B2a, as card 2 D6.4; card 2's `central_point` rule, `dry-run:docs/dryrun_evidence/
d6_dryrun.py`): numeric grid axes at the middle value (lower middle for an even count), log-uniform axes at the
geometric middle, uniform axes at the middle, categoricals at the adapter's default: its published value for the cell
(none exists for these 12 cells), else the value the adapter uses when the parameter is absent (`sheaf`
`diag/type_concat`, `hetsheaf.py:401`; GC `pooling` `late_token`, `hetsheaf.py:620`; design session: "categoricals,
incl. `sheaf`, at the adapter's default"), else the grid's first value (card 2's fallback; not reached here). Computed here from `HetSheafAdapter.search_space` / `published_params`; asserted to give `d` 3, `layers` 5,
`hidden_channels` 16 (card A5.3). Timing (perf_counter, CUDA synchronised at each stamp):
  train_epoch_s  last `Module.train(True)` before the first `Module.eval()` -> that `eval()` (the epoch's training steps)
  val_s          H2GB NC: the first `minibatch.evaluate_split` call (the whole validation split, each target once);
                 GC: first `eval()` -> first `compute_graph_classification_metrics` call (the validation `run_split`)
  test_s         the same for the test pass (second call)
  val_targets    H2GB NC: `ids.numel()` of the first `evaluate_split`; GC: graphs in the validation loader

Bundles and the one-epoch budget as A4 (`a4_dryrun.py`, imported): `data.batch_size` / large-LP `task_type` and
`target_relation` are config-validation overrides only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a4_dryrun as a4  # noqa: E402

ROOT = a4.ROOT

LADDER_CELLS = ["node_classification:acm", "node_classification:freebase", "link_prediction:acm_pa",
                "link_prediction:freebase_bb", "link_prediction:pubmed", "link_prediction:ogbn-mag_pp",
                "link_prediction:h-pokec_uu"]
CENTRAL_CELLS = ([f"node_classification:{d}" for d in ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem",
                                                         "ieee-cis-g", "h-pokec", "pdns")]
                 + [f"graph_classification:{d}" for d in ("mutag", "proteins", "enzymes", "nci1")])
#: The adapter's value when the parameter is absent (`hetsheaf.py:401` / `:424`, `:620`).
ADAPTER_CODE_DEFAULTS = {"sheaf": "diag/type_concat", "pooling": "late_token"}
H2GB_NC = {"ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns"}


def ladder_points() -> list[dict]:
    """A4's point with the A5.2 axes stepped; starts after A4's recorded `hidden_channels` 64 and 32 (both OOM)."""
    points = [{"hidden_channels": 16, "d": 5, "layers": 8}, {"hidden_channels": 8, "d": 5, "layers": 8}]
    points += [{"hidden_channels": 8, "d": d, "layers": 8} for d in (4, 3, 2)]
    points += [{"hidden_channels": 8, "d": 2, "layers": layers} for layers in (7, 6, 5, 4, 3, 2)]
    return points


def middle(options: list) -> object:
    ordered = sorted(options)
    return ordered[(len(ordered) - 1) // 2]


def central_values(task: str, dataset: str) -> tuple[dict, dict]:
    """Card 2's `central_point` (d6_dryrun.py) over HetSheaf's own search space."""
    from src.baselines.registry import get_adapter

    adapter = get_adapter("hetsheaf")
    space = adapter.search_space(task, dataset)
    published = adapter.published_params(task, dataset)
    default = dict(getattr(published, "values", {}) or {}) if published is not None else {}
    values, rule = {}, {}
    for name, spec in space.items():
        if "values" in spec:
            options = list(spec["values"])
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in options):
                values[name], rule[name] = middle(options), "middle of the grid"
            elif name in default and default[name] in options:
                values[name], rule[name] = default[name], "categorical: published value"
            elif ADAPTER_CODE_DEFAULTS.get(name) in options:
                values[name], rule[name] = ADAPTER_CODE_DEFAULTS[name], "categorical: adapter code default"
            else:
                values[name], rule[name] = options[0], "categorical: first grid value"
        elif spec.get("distribution") == "log_uniform_values":
            values[name], rule[name] = math.sqrt(spec["min"] * spec["max"]), "geometric middle"
        elif spec.get("distribution") in ("uniform", "q_uniform"):
            values[name], rule[name] = (spec["min"] + spec["max"]) / 2, "middle"
        else:
            raise ValueError(f"unknown search-space entry {name}: {spec}")
    assert (values["d"], values["layers"], values["hidden_channels"]) == (3, 5, 16), values  # card A5.3
    return values, rule


TIMES: dict = {}


def install_timers(torch) -> None:
    import src.baselines.minibatch as minibatch
    import src.utils.misc as misc

    def stamp() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    orig_train, orig_eval = torch.nn.Module.train, torch.nn.Module.eval

    def train(self, mode=True):
        if mode and "first_eval" not in TIMES:
            TIMES["train_start"] = stamp()
        return orig_train(self, mode)

    def eval_(self):
        if "train_start" in TIMES and "first_eval" not in TIMES:
            TIMES["first_eval"] = stamp()
        return orig_eval(self)

    torch.nn.Module.train, torch.nn.Module.eval = train, eval_

    orig_split = minibatch.evaluate_split

    def evaluate_split(loader, forward_seeds, target_type, ids, *a, **k):
        n = TIMES.setdefault("split_calls", 0)
        TIMES["split_calls"] = n + 1
        start = stamp()
        out = orig_split(loader, forward_seeds, target_type, ids, *a, **k)
        if n < 2:
            TIMES["val_s" if n == 0 else "test_s"] = stamp() - start
            if n == 0:
                TIMES["val_targets"] = int(ids.numel())
        return out

    minibatch.evaluate_split = evaluate_split

    orig_metrics = misc.compute_graph_classification_metrics

    def gc_metrics(*a, **k):
        TIMES.setdefault("metric_calls", []).append(stamp())
        return orig_metrics(*a, **k)

    misc.compute_graph_classification_metrics = gc_metrics


def run_point(cell: str, point: dict, kind: str) -> dict:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch

    from src.baselines import budget as budget_module
    from src.baselines import minibatch as minibatch_module
    from src.baselines.bundles import build_bundle
    from src.baselines.protocol import BaselineParams
    from src.baselines.registry import get_adapter

    task, dataset = cell.split(":")
    out = {"cell": cell, "kind": kind, "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
           if torch.cuda.is_available() else 0}
    one_epoch = budget_module.TrainingBudget(epochs=1, patience=1)
    budget_module.default_budget = lambda *a, **k: one_epoch  # large-LP loop (imports it at call time)
    minibatch_module.default_budget = lambda *a, **k: one_epoch  # H2GB NC plan

    overrides: dict = {}
    if kind == "ladder":
        values = dict(a4.LARGEST, epochs=1, early_stopping=1)
        if task == "link_prediction":
            values["decoder_type"] = "distmult"
        values.update(point)
        out["point"] = point
    else:
        values, rule = central_values(task, dataset)
        values.update(epochs=1, early_stopping=1)
        out["point"] = {k: v for k, v in values.items() if k not in ("epochs", "early_stopping")}
        out["rule"] = rule
    if task == "node_classification" and dataset in H2GB_NC:
        overrides = {"data": {"batch_size": int(values["batch_size"])}}  # config validation only (card 1d H3b)
    if task == "link_prediction" and dataset in ("ogbn-mag_pp", "h-pokec_uu"):
        from src.data.large_lp import LARGE_LP_TASKS

        overrides = {"data": {"task_type": "link_prediction",  # config validation only
                              "target_relation": list(LARGE_LP_TASKS[dataset].relation)}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    try:
        bundle = build_bundle(task, dataset, seed=0, overrides=overrides or None)
        out["load_s"] = round(time.time() - start, 1)
        if kind == "central":
            install_timers(torch)
            if task == "graph_classification":
                TIMES["val_targets"] = len(bundle.loaders[1].dataset)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        fit_start = time.time()
        get_adapter("hetsheaf").fit_task(bundle, BaselineParams(values=values, source="tuned"), task=task, seed=0,
                                         device=device)  # record discarded: no metric is read or written
        out.update(status="OK", epoch_and_eval_s=round(time.time() - fit_start, 1))
    except torch.cuda.OutOfMemoryError as exc:
        out.update(status="OOM", error=str(exc).split("\n")[0][:300], where=a4.first_vendor_or_src_frame(exc))
    except Exception as exc:  # noqa: BLE001 - recorded per point
        status = "non-finite" if "finite" in str(exc).lower() else "error"
        out.update(status=status, error=f"{type(exc).__name__}: {str(exc)[:300]}",
                   where=a4.first_vendor_or_src_frame(exc))
    if kind == "central":
        if "train_start" in TIMES and "first_eval" in TIMES:
            out["train_epoch_s"] = round(TIMES["first_eval"] - TIMES["train_start"], 2)
        calls = TIMES.get("metric_calls", [])
        if task == "graph_classification" and "first_eval" in TIMES and calls:
            TIMES["val_s"] = calls[0] - TIMES["first_eval"]
            if len(calls) >= 2:
                TIMES["test_s"] = calls[1] - calls[0]
        for key in ("val_s", "test_s"):
            if key in TIMES:
                out[key] = round(TIMES[key], 2)
        out["val_targets"] = TIMES.get("val_targets")
    if torch.cuda.is_available():
        out["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out["peak_host_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    out["total_s"] = round(time.time() - start, 1)
    return out


def one(cell: str, point: dict, kind: str, out_path: str) -> dict:
    command = [sys.executable, __file__, "--cell", cell, "--values", json.dumps(point), "--kind", kind]
    proc = subprocess.run(command, capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if lines:
        result = json.loads(lines[-1][7:])
    else:  # killed (e.g. host OOM): the subprocess wrote no result
        result = {"cell": cell, "kind": kind, "point": point, "status": "killed", "returncode": proc.returncode,
                  "error": proc.stderr.strip().splitlines()[-1][:300] if proc.stderr.strip() else ""}
    print(json.dumps(result), flush=True)
    if out_path:
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell")
    parser.add_argument("--values", default="{}")
    parser.add_argument("--kind", choices=("ladder", "central"))
    parser.add_argument("--ladder", type=int)
    parser.add_argument("--central", type=int)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.cell:
        print("RESULT " + json.dumps(run_point(args.cell, json.loads(args.values), args.kind)), flush=True)
        return
    if args.ladder is not None:
        for point in ladder_points():
            if one(LADDER_CELLS[args.ladder], point, "ladder", args.out)["status"] != "OOM":
                return
        print("LADDER BOTTOM REACHED: still OOM (stop item)", flush=True)
        return
    if args.central is not None:
        one(CENTRAL_CELLS[args.central], {}, "central", args.out)


if __name__ == "__main__":
    main()
