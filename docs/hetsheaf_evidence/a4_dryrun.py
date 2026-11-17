"""Card 3 A4: one epoch plus one validation pass per HetSheaf cell on real data, at the largest memory point of H-3
(`d` 5, `layers` 8, `hidden_channels` 64, general maps; learner `attention_type_ensemble`; batch 1024 on H2GB NC;
GC readout `late_token`; LP decoder `distmult`). Card 2's rule: status, peak GPU / host memory, wall time, GPU, node;
**no metric** (the adapter's record is discarded unread). One cell per subprocess (`--cell`), so an OOM or crash of
one cell does not affect the next and peak memory is per cell.

    python a4_dryrun.py --cells all --out cells.jsonl     # driver: one subprocess per cell
    python a4_dryrun.py --index 3 --out cell3.jsonl        # SLURM array: CELLS[3] (OOM -> one step-down run)
    python a4_dryrun.py --cell node_classification:acm     # one cell (used by the driver)

Bundles: the eight H2GB NC cells and ogbn-mag P-P cannot be built with `build_bundle(task, dataset)` alone (card 1d
H3b: `_base_config` -> `config.py:1023` needs `data.batch_size`; `config.py:924` / `:1159` have no task type or target relation for the large LP tasks).
This script passes the two config values as overrides (`data.batch_size` = the adapter's batch size 1024; `data.task_type`
= link_prediction and `data.target_relation` = the task's relation from `LARGE_LP_TASKS`); the adapter's protocol is unchanged (it trains with its own `batch_size` parameter).

Step-down (design session, A4 item 4): a cell that is OOM at the largest point runs once more with ONE memory axis one
grid step lower: `hidden_channels` 64 -> 32 (the OOM sites `sheaf_models.py:1238` and `disc_models.py:560` hold
per-edge tensors of width hidden_channels x (d + 2)).

One epoch: HGB-protocol cells take `epochs` = 1 (`budget_kwargs`); the H2GB NC and large-LP loops read
`default_budget`, which this script (only) replaces by a one-epoch budget; their epoch 0 always evaluates.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "H2S-NN"

CELLS = (
    [f"node_classification:{d}" for d in ("acm", "dblp", "imdb", "freebase", "ogbn-mag", "mag-year", "oag-cs",
                                           "oag-eng", "oag-chem", "ieee-cis-g", "h-pokec", "pdns")]
    + [f"link_prediction:{d}" for d in ("acm_pa", "dblp_ap", "imdb_ma", "freebase_bb", "pubmed", "lastfm",
                                        "ogbn-mag_pp")]
    + [f"graph_classification:{d}" for d in ("mutag", "proteins", "enzymes", "nci1")]
    + ["link_prediction:h-pokec_uu"]  # index 23: wired in A4 (author 2026-11-16), appended so 0-22 keep their index
)

LARGEST = {"sheaf": "general/attention_type_ensemble", "d": 5, "layers": 8, "hidden_channels": 64,
           "dropout": 0.5, "input_dropout": 0.5, "lr": 1e-3, "weight_decay": 1e-4}


def first_vendor_or_src_frame(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    for frame in reversed(frames):
        if "/H2S-NN/" in frame.filename and "site-packages" not in frame.filename:
            return f"{frame.filename.split('/H2S-NN/')[-1]}:{frame.lineno}"
    return f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "?"


STEP_DOWN = ("hidden_channels", 32)


def run_cell(cell: str, step_down: bool = False) -> dict:
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch

    from src.baselines import budget as budget_module
    from src.baselines import minibatch as minibatch_module
    from src.baselines.bundles import build_bundle
    from src.baselines.protocol import BaselineParams
    from src.baselines.registry import get_adapter, unsupported_reason

    task, dataset = cell.split(":")
    out = {"cell": cell, "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
           "step": os.environ.get("SLURM_STEP_ID"),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
           if torch.cuda.is_available() else 0}
    reason = unsupported_reason("hetsheaf", task, dataset)
    if reason:
        out.update(status="not runnable", error=reason)
        return out

    one_epoch = budget_module.TrainingBudget(epochs=1, patience=1)
    original = budget_module.default_budget
    budget_module.default_budget = lambda *a, **k: one_epoch  # large-LP loop (imports it at call time)
    minibatch_module.default_budget = lambda *a, **k: one_epoch  # H2GB NC plan

    values = dict(LARGEST, epochs=1, early_stopping=1)
    overrides: dict = {}
    if step_down:
        values[STEP_DOWN[0]] = STEP_DOWN[1]
        out["step_down"] = f"{STEP_DOWN[0]}={STEP_DOWN[1]}"
    if task == "node_classification" and dataset not in ("acm", "dblp", "imdb", "freebase"):
        values["batch_size"] = 1024
        overrides = {"data": {"batch_size": 1024}}  # config validation only (card 1d H3b); see the docstring
    if task == "link_prediction" and dataset in ("ogbn-mag_pp", "h-pokec_uu"):
        from src.data.large_lp import LARGE_LP_TASKS

        overrides = {"data": {"task_type": "link_prediction",  # config validation only; the bundle reads the data root
                              "target_relation": list(LARGE_LP_TASKS[dataset].relation)}}
    if task == "link_prediction":
        values["decoder_type"] = "distmult"
    if task == "graph_classification":
        values["pooling"] = "late_token"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    try:
        bundle = build_bundle(task, dataset, seed=0, overrides=overrides or None)
        load_s = time.time() - start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        fit_start = time.time()
        get_adapter("hetsheaf").fit_task(bundle, BaselineParams(values=values, source="tuned"), task=task, seed=0,
                                         device=device)  # record discarded: no metric is read or written
        out.update(status="OK", load_s=round(load_s, 1), epoch_and_eval_s=round(time.time() - fit_start, 1))
    except torch.cuda.OutOfMemoryError as exc:
        out.update(status="OOM", error=str(exc).split("\n")[0][:300], where=first_vendor_or_src_frame(exc))
    except Exception as exc:  # noqa: BLE001 - recorded per cell, the driver goes on
        status = "non-finite" if "finite" in str(exc).lower() else "error"
        out.update(status=status, error=f"{type(exc).__name__}: {str(exc)[:300]}", where=first_vendor_or_src_frame(exc))
    finally:
        budget_module.default_budget = original
    if torch.cuda.is_available():
        out["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out["peak_host_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    out["total_s"] = round(time.time() - start, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell")
    parser.add_argument("--step-down", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--cells", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.cell:
        print("RESULT " + json.dumps(run_cell(args.cell, args.step_down)), flush=True)
        return
    if args.index is not None:
        result = one(CELLS[args.index], False, args.out)
        if result["status"] == "OOM":
            one(CELLS[args.index], True, args.out)
        return
    cells = CELLS if args.cells == "all" else args.cells.split(",")
    for cell in cells:
        one(cell, False, args.out)


def one(cell: str, step_down: bool, out_path: str) -> dict:
    command = [sys.executable, __file__, "--cell", cell] + (["--step-down"] if step_down else [])
    proc = subprocess.run(command, capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if lines:
        result = json.loads(lines[-1][7:])
    else:  # killed (e.g. host OOM): the subprocess wrote no result
        result = {"cell": cell, "status": "killed", "returncode": proc.returncode,
                  "error": proc.stderr.strip().splitlines()[-1][:300] if proc.stderr.strip() else ""}
    if step_down:
        result.setdefault("step_down", f"{STEP_DOWN[0]}={STEP_DOWN[1]}")
    print(json.dumps(result), flush=True)
    if out_path:
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    main()
