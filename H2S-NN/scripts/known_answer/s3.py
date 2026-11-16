"""Card 4 K4: known-answer S3 runs (learnability of g), FROZEN pre-registration and settings
(`common.py`). Runs only after `s3_audit.py` passed (its JSON is read; a failed audit refuses to run).

Graph `s3_graph.graph_s3(n = 2000)`; l_max 4; r = the audit's chosen r (coverage 1.0 of the 64
length-4 cells per target on the seeds' evaluation draws); arms `full`, `endpoints`, `uniform`, `full` +
oracle (beta = log 512); seeds 1-3 (12 runs, one per array task). Per run: test accuracy and the mean
learned log-weight gap relevant - irrelevant, over the length-4 cells (the 64 per target; primary) and
over all cells.

    python scripts/known_answer/s3.py --task 0..11 --audit ../docs/card4_evidence/k4/s3_audit.json \
        --out_dir ../docs/card4_evidence/k4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import s3_graph  # noqa: E402

N = 2000
ARMS = ("full", "endpoints", "uniform", "full_oracle")
SEEDS = (1, 2, 3)
TASKS = [(arm, seed) for arm in ARMS for seed in SEEDS]


def gap_by_length(model, dev, relevant) -> dict:
    h0, geometry, _ = common.model_geometry(model, dev)
    groups, _, weights = common.cell_weights(model, dev, h0, geometry)
    out = {}
    for group, w, rel in zip(groups, weights, relevant):
        if group.length != 4:
            continue
        log_w = w.log()
        rel = rel.to(log_w.device)
        out = {"log_w_relevant_len4": float(log_w[rel].mean()),
               "log_w_irrelevant_len4": float(log_w[~rel].mean()),
               "log_w_gap_len4": float(log_w[rel].mean() - log_w[~rel].mean())}
    return out


def run(arm: str, seed: int, r: int, out_dir: Path) -> dict:
    graph, info = s3_graph.graph_s3(n=N)
    ablation = {"path_weight_input": arm} if arm in ("endpoints", "uniform") else {}
    cfg = common.frozen_config(f"k4_s3_{arm}_s{seed}", seed, info["max_length"], r, info["target"],
                               ablation=ablation)
    device = torch.device(cfg.device)
    state: dict = {}

    def prepare(model, data):
        state["relevant"] = common.relevance_by_group(data, info["carrier"], info["relevant_cpaper"])
        if arm == "full_oracle":
            common.install_oracle(model, data, state["relevant"], common.BETAS["log512"], device)

    data, model, log, best, seconds = common.train(cfg, graph, prepare)
    dev = data.to(device)
    relevant = state["relevant"]
    record = {
        "task": "S3", "arm": "full + oracle (log 512)" if arm == "full_oracle" else arm,
        "seed": seed, "n": N, "r": r,
        "test_acc": best["test_acc"], "val_acc": best["val_acc"], "best_epoch": int(best["best_epoch"]),
        "epochs_run": len(log), "seconds": seconds,
        "cells": {g.length: g.num_faces for g in data.face_groups},
        "coverage_relevant": float(sum(int(x.sum()) for x in relevant)) / N,
        **gap_by_length(model, dev, relevant),
        **common.log_weight_gap(model, dev, relevant),
        "tau_final": log[-1]["tau"],
        **common.run_identity(),
        "per_epoch": log,
    }
    common.write_json(out_dir / f"s3_{arm}_s{seed}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, help=f"0..{len(TASKS) - 1}")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit.get("audit_pass"):
        raise SystemExit("the S3 audit did not pass: no S3 run (card K4: stop the unit).")
    arm, seed = TASKS[args.task]
    record = run(arm, seed, int(audit["r_choice"]["chosen_r"]), args.out_dir)
    print({k: record.get(k) for k in ("arm", "seed", "test_acc", "best_epoch", "seconds",
                                      "coverage_relevant", "log_w_gap_len4", "tau_final")},
          flush=True)


if __name__ == "__main__":
    main()
