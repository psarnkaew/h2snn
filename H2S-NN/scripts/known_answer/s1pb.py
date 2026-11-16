"""Card 4 K3: known-answer S1'-B under v3-B (claim C12, v3-B capacity), as the FROZEN pre-registration
`docs/known_answer_preregistration_draft.md` (S1'-B; frozen settings of B-9, see `common.py`).

Graph `h5_synthetic.graph_s1p(n = 2000)` unchanged; l_max 2, r 640; arms `full` + oracle and
`covering_only` + oracle; beta in {0, log 64, log 512}; seeds 1-3 (18 runs, one per array task).
Per run: test accuracy (the pre-registered known-answer outcome on a synthetic task), relevant share of
D_P, dispersion, final tau, the dL/dbeta sign share (epochs before the best epoch with -dL/dbeta > 0).

    python scripts/known_answer/s1pb.py --task 0..17 --out_dir ../docs/card4_evidence/k3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import h5_synthetic as h5  # noqa: E402  (docs/WISP_evidence, unchanged)

N = 2000
ARMS = ("full", "covering_only")
BETA_LABELS = ("0", "log64", "log512")
SEEDS = (1, 2, 3)
TASKS = [(arm, beta, seed) for arm in ARMS for beta in BETA_LABELS for seed in SEEDS]


def run(arm: str, beta_label: str, seed: int, out_dir: Path) -> dict:
    graph, info = h5.graph_s1p(n=N)
    cfg = common.frozen_config(f"k3_s1pb_{arm}_b{beta_label}_s{seed}", seed, info["max_length"],
                               info["r"], info["target"], arm=arm)
    device = torch.device(cfg.device)
    state: dict = {}

    def oracle(model, data):
        state["relevant"] = common.relevance_by_group(data, info["carrier"], info["relevant_cpaper"])
        common.install_oracle(model, data, state["relevant"], common.BETAS[beta_label], device)

    def epoch_log(model):
        beta = model.oracle_beta
        row = {"dL_dbeta": None if beta.grad is None else float(beta.grad),
               "b": float(model.oracle_b.detach())}
        beta.grad = None
        return row

    data, model, log, best, seconds = common.train(cfg, graph, oracle, epoch_log)
    dev = data.to(device)
    relevant = state["relevant"]
    best_epoch = int(best["best_epoch"])
    before = [r["dL_dbeta"] for r in log[:best_epoch] if r["dL_dbeta"] is not None]
    record = {
        "task": "S1'-B", "arm": f"{arm} + oracle", "beta": beta_label, "seed": seed, "n": N,
        "test_acc": best["test_acc"], "val_acc": best["val_acc"], "best_epoch": best_epoch,
        "epochs_run": len(log), "seconds": seconds,
        "cells": {g.length: g.num_faces for g in data.face_groups},
        "relevant_cells": int(sum(int(r.sum()) for r in relevant)),
        "coverage_relevant": float(sum(int(r.sum()) for r in relevant)) / N,
        **common.relevant_share_of_degree(model, dev, relevant, arm),
        "dispersion": common.dispersion(model, dev, relevant, info["target"]),
        "tau_final": log[-1]["tau"], "tau_at_checkpoint": float(model.tau().detach()),
        "b_final": log[-1]["b"],
        "neg_dL_dbeta_positive_share_before_best": (sum(-g > 0 for g in before) / len(before)
                                                    if before else None),
        "epochs_before_best": len(before),
        **common.run_identity(),
        "per_epoch": log,
    }
    common.write_json(out_dir / f"s1pb_{arm}_b{beta_label}_s{seed}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, help=f"0..{len(TASKS) - 1}")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    arm, beta_label, seed = TASKS[args.task]
    record = run(arm, beta_label, seed, args.out_dir)
    print({k: record[k] for k in ("arm", "beta", "seed", "test_acc", "best_epoch", "seconds",
                                  "coverage_relevant", "relevant_share_D_median", "dispersion",
                                  "tau_final")}, flush=True)


if __name__ == "__main__":
    main()
