"""Card 4 K5: known-answer S2' (C2b read-out), option (a), FROZEN pre-registration and settings
(`common.py`). K2b's graph `i4_s2_unconfounded.graph_s2u(n = 2000)` unchanged, l_max 4, r = 200
(coverage 1.0 of sigma* in Part K; measured again here), arms `full` and `uniform`, seeds 1-3 (6 runs).
Read-out on the evaluation draw at the selected checkpoint (I4's `readout`, ported to v3-B: w_f from
the model's g): sigma* = the schema of the planted chains T-M1-M2-M3-C*; criterion (fixed): sigma* has
the STRICTLY highest mean learned log-weight per cell among the full-walk schemas (ties count against
it) on 3/3 `full` seeds. `uniform`'s accuracy is reported as "selection not required on this graph".

    python scripts/known_answer/s2p.py --task 0..5 --out_dir ../docs/card4_evidence/k5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import i4_s2_unconfounded as i4  # noqa: E402  (docs/WISP_evidence, unchanged)
from src.analysis.wisp_readouts import full_schema_keys, i_pi, mean_log_weight  # noqa: E402

N = 2000
R = 200
ARMS = ("full", "uniform")
SEEDS = (1, 2, 3)
TASKS = [(arm, seed) for arm in ARMS for seed in SEEDS]


def label(key, names, rel_names) -> str:
    parts = [names[key[0]]]
    for k in range(1, len(key), 3):
        rel = rel_names[key[k]]
        rel = rel.split("__")[1] if "__" in rel else rel
        parts.append(f"{rel}{'+-~'[[1, -1, 0].index(key[k + 1] - 1)]}")
        parts.append(names[key[k + 2]])
    return "-".join(parts)


def readout(model, data, dev, info) -> dict:
    h0, geometry, _ = common.model_geometry(model, dev)
    groups_dev, _, weights_dev = common.cell_weights(model, dev, h0, geometry)
    groups = [g for g in data.face_groups if g.num_faces]
    weights = [w.detach().cpu() for w in weights_dev]
    si = data.fact_layer.self_inverse
    mlw = mean_log_weight(groups, weights, data.node_type, si)
    pi = i_pi(groups, weights, data.node_type, si)
    pi0 = i_pi(groups, [torch.ones_like(w) for w in weights], data.node_type, si)
    n = info["n"]
    chain = torch.stack([data.node_slices[t][0] + torch.arange(n)
                         for t in ("target", "m1", "m2", "m3")]
                        + [data.node_slices["carrier"][0] + torch.arange(n)], 1)
    planted, covered = set(), set()
    for group in groups:
        if group.face_node_index.size(1) != 5:
            continue
        keys = full_schema_keys(group, data.node_type, si)
        rows = group.face_node_index.cpu()
        for orient in (rows, rows.flip(1)):
            local = orient[:, 0] - data.node_slices["target"][0]
            inside = (local >= 0) & (local < n)
            match = inside.clone()
            match[inside] = (orient[inside] == chain[local[inside]]).all(1)
            for f in match.nonzero().flatten().tolist():
                planted.add(keys[f])
                covered.add(int(local[f]))
    if len(planted) != 1:
        raise ValueError(f"the planted chains map to {len(planted)} schemas")
    star = planted.pop()
    names, rel_names = data.node_type_names, data.relation_names
    length_of = {k: (len(k) - 1) // 3 for k in mlw}
    rank = i4.strict_rank(mlw, star)
    ordered = sorted(mlw, key=lambda k: -mlw[k])
    runner_up = next(k for k in ordered if k != star)
    return {"sigma_star": label(star, names, rel_names), "planted_coverage": len(covered) / n,
            "num_full_schemas": len(mlw),
            "num_length4_schemas": sum(1 for k in mlw if length_of[k] == 4),
            "strict_rank_mean_log_weight": rank, "criterion_pass": rank == 1,
            "mean_log_w_star": mlw[star], "mean_log_w_best_other": mlw[runner_up],
            "best_other": label(runner_up, names, rel_names),
            "gap_star_minus_best_other": mlw[star] - mlw[runner_up],
            "schemas": sorted(({"schema": label(k, names, rel_names), "length": length_of[k],
                                "mean_log_w": mlw[k], "I_pi": pi[k], "I_pi_init": pi0[k]}
                               for k in mlw), key=lambda r: -r["mean_log_w"])}


def run(arm: str, seed: int, out_dir: Path) -> dict:
    graph, info = i4.graph_s2u(N)
    ablation = {"path_weight_input": "uniform"} if arm == "uniform" else {}
    cfg = common.frozen_config(f"k5_s2p_{arm}_s{seed}", seed, info["max_length"], R, info["target"],
                               ablation=ablation)
    data, model, log, best, seconds = common.train(cfg, graph)
    dev = data.to(torch.device(cfg.device))
    record = {"task": "S2'", "arm": arm, "seed": seed, "n": N, "r": R,
              "test_acc": best["test_acc"], "val_acc": best["val_acc"],
              "best_epoch": int(best["best_epoch"]), "epochs_run": len(log), "seconds": seconds,
              "cells": {g.length: g.num_faces for g in data.face_groups},
              "readout": readout(model, data, dev, info), "tau_final": log[-1]["tau"],
              **common.run_identity(), "per_epoch": log}
    common.write_json(out_dir / f"s2p_{arm}_s{seed}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, help=f"0..{len(TASKS) - 1}")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    arm, seed = TASKS[args.task]
    record = run(arm, seed, args.out_dir)
    ro = record["readout"]
    print({"arm": arm, "seed": seed, "test_acc": record["test_acc"],
           "planted_coverage": ro["planted_coverage"], "rank": ro["strict_rank_mean_log_weight"],
           "gap": ro["gap_star_minus_best_other"], "best_other": ro["best_other"]}, flush=True)


if __name__ == "__main__":
    main()
