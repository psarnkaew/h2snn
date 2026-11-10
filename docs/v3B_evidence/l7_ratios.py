"""L7: v3-B / v2 H6 (and v1 ACM) cost ratios from the smoke JSONs. Run from the repository root:

    python docs/v3B_evidence/l7_ratios.py

v2 H6 values (peak MB, epoch s, train s, draw s) from docs/H2S-NN_WISP_report.md H6 table.
"""

import json

V2_H6 = {"acm": (1661, 2.82, 1.40, 0.94), "dblp": (2705, 3.87, 1.36, 2.00), "imdb": (889, 1.33, 0.63, 0.40)}
V1_ACM = (9260, 3.0)
FILES = {
    "acm": "docs/v3B_evidence/v3b_cost_smoke_acm.json",
    "dblp": "docs/v3B_evidence/v3b_cost_smoke_dblp.json",
    "imdb": "docs/v3B_evidence/v3b_cost_smoke_imdb_l7.json",
    "imdb_l5": "docs/v3B_evidence/v3b_cost_smoke_imdb.json",
}

for key, path in FILES.items():
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    epoch = record["epochs"][1]
    mem, epoch_s, train_s, draw_s = V2_H6[key.split("_")[0]]
    solve = epoch["eval_solve"]
    rows = record["num_nodes"] + solve["num_fact_states"] + solve["num_cells"]
    print(key, "rows", rows, "eliminated", round(solve["eliminated_share"], 3),
          "E_shared/E_P", round(solve["num_shared_facts"] / solve["num_path_facts"], 3))
    print("  epoch1 mem", epoch["peak_memory_mb"], round(epoch["peak_memory_mb"] / mem, 2),
          "epoch", epoch["epoch_seconds"], round(epoch["epoch_seconds"] / epoch_s, 2),
          "train", epoch["train_step_seconds"], round(epoch["train_step_seconds"] / train_s, 2),
          "draw", epoch["draw_seconds"], round(epoch["draw_seconds"] / draw_s, 2))
    step = record.get("final_tolerance_step")
    if step:
        total = step["draw_seconds"] + step["forward_backward_seconds"]
        print("  tol1e-4 mem", step["peak_memory_mb_this_step"],
              round(step["peak_memory_mb_this_step"] / mem, 2),
              "draw+fwd/bwd", round(total, 3), round(total / epoch_s, 2),
              "fwd/bwd", step["forward_backward_seconds"],
              round(step["forward_backward_seconds"] / train_s, 2),
              "pcg", step["train_solve"]["pcg_fwd_iters"], step["train_solve"]["pcg_bwd_iters"])
    if key == "acm":
        print("  vs v1 mem", round(epoch["peak_memory_mb"] / V1_ACM[0], 3),
              "time", round(epoch["epoch_seconds"] / V1_ACM[1], 2))
