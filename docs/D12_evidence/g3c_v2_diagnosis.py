"""G3c diagnosis D1 (addendum in g3c_preregistration.md), post hoc, no training: mean w_f of the v2
full arm's selected checkpoint by cell category, for the cells of train targets and of test targets.
Categories of a target cell (P, A, Q): relevant = (lead A, key(Q) = key(P)); distractor = (non-lead,
match); lead_no_match; non_lead_no_match.
Run from H2S-NN/:  python ../docs/D12_evidence/g3c_v2_diagnosis.py [--seeds 42 43 44]   (CPU)
Writes docs/D12_evidence/g3c_v2_diagnosis.json."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "docs" / "D12_evidence"))
import g3c_binding as g3c  # noqa: E402
import src.data.dataset as dataset_module  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

KEY = slice(2, 2 + g3c.KEY_BITS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    graph, truth, _ = g3c.graph_v2()
    dataset_module._load_hetero_data = lambda _config: graph.clone()
    paper_x = graph["paper"].x
    out = {}
    for seed in args.seeds:
        cfg = g3c.config("v2", "full", seed)
        data = build_dataset(cfg)
        model = build_model(cfg, data)
        model.load_state_dict(torch.load(cfg.exp_dir / "checkpoints" / "best.pt", map_location="cpu")["state_dict"])
        group = next(grp for grp in data.face_groups if grp.template_name == "PAP")
        weights = g3c.path_weights(model, data, group)
        paper0, author0 = data.node_slices["paper"][0], data.node_slices["author"][0]
        nodes = group.face_node_index - torch.tensor([paper0, author0, paper0])  # local ids
        first_is_target = nodes[:, 0] < g3c.NT
        last_is_target = nodes[:, 2] < g3c.NT
        on_target = first_is_target | last_is_target
        assert not bool((first_is_target & last_is_target).any()), "no target-target cells in v2"
        target = torch.where(first_is_target, nodes[:, 0], nodes[:, 2])[on_target]
        other = torch.where(first_is_target, nodes[:, 2], nodes[:, 0])[on_target]
        lead = (nodes[:, 1] < g3c.NT)[on_target]
        match = (paper_x[target, KEY] == paper_x[other, KEY]).all(dim=1)
        w = weights[on_target]
        masks = {"train": data.target_train_mask, "test": data.target_test_mask}
        entry = {}
        for split, mask in masks.items():
            in_split = mask[target]
            categories = {"relevant": lead & match, "distractor": ~lead & match,
                          "lead_no_match": lead & ~match, "non_lead_no_match": ~lead & ~match}
            entry[split] = {name: {"mean_w": float(w[in_split & sel].mean()), "cells": int((in_split & sel).sum())}
                            for name, sel in categories.items()}
            # Per target: rank of the relevant cell among the target's 32 cells (1 = highest w_f).
            ranks = []
            for t in mask.nonzero().flatten().tolist():
                cells = target == t
                rel = cells & lead & match
                ranks.append(int((w[cells] > w[rel].max()).sum()) + 1)
            ranks_t = torch.tensor(ranks, dtype=torch.float)
            entry[split]["relevant_rank_among_own_cells"] = {
                "mean": float(ranks_t.mean()), "top1_fraction": float((ranks_t == 1).float().mean()),
                "cells_per_target": int((target == mask.nonzero().flatten()[0]).sum())}
        out[f"v2/full/s{seed}"] = entry
        print(seed, json.dumps(entry), flush=True)
    (ROOT.parent / "docs" / "D12_evidence" / "g3c_v2_diagnosis.json").write_text(json.dumps(out, indent=1),
                                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
