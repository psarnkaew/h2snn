"""K2b (Part K): S2 (the I4 graph) with 1,000 training targets (pre-registered:
k_preregistration.md, K2b). Run from H2S-NN/:
    python ../docs/WISP_evidence/k2b_s2_n2000.py --choose-r          (set-up: r from the draws only)
    python ../docs/WISP_evidence/k2b_s2_n2000.py --r R --arms full --seeds 42 43 44
    python ../docs/WISP_evidence/k2b_s2_n2000.py --r R --arms uniform --seeds 42
i4_s2_unconfounded.graph_s2u(n=2000) (split 50/25/25 -> 1,000 training targets); I4's config and
read-out (`i4.readout`, strict rank) unchanged; log_every 1000. r is chosen BEFORE training so that
sigma* is drawn for every target in each seed's evaluation draw (coverage 1.0, checked on the draw).
Per epoch (evaluation draw, eval mode, no gradient): train/val accuracy and the mean log-weight of
sigma* cells minus a b c e cells. Writes docs/WISP_evidence/k2b_results.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "docs" / "WISP_evidence"))
import h5_synthetic as h5  # noqa: E402
import i4_s2_unconfounded as i4  # noqa: E402
import src.data.dataset as dataset_module  # noqa: E402
import src.data.wisp_sampler  # noqa: E402,F401
import src.trainers.trainer  # noqa: E402,F401
from src.builders import build_experiment_components  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "WISP_evidence" / "k2b_results.json"
N = 2000


def chain_masks(data, n):
    """Per non-empty group: bool masks of sigma* cells (T_i M1_i M2_i M3_i C*_i) and of a b c e
    cells (T_i M1_i M2_i M3_i Ce_i), either orientation; and the covered targets of sigma*."""
    s = data.node_slices
    base = torch.stack([s[t][0] + torch.arange(n) for t in ("target", "m1", "m2", "m3")], 1)
    star_chain = torch.cat([base, (s["carrier"][0] + torch.arange(n)).unsqueeze(1)], 1)
    sib_chain = torch.cat([base, (s["carrier"][0] + n + torch.arange(n)).unsqueeze(1)], 1)
    star, sib, covered = [], [], set()
    for group in (g for g in data.face_groups if g.num_faces):
        rows = group.face_node_index.cpu()
        masks = []
        for chain in (star_chain, sib_chain):
            hit = torch.zeros(rows.size(0), dtype=torch.bool)
            if rows.size(1) == 5:
                for orient in (rows, rows.flip(1)):
                    local = orient[:, 0] - s["target"][0]
                    inside = (local >= 0) & (local < n)
                    match = inside.clone()
                    match[inside] = (orient[inside] == chain[local[inside]]).all(1)
                    hit |= match
            masks.append(hit)
        star.append(masks[0])
        sib.append(masks[1])
        if rows.size(1) == 5:
            for orient in (rows, rows.flip(1)):
                local = orient[:, 0] - s["target"][0]
                inside = (local >= 0) & (local < n)
                ok = inside.clone()
                ok[inside] = (orient[inside] == star_chain[local[inside]]).all(1)
                covered.update(local[ok].tolist())
    return star, sib, covered


def choose_r(graph, info, seeds=(42, 43, 44), ladder=(200, 300, 400, 600, 800, 1200)):
    """Set-up only: the smallest r on the ladder whose evaluation draws (seed = run seed) cover
    sigma* for every target."""
    record = {}
    for r in ladder:
        info = dict(info, r=r)
        coverage = {}
        for seed in seeds:
            cfg = h5.config(f"s2u_k2b_rcheck_n{N}", "full", seed, info, 1)
            data = build_dataset(cfg)
            _, _, covered = chain_masks(data, N)
            coverage[seed] = len(covered) / N
        record[r] = coverage
        print("r", r, "coverage", coverage, flush=True)
        if all(c == 1.0 for c in coverage.values()):
            return r, record
    raise SystemExit(f"no r on the ladder covers every target: {record}")


def gap(model, dev, star, sib):
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            h0 = model.encode_nodes(dev.x, dev.node_type)
            op = model.build_operator(h0, dev.edge_index, dev.edge_type, dev.face_groups,
                                      dev.num_nodes)
    finally:
        model.train(was)
    log_w = torch.cat([w.double().log() for w in op.path_weights]).cpu()
    star_all, sib_all = torch.cat(star), torch.cat(sib)
    return float(log_w[star_all].mean() - log_w[sib_all].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--choose-r", action="store_true")
    parser.add_argument("--r", type=int)
    parser.add_argument("--arms", nargs="+", default=["full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    graph, info = i4.graph_s2u(N)
    dataset_module._load_hetero_data = lambda _config: graph.clone()
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if args.choose_r:
        r, record = choose_r(graph, info)
        results["r_choice"] = {"r": r, "coverage_by_r_and_seed": record}
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print("chosen r", r, flush=True)
        return
    info = dict(info, r=args.r)
    for arm in args.arms:
        for seed in args.seeds:
            cfg = h5.config(f"s2u_k2b_n{N}", arm, seed, info, args.epochs)
            cfg.log_every = 1000
            if (cfg.exp_dir / "metrics.jsonl").exists():
                raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
            cfg.exp_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed, cfg.device)
            start = time.time()
            data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
            star, sib, covered = chain_masks(data, N)
            dev = data.to(torch.device(cfg.device))
            log = []

            def callback(epoch, tr, va, te):
                log.append({"epoch": epoch, "train_acc": tr["acc"], "val_acc": va["acc"],
                            "star_minus_abce": gap(model, dev, star, sib)})

            try:
                trainer.fit(data, epoch_callback=callback)
            finally:
                logger.close()
            best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in
                    (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            model.load_state_dict(torch.load(cfg.exp_dir / "checkpoints" / "best.pt",
                                             map_location=next(model.parameters()).device)["state_dict"])
            entry = {"n": N, "r": args.r, "coverage": len(covered) / N, "test_acc": best["test_acc"],
                     "val_acc": best["val_acc"], "best_epoch": best["best_epoch"],
                     "seconds": round(time.time() - start, 1),
                     "cells": {g.length: g.num_faces for g in data.face_groups},
                     "max_log_w_over_epochs": max((r["train"].get("path_log_w_max", float("-inf"))
                                                   for r in rows), default=None),
                     "star_minus_abce_at_best": log[best["best_epoch"]]["star_minus_abce"],
                     "readout": i4.readout(model, data, info), "per_epoch": log}
            results[f"s2u_n{N}/{arm}/s{seed}"] = entry
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            ro = entry["readout"]
            print("K2b", arm, seed, json.dumps({k: entry[k] for k in (
                "r", "coverage", "test_acc", "best_epoch", "seconds", "max_log_w_over_epochs",
                "star_minus_abce_at_best")}),
                json.dumps({k: ro[k] for k in ("rank_mean_log_weight", "rank_lift", "rank_I_pi",
                                               "primary_pass", "secondary_pass",
                                               "num_length4_schemas", "planted_coverage")}),
                flush=True)
            for row in ro["schemas"][:6]:
                print("   ", row["schema"], round(row["mean_log_w"], 3), round(row["lift"], 3))
            if (entry["max_log_w_over_epochs"] or 0) > 30:
                raise SystemExit(f"STOP: max log-weight > 30 in {arm} seed {seed}")


if __name__ == "__main__":
    main()
