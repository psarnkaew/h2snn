"""I4 (Part I): S2 without the length confound (spec v1.4 E4; pre-registered in
followup_preregistration.md, I4). Run from H2S-NN/:
    python ../docs/WISP_evidence/i4_s2_unconfounded.py --arms full --seeds 42 43 44
Graph (per target T; types T, M1, M2, M3, N1, N2, N3, C):
    T -a- M1 -b- M2 -c- M3,   T -f- N1 -g- N2 -h- N3,   M2 -m- N3,
    M3 -d- C* (class y),  M3 -e- Ce (random class),  N3 -k- Ck (random class)
Length-4 schemas in the draw: sigma* = a b c d and a b c e, a b m h, a b m k, f g h k, f g h m (six;
only sigma* reaches the labelled carrier). No dead ends before length 4. Under p, sigma* is 1/8 of
the length-4 walks (1/24 of all walks). Read-outs on the evaluation draw (= the training draw,
resample_paths false as in H5) at the selected checkpoint. Primary: sigma* has the STRICTLY
highest mean learned log-weight per cell among all full-walk schemas (ties count against it);
secondary: strictly highest lift of I_pi. `uniform` (constant w) is the non-vacuity control: it
must not rank sigma* strictly first. Writes docs/WISP_evidence/i4_results.json."""
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
import src.data.dataset as dataset_module  # noqa: E402
from src.analysis.wisp_readouts import full_schema_keys, i_pi, i_sigma, mean_log_weight  # noqa: E402
from src.builders import build_experiment_components  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "WISP_evidence" / "i4_results.json"
NOISE = 4


def graph_s2u(n=400):
    g = torch.Generator().manual_seed(0)
    y = h5._balanced(n, g)
    ar = torch.arange(n)
    carrier_class = torch.cat([y, torch.randint(2, (2 * n,), generator=g)])  # C*, then Ce, Ck
    relations = {
        ("target", "a", "m1"): torch.stack([ar, ar]),
        ("m1", "b", "m2"): torch.stack([ar, ar]),
        ("m2", "c", "m3"): torch.stack([ar, ar]),
        ("m3", "d", "carrier"): torch.stack([ar, ar]),               # C*_i = carrier i
        ("m3", "e", "carrier"): torch.stack([ar, n + ar]),           # Ce_i = carrier n + i
        ("target", "f", "n1"): torch.stack([ar, ar]),
        ("n1", "g", "n2"): torch.stack([ar, ar]),
        ("n2", "h", "n3"): torch.stack([ar, ar]),
        ("n3", "k", "carrier"): torch.stack([ar, 2 * n + ar]),       # Ck_i = carrier 2n + i
        ("m2", "m", "n3"): torch.stack([ar, ar]),
    }
    xs = {t: torch.randn(n, NOISE, generator=g)
          for t in ("target", "m1", "m2", "m3", "n1", "n2", "n3")}
    xs["carrier"] = torch.cat([torch.nn.functional.one_hot(carrier_class, 2).float(),
                               torch.randn(3 * n, NOISE, generator=g)], 1)
    return h5._hetero("target", xs, relations, y), {
        "relevant_cpaper": None, "max_length": 4, "r": 200, "target": "target",
        "carrier": "carrier", "n": n}


def label(key, names, rel_names):
    parts = [names[key[0]]]
    for k in range(1, len(key), 3):
        parts.append(f"{rel_names[key[k]].split('__')[1]}{'+-~'[[1, -1, 0].index(key[k + 1] - 1)]}")
        parts.append(names[key[k + 2]])
    return "-".join(parts)


def strict_rank(table, star):
    """1 + the number of OTHER schemas scoring >= sigma* (ties count against sigma*)."""
    return 1 + sum(v >= table[star] for k, v in table.items() if k != star)


def readout(model, data, info):
    device = next(model.parameters()).device
    dev = data.to(device)
    model.eval()
    with torch.no_grad():
        h0 = model.encode_nodes(dev.x, dev.node_type)
        op = model.build_operator(h0, dev.edge_index, dev.edge_type, dev.face_groups, dev.num_nodes)
    groups = [g for g in data.face_groups if g.num_faces]
    weights = [w.cpu() for w in op.path_weights]
    si = data.fact_layer.self_inverse
    mlw = mean_log_weight(groups, weights, data.node_type, si)
    pi = i_pi(groups, weights, data.node_type, si)
    pi0 = i_pi(groups, [torch.ones_like(w) for w in weights], data.node_type, si)
    sigma = i_sigma(groups, weights, data.node_type, si)
    lift = {k: pi[k] / pi0[k] for k in pi}
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
    assert len(planted) == 1, planted
    star = planted.pop()
    names, rel_names = data.node_type_names, data.relation_names
    length_of = {k: (len(k) - 1) // 3 for k in mlw}
    return {
        "sigma_star": label(star, names, rel_names),
        "planted_coverage": len(covered) / n,
        "num_full_schemas": len(mlw),
        "num_length4_schemas": sum(1 for k in mlw if length_of[k] == 4),
        "rank_mean_log_weight": strict_rank(mlw, star), "rank_lift": strict_rank(lift, star),
        "rank_I_pi": strict_rank(pi, star), "rank_I_sigma": strict_rank(sigma, star)
        if star in sigma else None,
        "primary_pass": strict_rank(mlw, star) == 1, "secondary_pass": strict_rank(lift, star) == 1,
        "I_pi_star": pi[star], "I_pi_init_star": pi0[star], "lift_star": lift[star],
        "I_sigma_star": sigma.get(star, 0.0),
        "schemas": sorted(({"schema": label(k, names, rel_names), "length": length_of[k],
                            "mean_log_w": mlw[k], "I_pi": pi[k], "I_pi_init": pi0[k],
                            "lift": lift[k]} for k in mlw), key=lambda r: -r["mean_log_w"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n", type=int, default=400)
    args = parser.parse_args()
    graph, info = graph_s2u(args.n)
    dataset_module._load_hetero_data = lambda _config: graph.clone()
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for arm in args.arms:
        for seed in args.seeds:
            cfg = h5.config(f"s2u_n{args.n}", arm, seed, info, args.epochs)
            cfg.log_every = 1000
            if (cfg.exp_dir / "metrics.jsonl").exists():
                raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
            cfg.exp_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed, cfg.device)
            start = time.time()
            data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
            try:
                trainer.fit(data)
            finally:
                logger.close()
            seconds = round(time.time() - start, 1)
            best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
            rows = [json.loads(l) for l in
                    (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            model.load_state_dict(torch.load(cfg.exp_dir / "checkpoints" / "best.pt",
                                             map_location=next(model.parameters()).device)["state_dict"])
            entry = {"n": args.n, "test_acc": best["test_acc"], "val_acc": best["val_acc"],
                     "best_epoch": best["best_epoch"], "seconds": seconds,
                     "cells": {g.length: g.num_faces for g in data.face_groups},
                     "max_log_w_over_epochs": max((r["train"].get("path_log_w_max", float("-inf"))
                                                   for r in rows), default=None),
                     "readout": readout(model, data, info)}
            results[f"s2u_n{args.n}/{arm}/s{seed}"] = entry
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            ro = entry["readout"]
            print("I4", arm, seed, json.dumps({k: entry[k] for k in ("test_acc", "seconds", "cells",
                                                                    "max_log_w_over_epochs")}),
                  json.dumps({k: ro[k] for k in ro if k != "schemas"}), flush=True)
            for row in ro["schemas"][:8]:
                print("   ", row["schema"], round(row["mean_log_w"], 3), round(row["lift"], 3))
            if (entry["max_log_w_over_epochs"] or 0) > 30:
                raise SystemExit(f"STOP: max log-weight > 30 in {arm} seed {seed}")


if __name__ == "__main__":
    main()
