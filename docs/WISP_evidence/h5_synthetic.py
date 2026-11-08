"""H5 known-answer tasks (pre-registered: preregistration.md, H0-decision addendum D, E, F).
Run from H2S-NN/:
    python ../docs/WISP_evidence/h5_synthetic.py --task s1|s1p|s2 [--arms ...] [--seeds ...]
The production pipeline end to end (fact layer, WISP sampler, model, trainer); only the dataset
loader is replaced. Arms: full, endpoints, schema, uniform, no_path_cells, oracle_weight (w_f from the
known answer: 1 on relevant cells, 0 elsewhere; everything else trained), endpoints_ctx_off.
Writes docs/WISP_evidence/h5_results.json (one entry per run)."""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import src.data.dataset as dataset_module  # noqa: E402
from src.analysis.wisp_readouts import full_schema_keys, i_pi, i_sigma, mean_log_weight  # noqa: E402
from src.builders import build_experiment_components  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.models.h2snn_model import H2SNN  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "WISP_evidence" / "h5_results.json"
NOISE = 4


def _split(n, generator_seed=1):
    order = torch.randperm(n, generator=torch.Generator().manual_seed(generator_seed))
    return {"train": order[: n // 2], "val": order[n // 2: 3 * n // 4], "test": order[3 * n // 4:]}


def _balanced(n, g):
    y = torch.cat([torch.zeros(n // 2), torch.ones(n - n // 2)]).long()
    return y[torch.randperm(n, generator=g)]


def _hetero(target_type, xs, relations, y):
    from torch_geometric.data import HeteroData

    data = HeteroData()
    for t, x in xs.items():
        data[t].x = x
    for (src, rel, dst), index in relations.items():
        data[src, rel, dst].edge_index = index
        data[dst, f"{rel}_rev", src].edge_index = index.flip(0)
    data[target_type].y = y
    for name, index in _split(y.numel()).items():
        mask = torch.zeros(y.numel(), dtype=torch.bool)
        mask[index] = True
        data[target_type][f"{name}_mask"] = mask
    return data


def _bits(values, width):
    return ((values.unsqueeze(-1) >> torch.arange(width)) & 1).float() * 2 - 1


# ----------------------------------------------------------------------------- S1 (parity, transport)
def graph_s1(n=1000):
    g = torch.Generator().manual_seed(0)
    y = _balanced(n, g)
    b_p = torch.randint(2, (n,), generator=g)
    b_a = torch.randint(2, (2 * n,), generator=g)          # authors 2i, 2i+1 of target i
    target_of_author = torch.arange(2 * n) // 2
    rel_bit = b_p[target_of_author] ^ b_a                  # relevant cpaper bit
    cp_bit = torch.stack([rel_bit, 1 - rel_bit], 1).reshape(-1)          # cpapers 2a, 2a+1
    cp_class = torch.stack([y[target_of_author], 1 - y[target_of_author]], 1).reshape(-1)
    cp_author = torch.arange(2 * n).repeat_interleave(2)
    paper_x = torch.cat([b_p.float().unsqueeze(1) * 2 - 1, torch.randn(n, NOISE, generator=g)], 1)
    author_x = b_a.float().unsqueeze(1) * 2 - 1
    cpaper_x = torch.cat([torch.nn.functional.one_hot(cp_class, 2).float(),
                          cp_bit.float().unsqueeze(1) * 2 - 1,
                          torch.randn(4 * n, NOISE, generator=g)], 1)
    relations = {("author", "writes", "paper"): torch.stack([torch.arange(2 * n), target_of_author]),
                 ("author", "writes_c", "cpaper"): torch.stack([cp_author, torch.arange(4 * n)])}
    relevant_cpaper = torch.arange(0, 4 * n, 2)
    return _hetero("paper", {"paper": paper_x, "author": author_x, "cpaper": cpaper_x},
                   relations, y), {"relevant_cpaper": relevant_cpaper, "max_length": 2, "r": 64,
                                   "target": "paper", "carrier": "cpaper"}


# ----------------------------------------------------------------------------- S1' (selection, weight)
def graph_s1p(n=500, authors=8, per_author=8, bits=6):
    g = torch.Generator().manual_seed(0)
    y = _balanced(n, g)
    codes = 2 ** bits
    b_p = torch.randint(codes, (n,), generator=g)
    num_a = n * authors
    b_a = torch.randint(codes, (num_a,), generator=g)
    target_of_author = torch.arange(num_a) // authors
    num_c = num_a * per_author
    cp_author = torch.arange(num_a).repeat_interleave(per_author)
    need = b_p[target_of_author[cp_author]] ^ b_a[cp_author]              # xor making parity 0
    offset = torch.randint(1, codes, (num_c,), generator=g)              # != 0: irrelevant
    cp_bits = need ^ offset
    cp_class = torch.randint(2, (num_c,), generator=g)
    relevant = torch.zeros(num_c, dtype=torch.bool)
    pick_author = torch.randint(authors, (n,), generator=g)
    pick_paper = torch.randint(per_author, (n,), generator=g)
    chosen = (torch.arange(n) * authors + pick_author) * per_author + pick_paper
    relevant[chosen] = True
    cp_bits[chosen] = need[chosen]
    cp_class[chosen] = y
    decoy_bits = (codes - 1) ^ b_a                                         # complement of b_A
    author_x = torch.cat([_bits(b_a, bits), _bits(decoy_bits, bits)])     # authors, then decoys
    decoy_of_cp = num_a + cp_author
    paper_x = torch.cat([_bits(b_p, bits), torch.randn(n, NOISE, generator=g)], 1)
    cpaper_x = torch.cat([torch.nn.functional.one_hot(cp_class, 2).float(), _bits(cp_bits, bits),
                          torch.randn(num_c, NOISE, generator=g)], 1)
    writes_c = torch.stack([torch.cat([cp_author, decoy_of_cp]),
                            torch.cat([torch.arange(num_c), torch.arange(num_c)])])
    relations = {("author", "writes", "paper"): torch.stack([torch.arange(num_a), target_of_author]),
                 ("author", "writes_c", "cpaper"): writes_c}
    assert int(((b_p[target_of_author[cp_author]] ^ b_a[cp_author] ^ cp_bits) == 0).sum()) == n
    return _hetero("paper", {"paper": paper_x, "author": author_x, "cpaper": cpaper_x},
                   relations, y), {"relevant_cpaper": chosen, "max_length": 2, "r": 640,
                                   "target": "paper", "carrier": "cpaper"}


# ----------------------------------------------------------------------------- S2 (planted schema)
def graph_s2(n=400, distractors_t=4, distractors_m=2):  # 600 pre-registered; run-size rule (h5_sizing_note.md)
    g = torch.Generator().manual_seed(0)
    y = _balanced(n, g)
    ar = torch.arange(n)
    num_rand = n * (distractors_t + 2 * distractors_m)
    carrier_class = torch.cat([y, torch.randint(2, (num_rand,), generator=g)])  # C* first
    carriers = n + num_rand
    rand = (n + torch.arange(num_rand)).view(n, -1)
    e_dst = rand[:, :distractors_t].reshape(-1)
    g_dst = rand[:, distractors_t:distractors_t + distractors_m].reshape(-1)
    h_dst = rand[:, distractors_t + distractors_m:].reshape(-1)
    relations = {
        ("target", "a", "m1"): torch.stack([ar, ar]),
        ("m1", "b", "m2"): torch.stack([ar, ar]),
        ("m2", "c", "m3"): torch.stack([ar, ar]),
        ("m3", "d", "carrier"): torch.stack([ar, ar]),
        ("target", "e", "carrier"): torch.stack([ar.repeat_interleave(distractors_t), e_dst]),
        ("m1", "g", "carrier"): torch.stack([ar.repeat_interleave(distractors_m), g_dst]),
        ("m2", "h", "carrier"): torch.stack([ar.repeat_interleave(distractors_m), h_dst]),
    }
    xs = {"target": torch.randn(n, NOISE, generator=g),
          "m1": torch.randn(n, NOISE, generator=g), "m2": torch.randn(n, NOISE, generator=g),
          "m3": torch.randn(n, NOISE, generator=g),
          "carrier": torch.cat([torch.nn.functional.one_hot(carrier_class, 2).float(),
                                torch.randn(carriers, NOISE, generator=g)], 1)}
    return _hetero("target", xs, relations, y), {"relevant_cpaper": None, "max_length": 4, "r": 200,
                                                  "target": "target", "carrier": "carrier", "n": n}


TASKS = {"s1": graph_s1, "s1p": graph_s1p, "s2": graph_s2}
ARMS = {"full": ({}, {}), "endpoints": ({}, {"path_weight_input": "endpoints"}),
        "schema": ({}, {"path_weight_input": "schema"}), "uniform": ({}, {"path_weight_input": "uniform"}),
        "no_path_cells": ({}, {"no_path_cells": True}), "oracle_weight": ({}, {}),
        "endpoints_ctx_off": ({"geometry_context": False}, {"path_weight_input": "endpoints"})}


def config(task, arm, seed, info, epochs):
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / "acm_h2snn.yaml").read_text(encoding="utf-8"))
    model_extra, ablation = ARMS[arm]
    raw["model"] = {**raw["model"], **model_extra}
    raw["data"] = {**raw["data"], "path_source": "wisp", "wisp_max_length": info["max_length"],
                   "wisp_walks_per_anchor": info["r"], "resample_paths": False}
    raw.update(seed=seed, epochs=epochs, device="cuda", ablation=ablation, checkpoint_metric="acc",
               checkpoint_mode="max", exp_name=f"h5_{task}_{arm}_s{seed}", output_dir="experiments/wisp_h5")
    cfg = TrainConfig.from_dict(raw)
    cfg.data = dataclasses.replace(cfg.data, target_node_type=info["target"])
    return cfg


def relevance_of_cells(data, info):
    """Per non-empty group, a bool per cell: the cell ends at a relevant carrier (S1, S1')."""
    start = data.node_slices[info["carrier"]][0]
    relevant = torch.zeros(data.num_nodes, dtype=torch.bool)
    relevant[start + info["relevant_cpaper"]] = True
    out = []
    for group in data.face_groups:
        if group.num_faces:
            nodes = group.face_node_index.cpu()
            out.append(relevant[nodes[:, 0]] | relevant[nodes[:, -1]])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--arms", nargs="+", default=["full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    graph, info = TASKS[args.task]()
    dataset_module._load_hetero_data = lambda _config: graph.clone()
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    original = H2SNN._path_weights
    for arm in args.arms:
        for seed in args.seeds:
            cfg = config(args.task, arm, seed, info, args.epochs)
            if (cfg.exp_dir / "metrics.jsonl").exists():
                raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
            cfg.exp_dir.mkdir(parents=True, exist_ok=True)
            seed_everything(seed, cfg.device)
            start = time.time()
            data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
            relevant_cells = None
            if arm == "oracle_weight":
                relevant_cells = [r.float() for r in relevance_of_cells(data, info)]
                sizes = {int(r.numel()): r for r in relevant_cells}
                assert len(sizes) == len(relevant_cells), "oracle needs distinct group sizes"

                def oracle(self, latent, group, like):
                    return sizes[group.num_faces].to(device=like.device, dtype=like.dtype)
                H2SNN._path_weights = oracle
            try:
                trainer.fit(data)
            finally:
                logger.close()
                H2SNN._path_weights = original
            best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
            rows = [json.loads(l) for l in (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            entry = {"test_acc": best["test_acc"], "val_acc": best["val_acc"], "best_epoch": best["best_epoch"],
                     "epochs_run": len(rows), "train_acc_at_best_trainmode": rows[best["best_epoch"]]["train"]["acc"],
                     "seconds": round(time.time() - start, 1),
                     "cells": {g.length: g.num_faces for g in data.face_groups},
                     "log_w_max_last": rows[-1]["train"].get("path_log_w_max")}
            if info["relevant_cpaper"] is not None:
                start_c = data.node_slices[info["carrier"]][0]
                ends = set()
                for group in data.face_groups:
                    ends |= set(group.face_node_index[:, -1].tolist()) | set(group.face_node_index[:, 0].tolist())
                entry["coverage_relevant_cells"] = float(sum(int(start_c + c) in ends for c in info["relevant_cpaper"].tolist())
                                                         / info["relevant_cpaper"].numel())
            if args.task == "s2" and arm == "full":
                device = next(model.parameters()).device
                model.load_state_dict(torch.load(cfg.exp_dir / "checkpoints" / "best.pt", map_location=device)["state_dict"])
                model.eval()
                dev = data.to(device)
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
                names = data.node_type_names
                rel_names = data.relation_names

                def label(key):
                    parts = [names[key[0]]]
                    for k in range(1, len(key), 3):
                        parts.append(f"{rel_names[key[k]]}{'+-~'[[1, -1, 0].index(key[k + 1] - 1)]}")
                        parts.append(names[key[k + 2]])
                    return "-".join(parts)

                lift = {k: pi[k] / pi0[k] for k in pi}
                # sigma*: the canonical key of the cells that are exactly a planted chain
                # T_i - M1_i - M2_i - M3_i - C*_i (either orientation); every such cell must agree.
                chain = torch.stack([data.node_slices[t][0] + torch.arange(info["n"])
                                     for t in ("target", "m1", "m2", "m3", "carrier")], 1)
                planted_keys, covered = set(), set()
                for group in groups:
                    if group.face_node_index.size(1) != 5:
                        continue
                    keys = full_schema_keys(group, data.node_type, si)
                    rows = group.face_node_index.cpu()
                    for orient in (rows, rows.flip(1)):
                        target_local = orient[:, 0] - data.node_slices["target"][0]
                        inside = (target_local >= 0) & (target_local < info["n"])
                        match = inside.clone()
                        match[inside] = (orient[inside] == chain[target_local[inside]]).all(1)
                        for f in match.nonzero().flatten().tolist():
                            planted_keys.add(keys[f])
                            covered.add(int(target_local[f]))
                assert len(planted_keys) == 1, planted_keys
                star = planted_keys.pop()

                def rank(table):
                    return 1 + sum(v > table[star] for v in table.values())

                entry["readout"] = {
                    "sigma_star": label(star),
                    "planted_coverage": len(covered) / info["n"],
                    "rank_mean_log_weight": rank(mlw), "mean_log_weight_star": mlw[star],
                    "rank_lift_I_pi": rank(lift), "lift_star": lift[star],
                    "rank_I_pi": rank(pi), "I_pi_star": pi[star], "I_pi_init_star": pi0[star],
                    "rank_I_sigma": rank(sigma), "I_sigma_star": sigma.get(star, 0.0),
                    "primary_pass": rank(mlw) == 1, "secondary_pass": rank(lift) == 1,
                    "top_mean_log_weight": sorted(((label(k), v) for k, v in mlw.items()), key=lambda t: -t[1])[:5],
                    "top_lift_I_pi": sorted(((label(k), v) for k, v in lift.items()), key=lambda t: -t[1])[:5],
                    "top_I_pi": sorted(((label(k), v) for k, v in pi.items()), key=lambda t: -t[1])[:5],
                    "top_I_sigma": sorted(((label(k), v) for k, v in sigma.items()), key=lambda t: -t[1])[:5],
                    "num_full_schemas": len(mlw)}
            results[f"{args.task}/{arm}/s{seed}"] = entry
            print(args.task, arm, seed, json.dumps(entry)[:600], flush=True)
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
