"""Card 4 K4: the S3 audit, run before any S3 training run (spec §0 rule 6; card K4 (i)-(iii)).

(i)  Tag balance on the generated graph: for every node type and every (relation, role), the node's own
     tag and its context tag mean (mean over that relation's neighbours, `neighbour_means` on a
     [tag, tagged] feature: the definition of the former `RelationalContext`, kept here because graph
     structure still reaches each node through the solve) have the same empirical distribution on the relevant cell's nodes as on the
     other nodes of that type. Statistic per type: total-variation distance between the two empirical
     distributions of the full vector (own + every relation slot). Passes iff TV <= 1e-12 for every type
     (the balanced design makes them equal exactly).
(ii) Fact-input probe: on the training targets, for each hop position (relation r1..r4) a logistic probe
     (Part K K0 solver, lambda 1e-3) predicts "the fact lies on the relevant cell" from the fact's input
     [x_u, x_v, one-hot relation] (x = the prepared node features; spec 1.25 Z1: the map MLP reads the
     endpoints' h0 and the relation only, no relational context `c`; card 4 used [x_u, c_u, x_v, c_v,
     rel]). Positives: the relevant cell's fact; negatives: the same
     hop of one irrelevant length-4 cell of the same target (fixed seed 0). The training targets are
     split in two halves (seed 0): fit on one, accuracy on the other. Passes iff every hop's held-out
     accuracy lies in the binomial 99 % interval of chance (p = 0.5, n = held-out samples).
(iii) Reachability (stated): one scalar moves at most sum_t lr_t = 0.01 * 100 = 1.0, so b and tau cannot
     supply a log-weight gap of ~ log 512 = 6.24; it must come from g.
r: the smallest r in (640, 1280, 2560, 5120, 10240) whose evaluation draws (seeds 1, 2, 3) contain all
64 length-4 cells of every target (set-up from the draws only, as Part K's K2b).

    python scripts/known_answer/s3_audit.py --out ../docs/card4_evidence/k4/s3_audit.json
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import s3_graph  # noqa: E402
from src.data.dataset import build_dataset, build_node_classification_paths  # noqa: E402

R_LADDER = (640, 1280, 2560, 5120, 10240)
SEEDS = (1, 2, 3)
TV_TOLERANCE = 1e-12
PROBE_SEED = 0
CELLS_PER_TARGET = s3_graph.FAN ** 3


def neighbour_means(h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                    num_relations: int, shared_role_relations: tuple[int, ...]) -> torch.Tensor:
    """`[N, 2R d]`: slot `2 r + s` = mean of `h` over the node's (relation r, role s) neighbours (zeros if
    none; one role for rho(r) = r). The definition of the former `RelationalContext` (spec 1.24, removed
    from the model by spec 1.25 Z1), used only for audit (i)."""
    num_nodes, d = h.shape
    slots_per_node = 2 * num_relations
    src, tgt = edge_index[0], edge_index[1]
    shared = torch.zeros(num_relations, dtype=torch.bool)
    if shared_role_relations:
        shared[list(shared_role_relations)] = True
    target_role = (~shared[edge_type]).long()
    slots = torch.cat([src * slots_per_node + 2 * edge_type,
                       tgt * slots_per_node + 2 * edge_type + target_role])
    values = torch.cat([h[tgt], h[src]])
    sums = h.new_zeros((num_nodes * slots_per_node, d)).index_add(0, slots, values)
    counts = h.new_zeros(num_nodes * slots_per_node).index_add(0, slots, h.new_ones(slots.numel()))
    return (sums / counts.clamp_min(1.0).unsqueeze(-1)).view(num_nodes, slots_per_node * d)


def global_ids(data, node_type: str, local: torch.Tensor) -> torch.Tensor:
    return data.node_slices[node_type][0] + local


def tag_balance(data, info) -> dict:
    """(i): per type, TV distance between relevant and other nodes' (own tag, context tag means)."""
    num = data.num_nodes
    tag = torch.zeros(num, 2, dtype=torch.float64)
    for node_type, key in (("p", "t_p"), ("b", "t_b"), ("q", "t_q")):
        ids = global_ids(data, node_type, torch.arange(info[key].numel()))
        tag[ids, 0] = info[key].double() * 2 - 1
        tag[ids, 1] = 1.0
    context = neighbour_means(tag, data.edge_index, data.edge_type, len(data.relation_names),
                              tuple(data.fact_layer.self_inverse))
    vector = torch.cat([tag, context], dim=1)
    relevant = {"a": info["a_rel"], "b": info["b_rel"], "c": info["c_rel"], "q": info["c_rel"]}
    out = {}
    for node_type, rel_local in relevant.items():
        start, end = data.node_slices[node_type]
        is_rel = torch.zeros(end - start, dtype=torch.bool)
        is_rel[rel_local] = True
        rows = vector[start:end]
        keys = [tuple(round(v, 9) for v in row.tolist()) for row in rows]
        pos = Counter(k for k, r in zip(keys, is_rel.tolist()) if r)
        neg = Counter(k for k, r in zip(keys, is_rel.tolist()) if not r)
        n_pos, n_neg = sum(pos.values()), sum(neg.values())
        tv = 0.5 * sum(abs(pos[k] / n_pos - neg[k] / n_neg) for k in set(pos) | set(neg))
        out[node_type] = {"relevant_nodes": n_pos, "other_nodes": n_neg, "tv_distance": tv,
                          "distinct_vectors": len(set(pos) | set(neg)),
                          "relevant_distribution": {str(k): v / n_pos for k, v in sorted(pos.items())}}
    out["pass"] = all(v["tv_distance"] <= TV_TOLERANCE for k, v in out.items() if k != "pass")
    out["slot_layout"] = ("vector = [own tag, own tagged] + per relation r (fact-layer order "
                          f"{list(data.relation_names)}) and role s: [mean tag, mean tagged]")
    return out


def fact_probe(data, info) -> dict:
    """(ii): one probe per hop position on the training targets."""
    x = data.x.double()
    n = info["n"]
    train_targets = torch.nonzero(data.target_train_mask).flatten()          # local target ids
    g = torch.Generator().manual_seed(PROBE_SEED)
    fan = s3_graph.FAN
    # an irrelevant length-4 chain per target (A, B, C uniform; redrawn if it is the relevant one)
    a_irr = torch.arange(n) * fan + torch.randint(fan, (n,), generator=g)
    b_irr = a_irr * fan + torch.randint(fan, (n,), generator=g)
    c_irr = b_irr * fan + torch.randint(fan, (n,), generator=g)
    same = c_irr == info["c_rel"]
    while bool(same.any()):
        c_irr[same] = (b_irr[same] * fan + torch.randint(fan, (int(same.sum()),), generator=g))
        same = c_irr == info["c_rel"]
    order = train_targets[torch.randperm(train_targets.numel(), generator=g)]
    half = order.numel() // 2
    fit_targets, test_targets = order[:half], order[half:]
    chains = {}
    for label, (a, b, cc) in ((1, (info["a_rel"], info["b_rel"], info["c_rel"])),
                              (0, (a_irr, b_irr, c_irr))):
        chains[label] = [global_ids(data, "p", torch.arange(n)), global_ids(data, "a", a),
                         global_ids(data, "b", b), global_ids(data, "c", cc),
                         global_ids(data, "q", cc)]
    relation_names = ("r1", "r2", "r3", "r4")
    results = {}
    for hop, rel in enumerate(relation_names):
        onehot = torch.zeros(len(relation_names), dtype=torch.float64)
        onehot[hop] = 1.0
        rows, labels, split = [], [], []
        for targets, is_test in ((fit_targets, False), (test_targets, True)):
            for label in (1, 0):
                u = chains[label][hop][targets]
                v = chains[label][hop + 1][targets]
                rows.append(torch.cat([x[u], x[v], onehot.expand(targets.numel(), -1)], dim=1))
                labels.append(torch.full((targets.numel(),), label))
                split.append(torch.full((targets.numel(),), is_test))
        features = torch.cat(rows)
        y = torch.cat(labels)
        test = torch.cat(split)
        accuracy = common.logistic_probe(features, y, ~test, test)
        n_test = int(test.sum())
        lo, hi = binom.interval(0.99, n_test, 0.5)
        results[rel] = {"held_out_accuracy": accuracy, "n_fit": int((~test).sum()), "n_test": n_test,
                        "chance_interval_99": [lo / n_test, hi / n_test],
                        "within": bool(lo / n_test <= accuracy <= hi / n_test)}
    results["pass"] = all(v["within"] for k, v in results.items() if k != "pass")
    results["features"] = "[x_u, x_v, onehot(r1..r4)] along the chain P->Q (spec 1.25 Z1: no c)"
    return results


def choose_r(graph, info) -> dict:
    out = {}
    base_cfg = common.frozen_config("k4_s3_audit", 1, info["max_length"], R_LADDER[0], info["target"])
    common.install_graph(graph)
    data = build_dataset(base_cfg)
    start_p = data.node_slices["p"][0]
    for r in R_LADDER:
        per_seed = {}
        for seed in SEEDS:
            cfg = common.frozen_config("k4_s3_audit", seed, info["max_length"], r, info["target"])
            groups = build_node_classification_paths(
                config=cfg, fact_layer=data.fact_layer, node_slices=data.node_slices,
                node_type=data.node_type, node_type_names=data.node_type_names,
                edge_index=data.edge_index, edge_type=data.edge_type,
                relation_tuples=data.relation_tuples, seed=seed)
            counts = torch.zeros(info["n"], dtype=torch.long)
            cells = {}
            for group in groups:
                cells[group.length] = group.num_faces
                if group.length != 4:
                    continue
                nodes = group.face_node_index
                p_end = torch.where((nodes[:, 0] >= start_p) & (nodes[:, 0] < start_p + info["n"]),
                                    nodes[:, 0], nodes[:, -1])
                counts.index_add_(0, p_end - start_p, torch.ones_like(p_end))
            per_seed[seed] = {"targets_with_all_64": float((counts == CELLS_PER_TARGET).double().mean()),
                              "length4_cells_share": float(counts.sum()) / (info["n"] * CELLS_PER_TARGET),
                              "max_per_target": int(counts.max()), "cells": cells}
        out[r] = per_seed
        if all(v["targets_with_all_64"] == 1.0 for v in per_seed.values()):
            return {"chosen_r": r, "ladder": out}
    return {"chosen_r": None, "ladder": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graph, info = s3_graph.graph_s3(n=2000)
    cfg = common.frozen_config("k4_s3_audit", 1, info["max_length"], R_LADDER[0], info["target"])
    common.install_graph(graph)
    data = build_dataset(cfg)
    record = {"graph": "s3_graph.graph_s3(n = 2000, seed 0)", "num_nodes": int(data.num_nodes),
              "num_facts": int(data.edge_index.size(1)), "relations": list(data.relation_names),
              "tag_balance": tag_balance(data, info), "fact_probe": fact_probe(data, info),
              "reachability": {"scalar_reach": 0.01 * common.EPOCHS, "needed_gap": math.log(512),
                               "statement": "b and theta_tau move at most 1.0 each in 100 epochs "
                                            "(lr 1e-2 constant): tau in [0.37, 2.72], b shifts every "
                                            "cell alike; a relevant/irrelevant gap ~ log 512 = 6.24 "
                                            "must come from g (one projection + GELU + linear per "
                                            "length; a 3-input XOR of +-1 tags is representable by "
                                            "one GELU hidden layer, DERIVED)"},
              **common.run_identity()}
    record["r_choice"] = choose_r(graph, info)
    record["audit_pass"] = bool(record["tag_balance"]["pass"] and record["fact_probe"]["pass"]
                                and record["r_choice"]["chosen_r"] is not None)
    common.write_json(args.out, record)
    print({"tag_balance_pass": record["tag_balance"]["pass"],
           "tv": {k: v["tv_distance"] for k, v in record["tag_balance"].items()
                  if isinstance(v, dict)},
           "probe": {k: (v["held_out_accuracy"], v["chance_interval_99"]) for k, v in
                     record["fact_probe"].items() if isinstance(v, dict)},
           "chosen_r": record["r_choice"]["chosen_r"], "audit_pass": record["audit_pass"]},
          flush=True)


if __name__ == "__main__":
    main()
