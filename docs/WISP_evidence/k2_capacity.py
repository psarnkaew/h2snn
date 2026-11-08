"""K2 (Part K, gate ii): capacity dose-response on S1' with a FIXED-beta oracle and an operator
probe (pre-registered: k_preregistration.md, K2). Run from H2S-NN/:
    python ../docs/WISP_evidence/k2_capacity.py --n 500 --betas 0 log8 log64 log512 --seeds 42 43 44
    python ../docs/WISP_evidence/k2_capacity.py --n 2000 --arm full --seeds 42        (context arm)
    python ../docs/WISP_evidence/k2_capacity.py --n 2000 --fit-check                  (memory check)
Graph: h5_synthetic.graph_s1p(n), unchanged except n (split 50/25/25). Config: I3's (h5.config:
acm_h2snn.yaml model block, path_source wisp, l_max 2, r 640, resample_paths false, 100 epochs,
validation selection), log_every 1000.
Oracle: w_f = exp(b + beta * o_f); beta is a constant buffer (requires_grad only to read dL/dbeta;
never given to the optimizer); b is a learnable scalar in the scalar group (b0 = log 0.5); g is not
used for weights; everything else trains; nothing is clamped.
Probe (at initialisation and at the selected checkpoint; does not train the model): u = A h0 - h0
on the evaluation draw; logistic regression of y on the target papers' u (standardised with the
training targets' mean/std), bias unpenalised, L2 lambda = 1e-3 on the mean BCE, LBFGS(lr 1,
max_iter 500, history 50, tol_grad 1e-9, tol_change 1e-12, strong Wolfe), float64, w0 = 0; test
accuracy. Control: the same on h0. Dispersion: 1 - || mean_f (2y_f - 1) t_f / ||t_f|| || with
t_f = T_P^T T_Q* h0(Q*) through the production PathTransportOperator.
Writes docs/WISP_evidence/k2_results.json. Accuracies are synthetic functional checks."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "docs" / "WISP_evidence"))
import h5_synthetic as h5  # noqa: E402
import i3_s1p_oracle as i3  # noqa: E402  (mechanism log)
import src.data.dataset as dataset_module  # noqa: E402
import src.data.wisp_sampler  # noqa: E402,F401  (imported up front)
import src.trainers.trainer  # noqa: E402,F401
from src.builders import (_checkpointer, build_logger, build_optimizer,  # noqa: E402
                          build_scheduler, build_trainer)
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import apply_operator, build_model  # noqa: E402
from src.models.map_operators import PathTransportOperator  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "WISP_evidence" / "k2_results.json"
BETAS = {"0": 0.0, "log8": math.log(8), "log64": math.log(64), "log512": math.log(512)}
LAMBDA = 1e-3


def install_oracle(model, relevant, beta, device, init):
    model.oracle_b = nn.Parameter(torch.tensor(math.log(init)))
    model.register_buffer("oracle_beta", torch.tensor(beta, device=device, requires_grad=True))
    base_scalars = model.scalar_parameters
    model.scalar_parameters = lambda: list(base_scalars()) + [model.oracle_b]
    by_size = {int(r.numel()): r.float() for r in relevant}
    assert len(by_size) == len(relevant)

    def weights(latent, group, like):
        o = by_size[group.num_faces].to(device=like.device, dtype=like.dtype)
        return torch.exp(model.oracle_b + model.oracle_beta * o)
    model._path_weights = weights


def logistic_probe(features, y, train, test):
    x = features.double()
    mean, std = x[train].mean(0), x[train].std(0)
    std = torch.where(std > 0, std, torch.ones_like(std))
    x = (x - mean) / std
    yd = y.double()
    w = torch.zeros(x.size(1), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, history_size=50, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = x[train] @ w + b
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yd[train]) \
            + 0.5 * LAMBDA * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = (x[test] @ w + b) > 0
    return float((pred.long() == y[test]).double().mean())


def probe_and_dispersion(model, dev, relevant_cells, train, test):
    """Probe accuracies (u and the h0 control), dispersion, relevant share of D (median)."""
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            h0 = model.encode_nodes(dev.x, dev.node_type)
            op = model.build_operator(h0, dev.edge_index, dev.edge_type, dev.face_groups,
                                      dev.num_nodes)
            u = apply_operator(op, h0) - h0
            targets = dev.target_indices
            y = dev.target_y
            out = {"probe_u": logistic_probe(u[targets].cpu(), y.cpu(), train, test),
                   "probe_h0": logistic_probe(h0[targets].cpu(), y.cpu(), train, test)}
            group = [g for g in dev.face_groups if g.num_faces][0]
            rel = relevant_cells[0].to(h0.device)
            nodes = group.face_node_index[rel]
            transport = PathTransportOperator(
                op.maps.src, op.maps.tgt, nodes, group.face_edge_index[rel], dev.edge_index,
                shared_scale=model.map_scale == "per_edge",
                face_edge_sign=group.face_edge_sign[rel].to(h0.dtype))
            cell_states = transport.apply(h0)                      # [R, 3, d], T_j x_j
            start, end = dev.node_slices["paper"]
            p_at_0 = (nodes[:, 0] >= start) & (nodes[:, 0] < end)
            p_pos = torch.where(p_at_0, 0, nodes.size(1) - 1)
            q_pos = nodes.size(1) - 1 - p_pos
            rows = torch.arange(nodes.size(0), device=h0.device)
            residual = torch.zeros_like(cell_states)
            residual[rows, p_pos] = cell_states[rows, q_pos]
            carried = transport.apply_T(residual, dev.num_nodes)    # T_P^T T_Q* h0(Q*) at P
            p_nodes = nodes[rows, p_pos]
            t = carried[p_nodes]
            label = dev.y[p_nodes].to(h0.dtype)
            signed = (2 * label - 1).unsqueeze(1) * t / t.norm(dim=1, keepdim=True).clamp_min(
                torch.finfo(t.dtype).tiny)
            out["dispersion"] = float(1 - signed.mean(0).norm())
            out["relevant_cells"] = int(rel.sum())
    finally:
        model.train(was)
    out.update(i3.mechanism(model, dev, relevant_cells))
    return out


def run(n, arm, beta_label, seed, graph, info, epochs):
    name = f"k2_{arm}" + (f"_b{beta_label}" if arm == "oracle" else "")
    h5.ARMS[name] = ({}, {})
    cfg = h5.config(f"s1p_k2_n{n}", name, seed, info, epochs)
    cfg.log_every = 1000
    if (cfg.exp_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    seed_everything(seed, cfg.device)
    start = time.time()
    data = build_dataset(cfg)
    model = build_model(cfg, data)
    relevant = h5.relevance_of_cells(data, info)
    beta = None
    if arm == "oracle":
        install_oracle(model, relevant, BETAS[beta_label], device, cfg.model.path_weight_init)
        beta = model.oracle_beta
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    logger = build_logger(cfg)
    trainer = build_trainer(model=model, optimizer=optimizer, scheduler=scheduler, cfg=cfg,
                            logger=logger, checkpointer=_checkpointer(cfg))
    model.to(device)
    dev = data.to(device)
    train = data.target_train_mask.cpu()
    test = data.target_test_mask.cpu()
    init_probe = probe_and_dispersion(model, dev, relevant, train, test)
    log = []

    def callback(epoch, tr, va, te):
        row = {"epoch": epoch, "train_acc": tr["acc"], "val_acc": va["acc"], "train_loss": tr["loss"]}
        if beta is not None:
            row["dL_dbeta"] = None if beta.grad is None else float(beta.grad)
            beta.grad = None
            row["b"] = float(model.oracle_b)
        row.update(i3.mechanism(model, dev, relevant))
        log.append(row)

    try:
        trainer.fit(data, epoch_callback=callback)
    finally:
        logger.close()
    best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
    model.load_state_dict(torch.load(cfg.exp_dir / "checkpoints" / "best.pt",
                                     map_location=device)["state_dict"])
    ckpt_probe = probe_and_dispersion(model, dev, relevant, train, test)
    best_epoch = best["best_epoch"]
    entry = {"n": n, "arm": arm, "beta": beta_label, "seed": seed, "test_acc": best["test_acc"],
             "val_acc": best["val_acc"], "best_epoch": best_epoch,
             "seconds": round(time.time() - start, 1),
             "cells": {g.length: g.num_faces for g in data.face_groups},
             "coverage_relevant_cells": float(sum(int(r.sum()) for r in relevant)
                                              / info["relevant_cpaper"].numel()),
             "max_log_w_over_epochs": max((r.get("log_w_max", float("-inf")) for r in log),
                                          default=None),
             "probe_init": init_probe, "probe_ckpt": ckpt_probe, "per_epoch": log}
    if beta is not None:
        before = [r["dL_dbeta"] for r in log[:best_epoch] if r["dL_dbeta"] is not None]
        entry["neg_grad_positive_fraction_before_best"] = (
            sum(-g > 0 for g in before) / len(before) if before else None)
        entry["epochs_before_best"] = len(before)
    return entry


def fit_check(n, info, graph):
    """One epoch at size n on the GPU: does it fit in 8 GB? (set-up, not a result)"""
    h5.ARMS["k2_fit"] = ({}, {})
    cfg = h5.config(f"s1p_k2fit_n{n}", "k2_fit", 42, info, 1)
    cfg.log_every = 1000
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    data = build_dataset(cfg)
    model = build_model(cfg, data)
    relevant = h5.relevance_of_cells(data, info)
    install_oracle(model, relevant, BETAS["log512"], torch.device("cuda"), 0.5)
    optimizer = build_optimizer(model, cfg)
    logger = build_logger(cfg)
    trainer = build_trainer(model=model, optimizer=optimizer, scheduler=None, cfg=cfg,
                            logger=logger, checkpointer=_checkpointer(cfg))
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        trainer.fit(data)
    finally:
        logger.close()
    print(json.dumps({"n": n, "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
                      "seconds_one_epoch": round(time.time() - start, 1),
                      "cells": {g.length: g.num_faces for g in data.face_groups},
                      "nodes": int(data.num_nodes)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--arm", choices=["oracle", "full"], default="oracle")
    parser.add_argument("--betas", nargs="+", default=["0"], choices=list(BETAS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fit-check", action="store_true")
    args = parser.parse_args()
    graph, info = h5.graph_s1p(n=args.n)
    dataset_module._load_hetero_data = lambda _config: graph.clone()
    if args.fit_check:
        fit_check(args.n, info, graph)
        return
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    betas = args.betas if args.arm == "oracle" else ["-"]
    for beta_label in betas:
        for seed in args.seeds:
            entry = run(args.n, args.arm, beta_label, seed, graph, info, args.epochs)
            key = f"n{args.n}/{args.arm}" + (f"/beta_{beta_label}" if args.arm == "oracle" else "") \
                + f"/s{seed}"
            results[key] = entry
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            short = {k: entry[k] for k in ("test_acc", "best_epoch", "seconds",
                                           "coverage_relevant_cells", "max_log_w_over_epochs")}
            short["neg_grad_pos_frac"] = entry.get("neg_grad_positive_fraction_before_best")
            for when in ("probe_init", "probe_ckpt"):
                p = entry[when]
                short[when] = {k: round(p[k], 4) if isinstance(p.get(k), float) else p.get(k)
                               for k in ("probe_u", "probe_h0", "dispersion",
                                         "relevant_share_median", "path_share_median")}
            if args.arm == "full":
                last = entry["per_epoch"][-1]
                short["gap_last"] = last.get("log_w_relevant", 0) - last.get("log_w_irrelevant", 0)
            print("K2", key, json.dumps(short), flush=True)
            if (entry["max_log_w_over_epochs"] or 0) > 30:
                raise SystemExit(f"STOP: max log-weight > 30 in {key}")


if __name__ == "__main__":
    main()
