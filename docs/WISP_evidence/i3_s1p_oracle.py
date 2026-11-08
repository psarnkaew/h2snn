"""I3 (Part I): S1' with a learnable-scale oracle and the learning-mechanism log
(pre-registered: followup_preregistration.md, I3). Run from H2S-NN/:
    python ../docs/WISP_evidence/i3_s1p_oracle.py --arms full oracle --seeds 42 43 44
Graph, split and config are H5's S1' (h5_synthetic.graph_s1p, h5_synthetic.config), at the spec
v1.4 defaults (g reads h0). Arms:
  full            the model
  oracle          w_f = exp(b + beta * o_f), o_f in {0, 1} the known relevance of cell f; b and beta
                  learnable scalars in the scalar optimizer group (b0 = log path_weight_init,
                  beta0 = 0); g is not used for weights; everything else trained; nothing clamped
  endpoints, uniform, no_path_cells   ladder controls
  full_no_h0      full with the readout on [x_K, 0] (the readout without the h0 half; G0b H1)
Per epoch (evaluation draw = training draw, resample_paths false as in H5; model in eval mode,
no gradient, after the epoch's update): mean log-weight of relevant vs irrelevant cells; median
over targets of the relevant cells' share of the target's scalar degree D and of the path part's
share of D; train/val accuracy; b and beta (oracle). Accuracies are synthetic functional checks.
Writes docs/WISP_evidence/i3_results.json."""
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
import src.data.dataset as dataset_module  # noqa: E402
from src.builders import (_checkpointer, build_logger, build_optimizer,  # noqa: E402
                          build_scheduler, build_trainer)
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "WISP_evidence" / "i3_results.json"
ARMS = {"full": ({}, {}), "oracle": ({}, {}), "endpoints": ({}, {"path_weight_input": "endpoints"}),
        "uniform": ({}, {"path_weight_input": "uniform"}), "no_path_cells": ({}, {"no_path_cells": True}),
        "full_no_h0": ({}, {})}


def make_config(arm, seed, info, epochs):
    h5.ARMS[f"i3_{arm}"] = ARMS[arm]
    cfg = h5.config("s1p", f"i3_{arm}", seed, info, epochs)
    cfg.log_every = 1000  # operator diagnostics only at epoch 0 (pays for the per-epoch log)
    return cfg


def install_oracle(model, relevant, init):
    """w_f = exp(b + beta * o_f); b, beta join the scalar optimizer group."""
    model.oracle_b = nn.Parameter(torch.tensor(math.log(init)))
    model.oracle_beta = nn.Parameter(torch.tensor(0.0))
    base_scalars = model.scalar_parameters
    model.scalar_parameters = lambda: list(base_scalars()) + [model.oracle_b, model.oracle_beta]
    by_size = {int(r.numel()): r.float() for r in relevant}
    assert len(by_size) == len(relevant), "oracle needs distinct group sizes"

    def weights(latent, group, like):
        o = by_size[group.num_faces].to(device=like.device, dtype=like.dtype)
        return torch.exp(model.oracle_b + model.oracle_beta * o)
    model._path_weights = weights


def install_no_h0_readout(model):
    encode = model.encode

    def without_h0(*args, **kwargs):
        state = encode(*args, **kwargs)
        return state._replace(representation=torch.cat(
            [state.x_star, torch.zeros_like(state.h0)], dim=-1))
    model.encode = without_h0


def mechanism(model, data, relevant):
    """The per-epoch log on the evaluation draw (no gradient; mode restored)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            h0 = model.encode_nodes(data.x, data.node_type)
            op = model.build_operator(h0, data.edge_index, data.edge_type, data.face_groups,
                                      data.num_nodes)
    finally:
        model.train(was_training)
    targets = data.target_indices
    lap = op.laplacian
    edge = lap.edge_degree()[targets]
    row = {}
    if not lap.path_terms:
        row.update(path_share_median=0.0, relevant_share_median=0.0)
        return row
    path = sum(t.diagonal(data.num_nodes) for t in lap.path_terms)[targets]
    rel_deg = torch.zeros(data.num_nodes, dtype=edge.dtype, device=edge.device)
    for term, rel in zip(lap.path_terms, relevant):
        mask = rel.to(term.weights.device, term.weights.dtype)
        per = (1.0 - 1.0 / term.cell_size) * (term.weights * mask).unsqueeze(1) \
            * term.incidence.squared_scales()
        rel_deg.index_add_(0, term.incidence.face_node_index.reshape(-1), per.reshape(-1))
    total = edge + path
    weights = torch.cat(op.path_weights)
    log_w = weights.clamp_min(torch.finfo(weights.dtype).tiny).log()  # read-out only
    rel_all = torch.cat([r.to(log_w.device) for r in relevant])
    row.update(
        path_share_median=float((path / total).median()),
        relevant_share_median=float((rel_deg[targets] / total).median()),
        log_w_relevant=float(log_w[rel_all].mean()),
        log_w_irrelevant=float(log_w[~rel_all].mean()),
        log_w_max=float(log_w.max()),
    )
    return row


def run(arm, seed, graph, info, epochs):
    cfg = make_config(arm, seed, info, epochs)
    if (cfg.exp_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, cfg.device)
    start = time.time()
    data = build_dataset(cfg)
    model = build_model(cfg, data)
    relevant = h5.relevance_of_cells(data, info)  # per non-empty group, a bool per cell
    if arm == "oracle":
        install_oracle(model, relevant, cfg.model.path_weight_init)
    if arm == "full_no_h0":
        install_no_h0_readout(model)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    logger = build_logger(cfg)
    trainer = build_trainer(model=model, optimizer=optimizer, scheduler=scheduler, cfg=cfg,
                            logger=logger, checkpointer=_checkpointer(cfg))
    device_data = data.to(torch.device(cfg.device))
    log = []

    def callback(epoch, train, val, test):
        row = {"epoch": epoch, "train_acc": train["acc"], "val_acc": val["acc"],
               "train_loss": train["loss"]}
        row.update(mechanism(model, device_data, relevant))
        if arm == "oracle":
            row.update(b=float(model.oracle_b), beta=float(model.oracle_beta))
        log.append(row)

    try:
        trainer.fit(data, epoch_callback=callback)
    finally:
        logger.close()
    best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
    entry = {"test_acc": best["test_acc"], "val_acc": best["val_acc"],
             "best_epoch": best["best_epoch"], "seconds": round(time.time() - start, 1),
             "cells": {g.length: g.num_faces for g in data.face_groups},
             "coverage_relevant_cells": float(sum(int(r.sum()) for r in relevant)
                                              / info["relevant_cpaper"].numel()),
             "max_log_w_over_epochs": max((r.get("log_w_max", float("-inf")) for r in log),
                                          default=None),
             "at_best": log[best["best_epoch"]], "last": log[-1], "per_epoch": log}
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    graph, info = h5.graph_s1p()
    dataset_module._load_hetero_data = lambda _config: graph.clone()
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for arm in args.arms:
        for seed in args.seeds:
            entry = run(arm, seed, graph, info, args.epochs)
            results[f"s1p/{arm}/s{seed}"] = entry
            OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
            short = {k: entry[k] for k in ("test_acc", "val_acc", "best_epoch", "seconds",
                                           "coverage_relevant_cells", "max_log_w_over_epochs")}
            short["last"] = {k: v for k, v in entry["last"].items() if k != "epoch"}
            print("I3", arm, seed, json.dumps(short), flush=True)
            if (entry["max_log_w_over_epochs"] or 0) > 30:
                raise SystemExit(f"STOP: max log-weight > 30 in {arm} seed {seed}")


if __name__ == "__main__":
    main()
