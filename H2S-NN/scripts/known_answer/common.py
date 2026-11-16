"""Card 4 K3-K5: shared pieces of the known-answer runs under v3-B (pre-registration
`docs/known_answer_preregistration_draft.md`, FROZEN 2026-11-16; its "Frozen settings" paragraph rules).

Frozen settings (B-9): `configs/experiments/acm_h2snn_v3b.yaml` model block (`propagation: anchored`,
solver values of spec §6, never tuned), `d` 32, `tau_init` 1, AdamW lr 1e-3 (weight decay 1e-4, the
protocol default), scalar group (theta_tau, b) at lr 1e-2 (`scalar_lr_scale` 10), no scheduler, no
gradient clipping, 100 epochs (patience 100: no early stop inside the 100 epochs), validation-accuracy
selection, `resample_paths: false` (one draw = training and evaluation, seeded by the run seed), the
graph's own 50/25/25 split (`h5_synthetic._split`, seed 1). Graphs are imported unchanged from
`docs/WISP_evidence/` (S1'-B, S2') or from `s3_graph.py` (S3). The oracle lives here, never in `src/`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT.parent / "docs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DOCS / "WISP_evidence"))

import src.data.dataset as dataset_module  # noqa: E402
from src.builders import (_checkpointer, build_logger, build_optimizer,  # noqa: E402
                          build_scheduler, build_trainer)
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402
from src.models.map_operators import PathTransportOperator  # noqa: E402
from src.models.sheaf_energy import SheafEnergyOperator, build_cell_group  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

BASE_CONFIG = ROOT / "configs" / "experiments" / "acm_h2snn_v3b.yaml"
EPOCHS = 100
BETAS = {"0": 0.0, "log64": math.log(64), "log512": math.log(512)}
PROBE_LAMBDA = 1e-3  # Part K K0: fixed, never tuned


def frozen_config(exp_name: str, seed: int, max_length: int, walks: int, target: str,
                  arm: str = "full", ablation: dict | None = None,
                  output_dir: str = "experiments/card4_known_answer") -> TrainConfig:
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["data"] = {**raw["data"], "path_source": "wisp", "wisp_max_length": max_length,
                   "wisp_walks_per_anchor": walks, "resample_paths": False,
                   "wisp_proposal": "uniform"}
    raw["model"] = {**raw["model"], "d": 32, "tau_init": 1.0, "arm": arm}
    raw["ablation"] = {"coupling": "symmetric", **(ablation or {})}
    raw["optim"] = {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "scheduler": "none",
                    "grad_clip": 0.0, "scalar_lr_scale": 10.0}
    raw.update(seed=seed, epochs=EPOCHS, early_stopping_patience=EPOCHS, device="cuda",
               checkpoint_metric="acc", checkpoint_mode="max", exp_name=exp_name,
               output_dir=output_dir, log_every=1000)
    cfg = TrainConfig.from_dict(raw)
    cfg.data = dataclasses.replace(cfg.data, target_node_type=target)
    return cfg


def install_graph(graph) -> None:
    """`build_dataset` returns a clone of `graph` (only the loader is replaced; the fact layer, WISP
    draw, labels and split run as in production)."""
    dataset_module._load_hetero_data = lambda _config: graph.clone()


def relevance_by_group(data, carrier: str, relevant_local: torch.Tensor) -> list[torch.Tensor]:
    """Per non-empty face group, a bool per cell: the cell has a relevant carrier at an end
    (`h5_synthetic.relevance_of_cells`)."""
    start = data.node_slices[carrier][0]
    relevant = torch.zeros(data.num_nodes, dtype=torch.bool)
    relevant[start + relevant_local] = True
    out = []
    for group in data.face_groups:
        if group.num_faces:
            nodes = group.face_node_index.cpu()
            out.append(relevant[nodes[:, 0]] | relevant[nodes[:, -1]])
    return out


def install_oracle(model, data, relevant: list[torch.Tensor], beta: float,
                   device: torch.device) -> None:
    """Fixed-beta oracle (Part K K2, `k2_capacity.install_oracle`): w_f = exp(b + beta o_f); beta a
    buffer (requires_grad only to read dL/dbeta, never optimised); b its own scalar in the scalar group
    (b0 = log path_weight_init); g unused for the weights; everything else trains; nothing clamped."""
    model.oracle_b = nn.Parameter(torch.tensor(math.log(model.model_config.path_weight_init)))
    model.register_buffer("oracle_beta", torch.tensor(beta, device=device, requires_grad=True))
    base_scalars = model.scalar_parameters
    model.scalar_parameters = lambda: list(base_scalars()) + [model.oracle_b]
    # Keyed by the group's cell length (one WISP group per length), not by its size as in Part K:
    # two lengths can hold the same number of cells (S3 at r 2560: 128,000 cells of length 3 and 4).
    lengths = [g.length for g in data.face_groups if g.num_faces]
    by_length = dict(zip(lengths, (r.float() for r in relevant)))
    if len(by_length) != len(relevant):
        raise ValueError("the oracle needs one face group per cell length")

    def weights(latent, group, like):
        o = by_length[group.length].to(device=like.device, dtype=like.dtype)
        if o.numel() != group.num_faces:
            raise ValueError("oracle relevance does not match the group's cells")
        return torch.exp(model.oracle_b + model.oracle_beta * o)
    model._path_weights = weights


def run_identity() -> dict:
    return {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip(),
            "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
            "array_task": os.environ.get("SLURM_ARRAY_TASK_ID", "")}


def train(cfg: TrainConfig, graph, before_optimizer: Callable | None = None,
          epoch_log: Callable | None = None):
    """Build data and model, apply `before_optimizer(model, data)` (oracle), train 100 epochs, reload
    the best-validation checkpoint. Returns (data, model, per-epoch log, best record, seconds)."""
    if (cfg.exp_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    install_graph(graph)
    seed_everything(cfg.seed, cfg.device)
    start = time.time()
    data = build_dataset(cfg)
    model = build_model(cfg, data)
    device = torch.device(cfg.device)
    if before_optimizer is not None:
        before_optimizer(model, data)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    logger = build_logger(cfg)
    trainer = build_trainer(model=model, optimizer=optimizer, scheduler=scheduler, cfg=cfg,
                            logger=logger, checkpointer=_checkpointer(cfg))
    log: list[dict] = []

    def callback(epoch, tr, va, te):
        row = {"epoch": epoch, "train_acc": tr["acc"], "val_acc": va["acc"],
               "tau": float(model.tau().detach())}
        if epoch_log is not None:
            row.update(epoch_log(model))
        log.append(row)

    try:
        trainer.fit(data, epoch_callback=callback)
    finally:
        logger.close()
    seconds = round(time.time() - start, 1)
    best = json.loads((cfg.exp_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
    payload = torch.load(cfg.exp_dir / "checkpoints" / "best.pt", map_location=device,
                         weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return data, model, log, best, seconds


def model_geometry(model, dev):
    """(h0, geometry, maps) of the model in eval mode on the evaluation draw."""
    with torch.no_grad():
        h0 = model.encode_nodes(dev.x, dev.node_type)
        geometry = model.geometry_inputs(h0, dev.edge_index, dev.edge_type)
        maps = model.node_to_edge(h=geometry, edge_index=dev.edge_index, edge_type=dev.edge_type,
                                  relation_embeddings=model.relation_embedding.weight).maps
    return h0, geometry, maps


def cell_weights(model, dev, h0, geometry):
    groups = tuple(g for g in dev.face_groups if g.num_faces > 0)
    with torch.no_grad():
        betas, weights, _ = model._cell_weights(h0, geometry, groups)
    return groups, betas, weights


def relevant_share_of_degree(model, dev, relevant, arm: str) -> dict:
    """Median over the target nodes of (a) the relevant cells' share of the node's scalar degree
    D_v = sum_e a_e + sum_f beta_f (node-cell term; 0 under covering_only, whose cells do not enter
    D_v) and (b) the relevant share of the cell mass sum_f beta_f at the target (the selection ratio)."""
    h0, geometry, maps = model_geometry(model, dev)
    groups, betas, _ = cell_weights(model, dev, h0, geometry)
    with torch.no_grad():
        cell_groups = [build_cell_group(maps, g, dev.edge_index, b) for g, b in zip(groups, betas)]
        op = SheafEnergyOperator(maps, dev.edge_index, dev.num_nodes, cell_groups,
                                 node_cell=arm != "covering_only", fact_cell=arm != "hub_A")
        d_v = op.degree()[0]
        rel_mass = torch.zeros(dev.num_nodes, dtype=d_v.dtype, device=d_v.device)
        all_mass = torch.zeros_like(rel_mass)
        for g, b, r in zip(groups, betas, relevant):
            mask = r.to(b.device, b.dtype)
            per = (b * mask).unsqueeze(1).expand(-1, g.length + 1).reshape(-1)
            rel_mass.index_add_(0, g.face_node_index.reshape(-1), per)
            all_mass.index_add_(0, g.face_node_index.reshape(-1),
                                b.unsqueeze(1).expand(-1, g.length + 1).reshape(-1))
        targets = dev.target_indices
        in_degree = rel_mass if arm != "covering_only" else torch.zeros_like(rel_mass)
        share_d = (in_degree[targets] / d_v[targets]).median()
        share_mass = (rel_mass[targets] / all_mass[targets].clamp_min(1e-30)).median()
    return {"relevant_share_D_median": float(share_d),
            "relevant_share_cell_mass_median": float(share_mass)}


def dispersion(model, dev, relevant, target_type: str) -> float:
    """1 - ||mean_f (2y_f - 1) t_f / ||t_f|| ||, t_f = T_P^T T_Q* h0(Q*) over the relevant cells
    (`k2_capacity.probe_and_dispersion`, production `PathTransportOperator`)."""
    h0, _, maps = model_geometry(model, dev)
    groups = [g for g in dev.face_groups if g.num_faces]
    vectors, labels = [], []
    start, end = dev.node_slices[target_type]
    with torch.no_grad():
        for group, rel in zip(groups, relevant):
            rel = rel.to(h0.device)
            if not bool(rel.any()):
                continue
            nodes = group.face_node_index[rel]
            transport = PathTransportOperator(
                maps.src, maps.tgt, nodes, group.face_edge_index[rel], dev.edge_index,
                shared_scale=True, face_edge_sign=group.face_edge_sign[rel])
            states = transport.apply(h0)                                # [R, L+1, d]
            p_at_0 = (nodes[:, 0] >= start) & (nodes[:, 0] < end)
            p_pos = torch.where(p_at_0, 0, nodes.size(1) - 1)
            q_pos = nodes.size(1) - 1 - p_pos
            rows = torch.arange(nodes.size(0), device=h0.device)
            residual = torch.zeros_like(states)
            residual[rows, p_pos] = states[rows, q_pos]
            carried = transport.apply_T(residual, dev.num_nodes)        # T_P^T T_Q* h0(Q*) at P
            p_nodes = nodes[rows, p_pos]
            vectors.append(carried[p_nodes])
            labels.append(dev.y[p_nodes].to(h0.dtype))
        t = torch.cat(vectors)
        label = torch.cat(labels)
        signed = (2 * label - 1).unsqueeze(1) * t / t.norm(dim=1, keepdim=True).clamp_min(
            torch.finfo(t.dtype).tiny)
    return float(1 - signed.mean(0).norm())


def log_weight_gap(model, dev, relevant) -> dict:
    """Mean learned log w_f of relevant minus irrelevant cells (evaluation draw, eval mode)."""
    h0, geometry, _ = model_geometry(model, dev)
    _, _, weights = cell_weights(model, dev, h0, geometry)
    log_w = torch.cat(weights).log()
    rel = torch.cat([r.to(log_w.device) for r in relevant])
    return {"log_w_relevant": float(log_w[rel].mean()), "log_w_irrelevant": float(log_w[~rel].mean()),
            "log_w_gap": float(log_w[rel].mean() - log_w[~rel].mean())}


def coverage(data, relevant) -> float:
    """Share of the relevant carriers that end a cell of the draw."""
    return float(sum(int(r.sum()) for r in relevant))


def logistic_probe(features: torch.Tensor, y: torch.Tensor, train: torch.Tensor,
                   test: torch.Tensor) -> float:
    """Part K K0 probe: standardised features (training mean/std), logistic regression with bias,
    mean BCE + lambda/2 ||w||^2 (lambda 1e-3, bias unpenalised), LBFGS(lr 1, max_iter 500, history 50,
    tol_grad 1e-9, tol_change 1e-12, strong Wolfe), float64, w0 = 0; test accuracy (threshold 0.5)."""
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
            + 0.5 * PROBE_LAMBDA * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = (x[test] @ w + b) > 0
    return float((pred.long() == y[test].long()).double().mean())


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
