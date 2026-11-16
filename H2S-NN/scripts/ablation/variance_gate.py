"""Card 4 K2 (decision B-10, spec 1.23 X-9b): the WISP proposal-variance gate on v3-B, before any
H2S-NN tuning. Port of `docs/WISP_evidence/h3_variance_gate.py` with its estimator unchanged:

- fixed model: ACM, seed 1, the central configuration (d 32, l_max 3, r 16, lr 1e-3, weight decay 1e-4,
  dropout 0.5 = encoder and classifier dropout), every other key at the protocol default of
  `configs/experiments/acm_h2snn_v3b.yaml` (HGB NC budget: 300 epochs, patience 30, validation
  Macro-F1; uniform proposal, resampled training draws), trained to early stopping; the fixed model is
  the selected (best-validation) checkpoint, in eval mode;
- q from that model's w_f m_f on one uniform draw (seed 43), detached (`proposal_from_mass`);
- 20 training draws per proposal at equal r (seeds 1000..1019, the same for both), built by the
  production draw (`build_node_classification_paths`, as the trainer does);
- 16 Rademacher probe matrices z [N, d] (generator seed 7, as H3);
- V(q) = mean_probes sum_rows (1/20) sum_draws (L_P^(i) z - mean)^2 (Welford over draws), with
  L_P = sum_f w_f m_f L_f the cells' vertex-2-cell Laplacian applied matrix-free (`PathCellTerm` on the
  model's forced transports; H3's `apply_path`), maps and g of the fixed model.
Gate (fixed): adaptive iff V(adaptive) <= 0.8 V(uniform). Writes the JSON given by --out (both V, the
ratio, the checkpoint's epoch, the resolved configuration). No accuracy is written.

Run from H2S-NN/:  python scripts/ablation/variance_gate.py --out ../docs/card4_evidence/k2_variance_gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.builders import build_experiment_components  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_node_classification_paths  # noqa: E402
from src.data.wisp_sampler import count_actions, proposal_from_mass  # noqa: E402
from src.models.map_operators import PathTransportOperator  # noqa: E402
from src.models.vertex_path_laplacian import PathCellTerm  # noqa: E402
from src.utils.misc import dataclass_to_dict, seed_everything  # noqa: E402

BASE_CONFIG = ROOT / "configs" / "experiments" / "acm_h2snn_v3b.yaml"
CENTRAL = {"seed": 1, "d": 32, "wisp_max_length": 3, "wisp_walks_per_anchor": 16, "lr": 1e-3,
           "weight_decay": 1e-4, "encoder_dropout": 0.5, "classifier_dropout": 0.5}
Q_SEED = 43
DRAW_SEEDS = tuple(range(1000, 1020))
NUM_PROBES = 16
PROBE_SEED = 7
GATE = 0.8


def central_config(device: str, output_dir: str) -> TrainConfig:
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["data"] = {**raw["data"], "wisp_max_length": CENTRAL["wisp_max_length"],
                   "wisp_walks_per_anchor": CENTRAL["wisp_walks_per_anchor"],
                   "wisp_proposal": "uniform"}
    raw["model"] = {**raw["model"], "d": CENTRAL["d"],
                    "encoder_dropout": CENTRAL["encoder_dropout"],
                    "classifier_dropout": CENTRAL["classifier_dropout"]}
    raw["optim"] = {**raw["optim"], "lr": CENTRAL["lr"], "weight_decay": CENTRAL["weight_decay"]}
    raw.update(seed=CENTRAL["seed"], device=device, exp_name="card4_k2_acm_seed1_central",
               output_dir=output_dir, log_every=10)
    return TrainConfig.from_dict(raw)


def draw(cfg: TrainConfig, data, seed: int, proposal):
    return build_node_classification_paths(
        config=cfg, fact_layer=data.fact_layer, node_slices=data.node_slices,
        node_type=data.node_type, node_type_names=data.node_type_names,
        edge_index=data.edge_index, edge_type=data.edge_type,
        relation_tuples=data.relation_tuples, seed=seed, proposal=proposal)


def path_laplacian(model, h0, geometry, maps, edge_index, groups, num_nodes):
    """`z -> sum_f w_f m_f L_f z` over the non-empty groups (the fixed model's maps and g)."""
    nonempty = tuple(g for g in groups if g.num_faces > 0)
    betas, weights, _ = model._cell_weights(h0, geometry, nonempty)
    terms = []
    for group, beta in zip(nonempty, betas):
        transport = PathTransportOperator(maps.src, maps.tgt, group.face_node_index,
                                          group.face_edge_index, edge_index, shared_scale=True,
                                          face_edge_sign=group.face_edge_sign)
        terms.append(PathCellTerm(transport, beta))

    def apply(z: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(z)
        for term in terms:
            out = out + term.apply(z, num_nodes)
        return out
    return apply, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="experiments/card4")
    args = parser.parse_args()

    cfg = central_config(args.device, args.output_dir)
    checkpoint = cfg.exp_dir / "checkpoints" / "best.pt"
    seed_everything(cfg.seed, cfg.device)
    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    start = time.time()
    if checkpoint.exists():
        raise SystemExit(f"{checkpoint} exists: the fixed model is trained once (rename the folder).")
    try:
        trainer.fit(data)
    finally:
        logger.close()
    train_seconds = time.time() - start
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    device = torch.device(cfg.device)
    model.to(device).eval()
    dev = data.to(device)
    self_inverse = data.fact_layer.self_inverse

    result = {"card": "docs/H2S-NN_card4_ablation_prep_card.md K2", "central": CENTRAL,
              "checkpoint_epoch": int(payload["epoch"]),
              "checkpoint_selection": f"best validation {cfg.checkpoint_metric} (early stopping, "
                                      f"patience {cfg.early_stopping_patience}, max {cfg.epochs} epochs)",
              "train_seconds": round(train_seconds, 1),
              "q_seed": Q_SEED, "draw_seeds": [DRAW_SEEDS[0], DRAW_SEEDS[-1]], "draws": len(DRAW_SEEDS),
              "probes": NUM_PROBES, "probe_seed": PROBE_SEED, "gate": f"V(adaptive) <= {GATE} V(uniform)",
              "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=ROOT).stdout.strip(),
              "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
              "resolved_config": json.loads(json.dumps(dataclass_to_dict(cfg), default=str))}
    with torch.no_grad():
        h0 = model.encode_nodes(dev.x, dev.node_type)
        geometry = model.geometry_inputs(h0, dev.edge_index, dev.edge_type)
        maps = model.node_to_edge(h=geometry, edge_index=dev.edge_index, edge_type=dev.edge_type,
                                  relation_embeddings=model.relation_embedding.weight).maps
        base_cpu = draw(cfg, data, Q_SEED, None)
        base = tuple(g.to(device) for g in base_cpu)
        _, base_weights = path_laplacian(model, h0, geometry, maps, dev.edge_index, base,
                                         dev.num_nodes)
        q = proposal_from_mass(base_cpu, tuple(w.detach() for w in base_weights), data.node_type,
                               len(data.node_type_names),
                               count_actions(data.edge_type, self_inverse),
                               cfg.data.wisp_max_length, self_inverse)
        result["q_len"] = q.length[2:].tolist()
        log_w = torch.cat(base_weights).log()
        result["log_w_on_q_draw"] = {"max": float(log_w.max()), "mean": float(log_w.mean()),
                                     "min": float(log_w.min())}
        generator = torch.Generator().manual_seed(PROBE_SEED)
        probes = [(torch.randint(0, 2, (data.num_nodes, model.d), generator=generator) * 2 - 1)
                  .float().to(device) for _ in range(NUM_PROBES)]
        for name, proposal in (("uniform", None), ("adaptive", q)):
            mean = [torch.zeros_like(z) for z in probes]
            m2 = [torch.zeros_like(z) for z in probes]
            seconds, cells = [], []
            for n, seed in enumerate(DRAW_SEEDS, start=1):
                t = time.perf_counter()
                groups = tuple(g.to(device) for g in draw(cfg, data, seed, proposal))
                seconds.append(time.perf_counter() - t)
                cells.append(sum(g.num_faces for g in groups))
                apply, _ = path_laplacian(model, h0, geometry, maps, dev.edge_index, groups,
                                          dev.num_nodes)
                for k, z in enumerate(probes):
                    y = apply(z)
                    delta = y - mean[k]
                    mean[k] += delta / n
                    m2[k] += delta * (y - mean[k])
            variance = sum(float(m.sum()) / len(DRAW_SEEDS) for m in m2) / len(probes)
            result[name] = {"V": variance, "draw_seconds_median": sorted(seconds)[len(seconds) // 2],
                            "cells_per_draw_mean": sum(cells) / len(cells)}
            print(name, json.dumps(result[name]), flush=True)
    ratio = result["adaptive"]["V"] / result["uniform"]["V"]
    result["ratio"] = ratio
    result["reduction"] = 1.0 - ratio
    result["adaptive_adopted"] = bool(ratio <= GATE)
    result["verdict"] = ("ratio <= 0.8: adaptive passes the gate -> author decision (stop item)"
                         if ratio <= GATE else "ratio > 0.8: uniform stays (no code change)")
    print(json.dumps({k: result[k] for k in ("checkpoint_epoch", "ratio", "verdict")}, indent=1))
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
