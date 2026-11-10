"""v3-B card U5: 2-epoch cost smoke of the anchored solve on IMDB (no accuracy numbers).

Runs `epochs` normal training steps (forward + backward + optimizer step, full batch, a fresh WISP
training draw per epoch as the trainer does) of the config, then one evaluation forward per epoch
(the fixed evaluation draw; warm start from the previous epoch). Records per epoch: loss, the
gradient norm of every parameter group (incl. theta_tau), PCG iterations forward / backward,
convergence, tau, max log-weight, |E_P|, |E_shared|, eliminated share, time and peak memory. If the
GPU runs out of memory, `r` is halved and the run restarts (recorded). Writes JSON.

    conda run -n h2snn python scripts/v3b_cost_smoke.py --config configs/experiments/imdb_h2snn_v3b.yaml \
        --device cuda --epochs 2 --out ../docs/v3B_evidence/v3b_cost_smoke_imdb.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--walks_per_anchor", type=int, default=0,
                        help="override r (0 = the config's resolved value); the OOM fallback halves it")
    return parser.parse_args()


def load_config(path: Path, device: str, walks: int) -> TrainConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["device"] = device
    if walks:
        payload.setdefault("data", {})["wisp_walks_per_anchor"] = walks
    return TrainConfig.from_dict(payload)


def group_gradient_norms(model) -> dict[str, float]:
    groups = {
        "node_encoder": model.node_encoder,
        "relation_embedding": model.relation_embedding,
        "fact_maps": model.node_to_edge,
        "relational_context": model.relational_context,
        "g_path_encoder": model.path_encoder,
        "g_path_gate": model.path_gate,
        "free_cell_maps": model.free_cell_maps,
        "readout": model.readout,
    }
    norms: dict[str, float] = {}
    for name, module in groups.items():
        if module is None:
            continue
        params = list(module.parameters())
        if not params:
            continue
        norms[name] = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in params if p.grad is not None))
    if model.theta_tau is not None and model.theta_tau.grad is not None:
        norms["theta_tau"] = float(model.theta_tau.grad.abs())
    return norms


def run(cfg: TrainConfig, epochs: int) -> dict:
    from src.builders import build_dataset, build_model, build_optimizer
    from src.trainers.trainer import Trainer

    device = torch.device(cfg.device)
    seed_everything(cfg.seed, cfg.device)
    t0 = time.perf_counter()
    data = build_dataset(cfg)
    load_seconds = time.perf_counter() - t0
    model = build_model(cfg, data).to(device)
    optimizer = build_optimizer(model, cfg)
    groups = [{"lr": g["lr"], "num_params": sum(p.numel() for p in g["params"])}
              for g in optimizer.param_groups]
    prepared = data.to(device)
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=None,
                      criterion=torch.nn.CrossEntropyLoss(), cfg=cfg, logger=None,  # type: ignore[arg-type]
                      checkpointer=None)  # type: ignore[arg-type]
    record: dict = {
        "config": str(cfg.exp_name), "device": str(device), "dataset": cfg.data.dataset_name,
        "d": cfg.model.d, "arm": cfg.model.arm, "walks_per_anchor": cfg.data.wisp_walks_per_anchor,
        "l_max": cfg.data.wisp_max_length, "solver": {
            "tolerance": cfg.model.solver_tolerance,
            "warmup_tolerance": cfg.model.solver_warmup_tolerance,
            "warmup_epochs": cfg.model.solver_warmup_epochs,
            "max_iterations": cfg.model.solver_max_iterations},
        "num_nodes": data.num_nodes, "num_facts": data.num_edges,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_groups": groups, "load_seconds": round(load_seconds, 2), "epochs": [],
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        t_draw = time.perf_counter()
        train_data = trainer.training_draw(data, prepared, epoch)
        draw_seconds = time.perf_counter() - t_draw
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_train = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        output = model(train_data, epoch=epoch)
        loss, _ = trainer._compute_metrics(output, train_data, train_data.target_train_mask, epoch)
        loss.backward()
        norms = group_gradient_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - t_train
        train_stats = dict(model.last_solver_stats)
        path_stats = dict(model.last_path_stats)
        t_eval = time.perf_counter()
        model.eval()
        with torch.no_grad():
            model(prepared, epoch=-1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        eval_seconds = time.perf_counter() - t_eval
        eval_stats = dict(model.last_solver_stats)
        peak_mb = (torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda"
                   else float("nan"))
        record["epochs"].append({
            "epoch": epoch,
            "loss": float(loss),
            "loss_finite": bool(torch.isfinite(loss)),
            "gradient_norms": norms,
            "all_groups_nonzero": all(v > 0 for v in norms.values()),
            "tau_after_step": float(model.tau()),
            "path_log_w_max": path_stats.get("path_log_w_max"),
            "path_log_w_mean": path_stats.get("path_log_w_mean"),
            "train_solve": train_stats,
            "eval_solve": eval_stats,
            "draw_seconds": round(draw_seconds, 3),
            "train_step_seconds": round(train_seconds, 3),
            "eval_forward_seconds": round(eval_seconds, 3),
            "epoch_seconds": round(draw_seconds + train_seconds, 3),
            "peak_memory_mb": round(peak_mb, 1),
        })
        print(json.dumps(record["epochs"][-1], indent=None), flush=True)
    return record


def main() -> None:
    args = parse_args()
    walks = args.walks_per_anchor
    attempts = []
    while True:
        cfg = load_config(args.config, args.device, walks)
        try:
            record = run(cfg, args.epochs)
            break
        except torch.cuda.OutOfMemoryError as exc:  # card: halve r until it fits, record it
            r = cfg.data.wisp_walks_per_anchor
            attempts.append({"walks_per_anchor": r, "error": str(exc)[:200]})
            print(f"OOM at r = {r}; halving", flush=True)
            torch.cuda.empty_cache()
            if r <= 4:
                record = {"stopped": "OOM on IMDB at r <= 4 (card stop condition)",
                          "attempts": attempts}
                break
            walks = max(r // 2, 1)
    record["oom_attempts"] = attempts
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
