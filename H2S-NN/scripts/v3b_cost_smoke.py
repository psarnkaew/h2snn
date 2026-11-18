"""v3-B card U5: 2-epoch cost smoke of the anchored solve on IMDB (no accuracy numbers).

Runs `epochs` normal training steps (forward + backward + optimizer step, full batch, a fresh WISP
training draw per epoch as the trainer does) of the config, then one evaluation forward per epoch
(the fixed evaluation draw; warm start from the previous epoch). Records per epoch: loss, the
gradient norm of every parameter group (incl. theta_tau), PCG iterations forward / backward,
convergence, tau, max log-weight, |E_P|, |E_shared|, eliminated share, time and peak memory.
Writes JSON. `r` is the config's (the calibrated value; never overridden or halved, spec E11/E14):
on CUDA out-of-memory the error is written to the JSON and the script exits with status 1 (L8a).

    conda run -n h2snn python scripts/v3b_cost_smoke.py --config configs/experiments/imdb_h2snn_v3b.yaml \
        --device cuda --epochs 2 --out ../docs/v3B_evidence/v3b_cost_smoke_imdb.json

L7 (ACM / DBLP / IMDB cost gate): add `--final_tolerance_step` for one extra training forward +
backward at the final tolerance 1e-4 (PCG iterations of the spec §6 gate), recorded separately.

Model change TD-1 (card M6): `--code_root <H2S-NN dir>` imports `src` from that tree (the same script
times `v3b` and the TD-1 head in one job); `--any_r` accepts an r that is not the calibrated value (the
card 2 central point, r = 16; spec 1.15 J2 made r a tuned axis). Per epoch it also records the forward
and backward time of the training step, the forward solve's time per PCG iteration (one system apply)
and the map build time, and once the float storage of the map operators (per side, unique, per fact).
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
if "--code_root" in sys.argv:  # before any `src` import: the tree to time (card M6)
    PROJECT_ROOT = Path(sys.argv[sys.argv.index("--code_root") + 1]).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import WISP_WALKS_PER_ANCHOR, TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code_root", type=Path, default=PROJECT_ROOT,
                        help="H2S-NN tree whose src is timed (default: this script's tree)")
    parser.add_argument("--any_r", action="store_true",
                        help="accept a non-calibrated r (central point of the tuned r axis)")
    parser.add_argument("--final_tolerance_step", action="store_true",
                        help="after the epochs, one extra training forward + backward (fresh draw, "
                             "no optimizer step) at the final solver tolerance, recorded separately")
    return parser.parse_args()


def load_config(path: Path, device: str, any_r: bool = False) -> TrainConfig:
    """The config as written (only the device is set). `r` comes from the config and must be the
    dataset's calibrated value when a table exists (spec v1.6 E11, v1.7 E14); it is never
    overridden or halved here."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["device"] = device
    cfg = TrainConfig.from_dict(payload)
    table = WISP_WALKS_PER_ANCHOR.get(cfg.data.dataset_name, {})
    calibrated = table.get(cfg.data.wisp_max_length)
    if table and cfg.data.wisp_walks_per_anchor != calibrated and not any_r:
        raise ValueError(
            f"resolved r = {cfg.data.wisp_walks_per_anchor} is not the calibrated value "
            f"{calibrated} for {cfg.data.dataset_name!r} at l_max {cfg.data.wisp_max_length} "
            "(spec E11/E14: r is never changed).")
    return cfg


def group_gradient_norms(model) -> dict[str, float]:
    groups = {
        "node_encoder": model.node_encoder,
        "relation_embedding": model.relation_embedding,
        "fact_maps": model.node_to_edge,
        "relational_context": getattr(model, "relational_context", None),  # base code only (1.24)
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def map_storage(maps, num_facts: int) -> dict:
    """Float elements the two map operators hold (the lazily built layout cache excluded): per
    side, unique over both sides (shared tensors counted once), and per fact (tensors with one row
    per fact). Works for the Householder (v3b) and the TD-1 operators."""
    def tensors(operator):
        found = []
        for name, value in vars(operator).items():
            if name == "_layout":
                continue
            for item in (value if isinstance(value, (tuple, list)) else (value,)):
                if isinstance(item, torch.Tensor) and item.is_floating_point():
                    found.append(item)
        return found

    unique = {}
    out = {}
    for side in ("src", "tgt"):
        operator = getattr(maps, side)
        out[f"stored_elements_{side}"] = int(operator.stored_elements())
        for tensor in tensors(operator):
            unique[(tensor.data_ptr(), tensor.numel())] = tensor
    out["unique_floats"] = int(sum(t.numel() for t in unique.values()))
    per_fact = sum(t.numel() for t in unique.values() if t.dim() >= 1 and t.size(0) == num_facts)
    out["per_fact_floats"] = per_fact / num_facts
    out["per_relation_floats"] = out["unique_floats"] - per_fact
    return out


class SolveTimer:
    """Wraps the model module's `anchored_solve`: wall time of each forward solve (synchronized)."""

    def __init__(self, module, device: torch.device) -> None:
        self.module, self.device, self.seconds = module, device, []
        self.original = module.anchored_solve

        def timed(*args, **kwargs):
            _sync(self.device)
            start = time.perf_counter()
            result = self.original(*args, **kwargs)
            _sync(self.device)
            self.seconds.append(time.perf_counter() - start)
            return result

        module.anchored_solve = timed


def map_build_seconds(model, data, device: torch.device, repeats: int = 10) -> float:
    """Mean time of the node->edge map build (eval, no grad) on the evaluation graph."""
    model.eval()
    with torch.no_grad():
        h0 = model.encode_nodes(data.x, data.node_type)
        geometry = model.geometry_inputs(h0, data.edge_index, data.edge_type)
        model.node_to_edge(h=geometry, edge_index=data.edge_index, edge_type=data.edge_type,
                           relation_embeddings=model.relation_embedding.weight)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model.node_to_edge(h=geometry, edge_index=data.edge_index, edge_type=data.edge_type,
                               relation_embeddings=model.relation_embedding.weight)
        _sync(device)
    return (time.perf_counter() - start) / repeats


def run(cfg: TrainConfig, epochs: int, final_tolerance_step: bool = False) -> dict:
    import src.models.h2snn_model as model_module
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
    timer = SolveTimer(model_module, device)
    captured = {}

    def keep_maps(module, inputs, output):  # returns None: the module's output is not replaced
        captured.setdefault("maps", output.maps)

    hook = model.node_to_edge.register_forward_hook(keep_maps)
    record: dict = {
        "code_root": str(PROJECT_ROOT),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
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
        timer.seconds.clear()
        output = model(train_data, epoch=epoch)
        loss, _ = trainer._compute_metrics(output, train_data, train_data.target_train_mask, epoch)
        _sync(device)
        forward_seconds = time.perf_counter() - t_train
        solve_seconds = sum(timer.seconds)
        t_backward = time.perf_counter()
        loss.backward()
        _sync(device)
        backward_seconds = time.perf_counter() - t_backward
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
            "train_forward_seconds": round(forward_seconds, 4),
            "train_backward_seconds": round(backward_seconds, 4),
            "forward_solve_seconds": round(solve_seconds, 4),
            "forward_solve_seconds_per_pcg_iteration": round(
                solve_seconds / max(train_stats.get("pcg_fwd_iters", 0.0), 1.0), 6),
            "eval_forward_seconds": round(eval_seconds, 3),
            "epoch_seconds": round(draw_seconds + train_seconds, 3),
            "peak_memory_mb": round(peak_mb, 1),
        })
        print(json.dumps(record["epochs"][-1], indent=None), flush=True)
    hook.remove()
    if "maps" in captured:
        record["map_storage"] = map_storage(captured["maps"], int(prepared.edge_index.size(1)))
    record["map_build_seconds"] = round(map_build_seconds(model, prepared, device), 6)
    if final_tolerance_step:
        record["final_tolerance_step"] = final_tolerance_probe(model, trainer, data, prepared,
                                                               cfg, epochs, device)
        print(json.dumps(record["final_tolerance_step"], indent=None), flush=True)
    return record


def final_tolerance_probe(model, trainer, data, prepared, cfg: TrainConfig, epochs: int,
                          device: torch.device) -> dict:
    """One training forward + backward at the final tolerance (spec §6 cost gate: PCG iterations
    at tol 1e-4). Fresh training draw (seed + `epochs`, as the next epoch would get); the epoch
    passed to the model is `solver_warmup_epochs`, the first epoch of the final tolerance. No
    optimizer step, no clipping. Peak memory is measured for this step alone."""
    t_draw = time.perf_counter()
    train_data = trainer.training_draw(data, prepared, epochs)
    draw_seconds = time.perf_counter() - t_draw
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t_step = time.perf_counter()
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(train_data, epoch=cfg.model.solver_warmup_epochs)
    loss, _ = trainer._compute_metrics(output, train_data, train_data.target_train_mask,
                                       cfg.model.solver_warmup_epochs)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - t_step
    peak_mb = (torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda"
               else float("nan"))
    return {
        "draw_seed_offset": epochs,
        "model_epoch": cfg.model.solver_warmup_epochs,
        "loss_finite": bool(torch.isfinite(loss)),
        "train_solve": dict(model.last_solver_stats),
        "draw_seconds": round(draw_seconds, 3),
        "forward_backward_seconds": round(step_seconds, 3),
        "peak_memory_mb_this_step": round(peak_mb, 1),
    }


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {path}", flush=True)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.device, args.any_r)
    try:
        record = run(cfg, args.epochs, args.final_tolerance_step)
    except torch.cuda.OutOfMemoryError as exc:  # r is never changed (spec E11/E14): record, fail
        write_record(args.out, {
            "config": str(cfg.exp_name), "dataset": cfg.data.dataset_name,
            "walks_per_anchor": cfg.data.wisp_walks_per_anchor,
            "stopped": "CUDA out of memory (r is never changed; spec v1.6 E11, v1.7 E14)",
            "error": str(exc)[:500],
        })
        return 1
    write_record(args.out, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
