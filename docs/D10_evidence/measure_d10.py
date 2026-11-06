"""D10 measurements and smoke run (run from H2S-NN/). One mode per process (clean peak memory):

  python ../docs/D10_evidence/measure_d10.py bench  <cooperative|symmetric>   # forward+backward
  python ../docs/D10_evidence/measure_d10.py smoke                            # 20 epochs, cooperative
  python ../docs/D10_evidence/measure_d10.py spectrum                          # T3 numbers

ACM, d = 32, factorized k = 2, tied facts + walk sampler (the post-D7 intended model).
No accuracy is computed in the smoke run.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path.cwd()))
from src.builders import build_optimizer, build_scheduler  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

GIB = 1024**3


def acm_config(coupling: str, epochs: int = 20) -> TrainConfig:
    with open("configs/experiments/acm_h2snn.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["model"].update({"d": 32, "householder_k": 2, "map_family": "factorized"})
    raw["data"].update({"fact_layer": "tied", "path_sampler": "walk"})
    raw["ablation"] = {"coupling": coupling}
    raw["epochs"] = epochs
    return TrainConfig.from_dict(raw)


def bench(coupling: str) -> None:
    cfg = acm_config(coupling)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    data = build_dataset(cfg).to(device)
    model = build_model(cfg, data).to(device)
    model.train()
    targets = data.target_y[data.target_train_mask]

    def step() -> None:
        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output.logits[data.target_train_mask], targets)
        loss.backward()

    step()  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    print(json.dumps({
        "coupling": coupling, "params": sum(p.numel() for p in model.parameters()),
        "peak_gib": torch.cuda.max_memory_allocated() / GIB,
        "time_s_median": statistics.median(times), "times_s": times,
        "gpu": torch.cuda.get_device_name(0),
    }))


def smoke() -> None:
    cfg = acm_config("cooperative")
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    data = build_dataset(cfg).to(device)
    model = build_model(cfg, data).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    targets = data.target_y[data.target_train_mask]
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data, epoch=epoch)
        loss = torch.nn.functional.cross_entropy(output.logits[data.target_train_mask], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        stats = model.diffusion_scalars()
        eta_lambda = model.eta_lambda_max(data.x, data.node_type, data.edge_index,
                                          data.edge_type, data.face_groups, data.num_nodes,
                                          steps=30)
        record = {"epoch": epoch, "loss": float(loss), **stats,
                  "w_mean": float(output.path_gate_mean), "w_max": float(output.path_gate_max),
                  "eta_lambda_max": eta_lambda}
        print(json.dumps(record), flush=True)
        if not math.isfinite(float(loss)) or eta_lambda >= 2.0:
            print(json.dumps({"stop": "divergence condition", "epoch": epoch}))
            break
    # 30-step estimate vs a converged one, at the final parameters
    model.eval()
    with torch.no_grad():
        h0 = model.encode_nodes(data.x, data.node_type)
        operator = model.build_operator(h0, data.edge_index, data.edge_type, data.face_groups,
                                        data.num_nodes)
        for steps in (30, 300):
            print(json.dumps({"final_lambda_max_steps": steps,
                              "lambda_max": model.lambda_max(operator, data.num_nodes, steps)}))


def spectrum() -> None:
    from tests import test_cooperative as T

    model = T._gate_model("cooperative")
    T._span_gate_heads(model)
    probe = T.SpectrumTests()
    imag, margin, ratio, lowest, (gate_low, gate_high) = probe._measure(model, probe.FIXTURES)
    worst = 0.0
    for seed in range(probe.FIXTURES):
        system = T.System(1000 + seed)
        pairs = [(int(a), int(b)) for a, b in system.paths.edge_index.t().tolist()]
        matrix = T._csnn_composed_operator(system.num_nodes, system.d, pairs, system.generator)
        worst = max(worst, float(torch.linalg.eigvals(matrix).imag.abs().max()))
    print(json.dumps({
        "fixtures": probe.FIXTURES, "gate_range": [gate_low, gate_high],
        "max_abs_imag_over_max_abs_lambda": imag, "max_lambda_minus_2max_gb": margin,
        "max_lambda_over_2max_gb": ratio, "min_eigenvalue": lowest,
        "csnn_control_max_abs_imag": worst}))


if __name__ == "__main__":
    {"bench": lambda: bench(sys.argv[2]), "smoke": smoke, "spectrum": spectrum}[sys.argv[1]]()
