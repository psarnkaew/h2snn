"""D4 microbenchmark on ACM, architecture v2: one forward + backward, 1 warm-up + 5 timed.

    python measure_d4.py <family: dense|factorized> <k> <d> <checkpointing: 0|1>

Prints one JSON line; one configuration per process so peak-memory stats are clean. Capped
at 95% of device memory (as D1) so an oversize run fails with a clean OOM.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "acm_prepared.pt"
GIB = 1024**3


def acm_config(family: str, k: int, d: int, checkpointing: bool, epochs: int | None = None):
    with open("configs/experiments/acm_h2snn.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["model"].update({"architecture": "v2", "map_family": family, "householder_k": k,
                         "d": d, "checkpointing": checkpointing,
                         "node_edge_memory_guard_fraction": 0.95})
    if epochs is not None:
        raw["epochs"] = epochs
    return TrainConfig.from_dict(raw)


def load_data(cfg: TrainConfig):
    if CACHE.exists():
        return torch.load(CACHE, weights_only=False)
    from src.data.dataset import build_dataset

    data = build_dataset(cfg)
    torch.save(data, CACHE)
    return data


def main() -> None:
    family, k, d, checkpointing = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))
    cfg = acm_config(family, k, d, checkpointing)
    torch.cuda.set_per_process_memory_fraction(0.95)
    device = torch.device("cuda")
    data = load_data(cfg).to(device)
    torch.manual_seed(0)
    model = build_model(cfg, data).to(device).train()
    record: dict[str, object] = {
        "arch": "v2", "family": family, "k": k if family == "factorized" else None, "d": d,
        "checkpointing": checkpointing, "N": data.num_nodes, "M": data.num_edges,
        "faces": sum(g.num_faces for g in data.face_groups),
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_node_to_edge": sum(p.numel() for p in model.node_to_edge.parameters()),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / GIB, 2),
        "torch": torch.__version__,
    }
    targets = data.target_y[data.target_train_mask]

    def step() -> None:
        output = model(data)
        logits = output.logits[data.target_train_mask]
        torch.nn.functional.cross_entropy(logits, targets).backward()

    try:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step()  # warm-up
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        record["warmup_peak_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - start))
            model.zero_grad(set_to_none=True)
        record["peak_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
        record["peak_above_inputs_gib"] = round((torch.cuda.max_memory_allocated() - base) / GIB, 3)
        record["ms_mean"] = round(statistics.mean(times), 1)
        record["ms_std"] = round(statistics.stdev(times), 1)
        record["status"] = "ok"
    except (torch.OutOfMemoryError, RuntimeError) as exc:  # reported, not hidden
        record["status"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        record["peak_gib_before_failure"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
