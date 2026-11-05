"""D1 microbenchmark on ACM: one forward+backward, 1 warm-up + 5 timed repeats.

    python measure_d1.py <stage: full|maps> <family: dense|factorized> <k> <d>

Prints one JSON line. Each configuration runs in its own process so peak-memory stats are
not polluted by earlier allocations. The process is capped at 95% of device memory so an
oversize run fails with a clean OOM instead of spilling to system RAM (WDDM).
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
from src.models.coboundary_ops import CoboundaryOps  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

SCRATCH = Path(__file__).resolve().parent
CACHE = SCRATCH / "acm_prepared.pt"
GIB = 1024**3


def load_data(cfg: TrainConfig):
    if CACHE.exists():
        return torch.load(CACHE, weights_only=False)
    from src.data.dataset import build_dataset

    data = build_dataset(cfg)
    torch.save(data, CACHE)
    return data


def main() -> None:
    stage, family, k, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    with open("configs/experiments/acm_h2snn.yaml", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["model"].update(
        {"map_family": family, "householder_k": k, "d": d,
         "node_edge_memory_guard_fraction": 0.95}
    )
    cfg = TrainConfig.from_dict(raw)
    torch.cuda.set_per_process_memory_fraction(0.95)
    device = torch.device("cuda")
    data = load_data(cfg).to(device)
    torch.manual_seed(0)
    model = build_model(cfg, data).to(device).train()
    record: dict[str, object] = {
        "stage": stage, "family": family, "k": k if family == "factorized" else None, "d": d,
        "M": data.num_edges, "faces": sum(g.num_faces for g in data.face_groups),
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_node_to_edge": sum(p.numel() for p in model.node_to_edge.parameters()),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / GIB, 2),
        "torch": torch.__version__,
    }
    targets = data.target_y[data.target_train_mask]

    if stage == "full":
        def step() -> None:
            output = model(data)
            logits = output.logits[data.target_train_mask]
            torch.nn.functional.cross_entropy(logits, targets).backward()
    else:
        # Map stage only: node->edge maps + K applications of L0 over the observed edges.
        h0 = torch.randn(data.num_nodes, d, device=device, requires_grad=True)
        src_idx, tgt_idx = data.edge_index

        def step() -> None:
            out = model.node_to_edge(h=h0, edge_index=data.edge_index, edge_type=data.edge_type,
                                     relation_embeddings=model.relation_embedding.weight)
            x = h0
            for _ in range(cfg.model.K):
                if family == "dense":
                    x = x - 0.05 * CoboundaryOps.L0(x, out.F_src, out.F_tgt, src_idx, tgt_idx,
                                                    data.num_nodes)
                else:
                    x = x - 0.05 * CoboundaryOps.L0_matrix_free(x, out.maps, src_idx, tgt_idx,
                                                                data.num_nodes)
            x.square().sum().backward()

        if family == "factorized":
            maps = model.node_to_edge(h=h0, edge_index=data.edge_index, edge_type=data.edge_type,
                                      relation_embeddings=model.relation_embedding.weight).maps
            record["map_stored_elements"] = maps.src.stored_elements() + maps.tgt.stored_elements()
            del maps
        else:
            record["map_stored_elements"] = 2 * data.num_edges * d * d

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
