"""H6 smoke and cost (reduced settings, pre-registered in preregistration.md, H6).
Run from H2S-NN/:  python ../docs/WISP_evidence/h6_smoke_cost.py [--only NAME ...]
Writes docs/WISP_evidence/h6_smoke_cost.json. Accuracy is logged by the trainer (metrics.jsonl)
and is NOT copied here (no performance results in Part H).

Node-classification runs are instrumented on the trainer instance (no src change):
  draw_s    : wall time of Trainer.training_draw (CPU build of the training draw + move to GPU)
  train_s   : wall time of Trainer._train_epoch (forward + backward + step; epoch 0 also runs the
              logged operator diagnostics), with cuda synchronisation
  epoch_s   : wall time of one whole epoch of Trainer.fit (draw, train, val + test eval, logging,
              checkpoints), from consecutive epoch callbacks
  peak_train_mb : torch.cuda.max_memory_allocated within _train_epoch (reset per epoch)
Timed epochs exclude epoch 0 (warm-up and diagnostics)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.analysis.wisp_readouts import full_schema_keys  # noqa: E402
from src.builders import build_experiment_components  # noqa: E402
from src.configs.config import TrainConfig  # noqa: E402
from src.utils.misc import seed_everything  # noqa: E402

OUT = ROOT.parent / "docs" / "WISP_evidence" / "h6_smoke_cost.json"

# name: (dataset, data overrides, model overrides, ablation, epochs)
RUNS = {
    "acm_schemas_cost": ("acm", {"path_source": "schemas"}, {}, {}, 4),
    "acm_full_cost": ("acm", {"path_source": "wisp"}, {}, {}, 4),
    "acm_endpoints": ("acm", {"path_source": "wisp"}, {}, {"path_weight_input": "endpoints"}, 2),
    "acm_schema": ("acm", {"path_source": "wisp"}, {}, {"path_weight_input": "schema"}, 2),
    "acm_uniform": ("acm", {"path_source": "wisp"}, {}, {"path_weight_input": "uniform"}, 2),
    "acm_full_adaptive": ("acm", {"path_source": "wisp", "wisp_proposal": "adaptive"}, {}, {}, 2),
    "dblp_full": ("dblp", {"path_source": "wisp"}, {}, {}, 2),
    "imdb_full": ("imdb", {"path_source": "wisp"}, {}, {}, 2),
    "amazonbook_lp_exp": ("amazonbook", {}, {"path_weight": "exp"}, {}, 1),
    "mutag_gc_exp": ("mutag", {}, {"path_weight": "exp"}, {}, 1),
    # correction (independent review MAJOR-1): the pre-registered r rule (h6_calibrate_r.json),
    # 4 epochs so that 3 epochs are timed (MINOR-12)
    "dblp_full_r65": ("dblp", {"path_source": "wisp", "wisp_walks_per_anchor": 65}, {}, {}, 4),
    "imdb_full_r12": ("imdb", {"path_source": "wisp", "wisp_walks_per_anchor": 12}, {}, {}, 4),
    # control for the amazonbook OOM: the pre-Part-H behaviour on the same data and GPU
    "amazonbook_lp_sigmoid_control": ("amazonbook", {}, {"path_weight": "sigmoid_gamma"}, {}, 1),
    # the LP smoke on the CPU (the 8 GB GPU runs out of memory; see the report)
    "amazonbook_lp_exp_cpu": ("amazonbook", {}, {"path_weight": "exp"}, {}, 1),
}
DEVICE = {"amazonbook_lp_exp_cpu": "cpu"}


def make_config(name):
    dataset, data_extra, model_extra, ablation, epochs = RUNS[name]
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml")
                         .read_text(encoding="utf-8"))
    raw["data"] = {**raw["data"], **data_extra}
    raw["model"] = {**raw["model"], **model_extra}
    raw.update(epochs=epochs, seed=42, device=DEVICE.get(name, "cuda"), ablation=ablation,
               log_every=1000,
               exp_name=f"h6_{name}", output_dir="experiments/wisp_h6")
    return TrainConfig.from_dict(raw)


def draw_stats(groups, node_type, self_inverse):
    """Cells, incidences, distinct canonical full-walk signatures, length histogram (eval draw)."""
    groups = [g for g in groups if g.num_faces]
    keys = set()
    for g in groups:
        keys.update(full_schema_keys(g, node_type, self_inverse))
    by_length = {}
    for g in groups:
        row = by_length.setdefault(int(g.length), {"cells": 0, "incidences": 0, "mass": 0.0})
        row["cells"] += g.num_faces
        row["incidences"] += g.num_faces * (int(g.length) + 1)
        if g.path_weight is not None:
            row["mass"] += float(g.path_weight.double().sum())
    return {"cells": sum(r["cells"] for r in by_length.values()),
            "incidences": sum(r["incidences"] for r in by_length.values()),
            "distinct_signatures": len(keys), "groups": len(groups),
            "by_length": dict(sorted(by_length.items()))}


def instrument(trainer, record):
    draw, train = trainer.training_draw, trainer._train_epoch

    def timed_draw(data, prepared, epoch):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = draw(data, prepared, epoch)
        torch.cuda.synchronize()
        record.setdefault("draw_s", []).append(time.perf_counter() - t)
        return out

    def timed_train(data, epoch):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        out = train(data, epoch)
        torch.cuda.synchronize()
        record.setdefault("train_s", []).append(time.perf_counter() - t)
        record.setdefault("peak_train_mb", []).append(torch.cuda.max_memory_allocated() / 2**20)
        return out

    trainer.training_draw, trainer._train_epoch = timed_draw, timed_train


def run(name):
    cfg = make_config(name)
    if (cfg.exp_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{cfg.exp_dir} already has metrics.jsonl (the logger appends).")
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, cfg.device)
    start = time.perf_counter()
    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    setup_s = time.perf_counter() - start
    record: dict = {"task": cfg.task_type, "epochs": cfg.epochs, "setup_s": setup_s}
    callback = None
    if cfg.task_type == "node_classification":
        record["eval_draw"] = draw_stats(data.face_groups, data.node_type,
                                         data.fact_layer.self_inverse)
        instrument(trainer, record)
        stamps = []

        def callback(epoch, *_):
            torch.cuda.synchronize()
            stamps.append(time.perf_counter())
    torch.cuda.reset_peak_memory_stats()
    fit_start = time.perf_counter()
    try:
        if callback is None:
            trainer.fit(data)
        else:
            trainer.fit(data, epoch_callback=callback)
    finally:
        logger.close()
    record["fit_s"] = time.perf_counter() - fit_start
    record["peak_fit_mb"] = torch.cuda.max_memory_allocated() / 2**20
    if callback is not None:
        edges = [fit_start] + stamps
        record["epoch_s"] = [b - a for a, b in zip(edges, edges[1:])]
        timed = range(1, len(record["epoch_s"]))
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        record["timed"] = {
            "epochs": list(timed),
            "draw_s": mean([record["draw_s"][e] for e in timed]),
            "train_s": mean([record["train_s"][e] for e in timed]),
            "epoch_s": mean([record["epoch_s"][e] for e in timed]),
            "peak_train_mb": max(record["peak_train_mb"][e] for e in timed),
        }
        t = record["timed"]
        t["sampling_share_of_epoch"] = t["draw_s"] / t["epoch_s"]
        t["sampling_share_of_draw_plus_train"] = t["draw_s"] / (t["draw_s"] + t["train_s"])
    rows = [json.loads(line) for line in
            (cfg.exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    epoch_rows = [r for r in rows if "train" in r]
    record["train_loss"] = [r["train"]["loss"] for r in epoch_rows]
    record["finite_losses"] = all(math.isfinite(x) for x in record["train_loss"])
    record["path_log_w_max"] = [r["train"].get("path_log_w_max") for r in epoch_rows]
    record["path_log_w_mean"] = [r["train"].get("path_log_w_mean") for r in epoch_rows]
    logged = [x for x in record["path_log_w_max"] if x is not None]
    record["max_log_weight"] = max(logged) if logged else None
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=list(RUNS))
    args = parser.parse_args()
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in args.only:
        record = run(name)
        results[name] = record
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
        summary = {k: record.get(k) for k in ("timed", "max_log_weight", "finite_losses")}
        print("H6", name, json.dumps(summary), flush=True)
        if record["max_log_weight"] is not None and record["max_log_weight"] > 30:
            raise SystemExit(f"STOP: max log-weight {record['max_log_weight']} > 30 in {name}")


if __name__ == "__main__":
    main()
