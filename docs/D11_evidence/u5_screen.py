"""U5 core-claim screen driver (plan §12.6). Run from H2S-NN/ with the h2snn environment.

  python ../docs/D11_evidence/u5_screen.py run [--workers 2] [--datasets acm dblp imdb]
         [--variants full gamma0 endpoint_edges h0_only pair_attention] [--seeds 42 43 44]
  python ../docs/D11_evidence/u5_screen.py count     # parameter counts per (dataset, variant)
  python ../docs/D11_evidence/u5_screen.py summarize # tables, trigger evaluation (pre-registered rules)

One fixed config per dataset (configs/experiments/<dataset>_h2snn.yaml); a row differs only by its
`ablation` entry; nothing is tuned. Runs are resumable (a run with best_test_metrics.json is skipped).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path.cwd()
EVIDENCE = ROOT.parent / "docs" / "D11_evidence"
SCREEN_DIR = EVIDENCE / "screen"
RUN_ROOT = "experiments/d11_screen"
CFG_DIR = ROOT / "experiments" / "d11_screen_configs"
PYTHON = sys.executable
DATASETS = ("acm", "dblp", "imdb")
VARIANTS = {
    "full": {},
    "gamma0": {"no_path_cells": True},
    "endpoint_edges": {"endpoint_edges": True},
    "h0_only": {"h0_only_readout": True},
    "pair_attention": {"coupling": "pair_attention"},
}
SEEDS = (42, 43, 44)
GATE_INIT = 0.6


def exp_name(dataset: str, variant: str, seed: int) -> str:
    return f"d11_{dataset}_{variant}_s{seed}"


def make_config(dataset: str, variant: str, seed: int) -> Path:
    base = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml").read_text(
        encoding="utf-8"))
    base["seed"] = seed
    base["exp_name"] = exp_name(dataset, variant, seed)
    base["output_dir"] = RUN_ROOT
    base["device"] = "cuda"
    base["ablation"] = dict(VARIANTS[variant])
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{exp_name(dataset, variant, seed)}.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def run_dir(dataset: str, variant: str, seed: int) -> Path:
    return ROOT / RUN_ROOT / exp_name(dataset, variant, seed)


def cmd_run(args: argparse.Namespace) -> None:
    jobs = [(d, v, s) for d in args.datasets for s in args.seeds for v in args.variants
            if not (run_dir(d, v, s) / "best_test_metrics.json").exists()]
    print(f"{len(jobs)} runs to do", flush=True)
    running: list[tuple[subprocess.Popen, tuple, float]] = []
    while jobs or running:
        while jobs and len(running) < args.workers:
            job = jobs.pop(0)
            cfg = make_config(*job)
            log = open(EVIDENCE / "screen_logs" / f"{exp_name(*job)}.txt", "w", encoding="utf-8") \
                if (EVIDENCE / "screen_logs").exists() else subprocess.DEVNULL
            proc = subprocess.Popen([PYTHON, "-W", "ignore", "scripts/train_h2snn.py", "--config",
                                     str(cfg), "--device", "cuda"], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT)
            running.append((proc, job, time.time()))
            print(f"started {job}", flush=True)
        for entry in list(running):
            proc, job, started = entry
            if proc.poll() is not None:
                running.remove(entry)
                print(f"finished {job} rc={proc.returncode} in {(time.time() - started) / 60:.1f} min",
                      flush=True)
        time.sleep(5)


def _read_run(dataset: str, variant: str, seed: int):
    directory = run_dir(dataset, variant, seed)
    metrics_file = directory / "best_test_metrics.json"
    if not metrics_file.exists():
        return None
    best = json.loads(metrics_file.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    return best, rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def cmd_count(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(ROOT))
    from src.configs.config import TrainConfig
    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model

    counts = {}
    for dataset in DATASETS:
        base = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml").read_text(
            encoding="utf-8"))
        data = build_dataset(TrainConfig.from_dict({**base, "device": "cuda"})).to("cuda")
        for variant, ablation in VARIANTS.items():
            cfg = TrainConfig.from_dict({**base, "device": "cuda", "ablation": ablation})
            torch.manual_seed(0)
            model = build_model(cfg, data).to("cuda").train()
            total = sum(p.numel() for p in model.parameters())
            loss = F.cross_entropy(model(data).logits[data.target_train_mask],
                                   data.target_y[data.target_train_mask]) \
                if not data.is_multilabel else F.binary_cross_entropy_with_logits(
                    model(data).logits[data.target_train_mask],
                    data.target_y[data.target_train_mask].float())
            loss.backward()
            used = sum(p.numel() for p in model.parameters() if p.grad is not None)
            counts[f"{dataset}/{variant}"] = {"total": total, "receiving_gradient": used}
            print(dataset, variant, counts[f"{dataset}/{variant}"], flush=True)
            del model
            torch.cuda.empty_cache()
    (SCREEN_DIR / "param_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")


def cmd_summarize(args: argparse.Namespace) -> None:
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    table: dict[tuple[str, str], dict] = {}
    lines = ["| dataset | row | n | test macro-F1 | test micro-F1 | val macro-F1 | best epoch |",
             "|---|---|---|---|---|---|---|"]
    gate_move = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            runs = [r for s in SEEDS if (r := _read_run(dataset, variant, s)) is not None]
            if not runs:
                continue
            macro = [b["test_macro_f1"] for b, _ in runs]
            micro = [b["test_micro_f1"] for b, _ in runs]
            val = [b["val_macro_f1"] for b, _ in runs]
            epochs = [b["best_epoch"] for b, _ in runs]
            (m, sd), (mi, sdi), (vm, _) = _mean_std(macro), _mean_std(micro), _mean_std(val)
            table[(dataset, variant)] = {"mean": m, "std": sd, "n": len(runs), "values": macro,
                                         "micro": mi}
            lines.append(f"| {dataset} | {variant} | {len(runs)} | {100 * m:.2f} ± {100 * sd:.2f} | "
                         f"{100 * mi:.2f} ± {100 * sdi:.2f} | {100 * vm:.2f} | "
                         f"{statistics.fmean(epochs):.0f} |")
            if variant == "full":
                for _, rows in runs:
                    g = [r["train"]["g_mean"] for r in rows if "g_mean" in r["train"]]
                    b = [r["train"]["b_mean"] for r in rows if "b_mean" in r["train"]]
                    gate_move.append((max(abs(x - GATE_INIT) for x in g), max(abs(x - GATE_INIT) for x in b)))
    text = "\n".join(lines)

    def relation(a: str, b: str, dataset: str) -> str:
        x, y = table.get((dataset, a)), table.get((dataset, b))
        if x is None or y is None or x["n"] < 2 or y["n"] < 2:
            return "n/a"
        delta, s = x["mean"] - y["mean"], max(x["std"], y["std"])
        return "beats" if delta > s else ("loses" if delta < -s else "~")

    pair_lines = ["| dataset | full vs gamma0 | full vs endpoint_edges | full vs h0_only | pair_attention vs full |",
                  "|---|---|---|---|---|"]
    rel = {}
    for dataset in DATASETS:
        rel[dataset] = {"g0": relation("full", "gamma0", dataset),
                        "ee": relation("full", "endpoint_edges", dataset),
                        "h0": relation("full", "h0_only", dataset),
                        "pa": relation("pair_attention", "full", dataset)}
        pair_lines.append(f"| {dataset} | {rel[dataset]['g0']} | {rel[dataset]['ee']} | "
                          f"{rel[dataset]['h0']} | {rel[dataset]['pa']} |")
    n_ds = len(DATASETS)
    t1 = sum(1 for d in DATASETS if rel[d]["g0"] == "beats" and rel[d]["ee"] == "beats") < 2
    t2 = sum(1 for d in DATASETS if rel[d]["h0"] == "~") >= 2
    t3 = (not t1) and sum(1 for d in DATASETS if rel[d]["h0"] == "beats") >= 2
    t5 = sum(1 for d in DATASETS if rel[d]["pa"] == "beats") >= 2
    t4 = bool(gate_move) and statistics.fmean([max(g, b) for g, b in gate_move]) < 0.05
    triggers = [
        f"T1 path cells do not beat gamma0 and endpoint_edges (on >= 2 of {n_ds}): {'HIT -> STOP' if t1 else 'not hit'}",
        f"T2 full ~ h0_only on >= 2 datasets: {'HIT -> expressivity work' if t2 else 'not hit'}",
        f"T3 path cells win and full beats h0_only on >= 2 datasets: {'HIT -> Tier 0, 10 seeds' if t3 else 'not hit'}",
        f"T4 gates do not move (mean max |gate mean - 0.6| = "
        f"{statistics.fmean([max(g, b) for g, b in gate_move]) if gate_move else float('nan'):.3f}): "
        f"{'HIT -> gates are fixed hyperparameters' if t4 else 'not hit'}",
        f"T5 pair_attention beats full on >= 2 datasets: {'HIT -> STOP (A4 in doubt)' if t5 else 'not hit'}",
        f"T6 otherwise A4 retained: {'yes' if not t5 else 'no'}",
    ]
    out = "\n".join(["## Test macro-F1 (mean ± sample std over seeds)", text, "",
                    "## Pairwise relations (a beats b iff mean diff > max std)", *pair_lines, "",
                    "## Triggers (plan 12.6, rules fixed in pre_registration.md)", *triggers])
    (SCREEN_DIR / "summary.md").write_text(out, encoding="utf-8")
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "count", "summarize"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    (EVIDENCE / "screen_logs").mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "count": cmd_count, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    main()
