"""G0b gap localization ladder driver (see g0b_preregistration.md). Run from H2S-NN/ (h2snn env):

  python ../docs/D12_evidence/g0b_ladder.py run        # L0, L1 (in-process), L2-L4 (trainer.py subprocesses); resumable
  python ../docs/D12_evidence/g0b_ladder.py evaltrain  # eval-mode train/val/test macro-F1 of each trainer run's best.pt
  python ../docs/D12_evidence/g0b_ladder.py summarize  # table + reading rules -> g0b_summary.md
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
EVIDENCE = ROOT.parent / "docs" / "D12_evidence"
RUNS = EVIDENCE / "g0b_runs"
RUN_ROOT = "experiments/d12_g0b"
CFG_DIR = ROOT / "experiments" / "d12_g0b_configs"
PYTHON = sys.executable
DATASETS = ("acm", "dblp")
L0_SEEDS = (42, 43, 44, 45, 46)
SEEDS = (42, 43, 44)
TRAINER_RUNGS = {
    "L2raw": ({"h0_only_readout": True}, {}),
    "L2row": ({"h0_only_readout": True}, {"feature_normalization": "row"}),
    "L3": ({"identity_maps": True, "no_path_cells": True, "coupling": "symmetric"}, {}),
    "L4": ({}, {}),
}
BANDS = {"L1": {"acm": (92.25, 0.82), "dblp": (90.87, 1.77)}}  # HetSheaf Table 2, GCN macro-F1


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def run_gnn(rung: str, model_name: str, dataset: str, seed: int) -> None:
    out = RUNS / f"{rung}_{dataset}_s{seed}.json"
    if out.exists():
        return
    sys.path.insert(0, str(EVIDENCE))
    import torch
    import g0_rgcn

    start = time.time()
    summary = g0_rgcn.run(dataset, "O", seed, torch.device("cuda"), model_name=model_name,
                          prefix=f"g0b_runs/{rung}_curve")
    best = summary["test_at_best_val_macro_f1"]
    out.write_text(json.dumps({"rung": rung, "dataset": dataset, "seed": seed, "best_epoch": best["epoch"],
                               "train_macro_f1": best["train_macro_f1"], "val_macro_f1": best["val_macro_f1"],
                               "test_macro_f1": best["macro_f1"], "epochs_run": summary["epochs_run"],
                               "minutes": (time.time() - start) / 60}, indent=1), encoding="utf-8")
    _log(f"{rung} {dataset} s{seed}: test {100 * best['macro_f1']:.2f}")


def exp_name(rung: str, dataset: str, seed: int) -> str:
    return f"g0b_{rung}_{dataset}_s{seed}"


def run_trainer(rung: str, dataset: str, seed: int) -> None:
    run_dir = ROOT / RUN_ROOT / exp_name(rung, dataset, seed)
    if (run_dir / "best_test_metrics.json").exists():
        return
    ablation, data_extra = TRAINER_RUNGS[rung]
    base = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml").read_text(encoding="utf-8"))
    base.update(seed=seed, exp_name=exp_name(rung, dataset, seed), output_dir=RUN_ROOT, device="cuda",
                ablation=dict(ablation))
    base["data"] = {**base["data"], **data_extra}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CFG_DIR / f"{exp_name(rung, dataset, seed)}.yaml"
    cfg.write_text(yaml.safe_dump(base), encoding="utf-8")
    for attempt in range(1, 4):
        start = time.time()
        with open(EVIDENCE / "g0b_logs" / f"{exp_name(rung, dataset, seed)}.txt", "w", encoding="utf-8") as log:
            rc = subprocess.call([PYTHON, "-W", "ignore", "scripts/train_h2snn.py", "--config", str(cfg),
                                  "--device", "cuda"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        _log(f"{rung} {dataset} s{seed}: rc={rc} in {(time.time() - start) / 60:.1f} min (attempt {attempt})")
        if rc == 0:
            return
        time.sleep(60)


def cmd_run(_: argparse.Namespace) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "g0b_logs").mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for seed in L0_SEEDS:
            run_gnn("L0", "rgcn", dataset, seed)
        for seed in SEEDS:
            run_gnn("L1", "gcn", dataset, seed)
    for rung in TRAINER_RUNGS:
        for dataset in DATASETS:
            for seed in SEEDS:
                run_trainer(rung, dataset, seed)
    _log("run done")


def cmd_evaltrain(_: argparse.Namespace) -> None:
    """Eval-mode macro-F1 of the selected (best-val) checkpoint on train/val/test; val/test must equal
    best_test_metrics.json (consistency check)."""
    import torch

    sys.path.insert(0, str(ROOT))
    from src.configs.config import TrainConfig
    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model
    from src.utils.misc import compute_classification_metrics

    for rung in TRAINER_RUNGS:
        for dataset in DATASETS:
            for seed in SEEDS:
                out = RUNS / f"{rung}_{dataset}_s{seed}.json"
                run_dir = ROOT / RUN_ROOT / exp_name(rung, dataset, seed)
                if out.exists() or not (run_dir / "best_test_metrics.json").exists():
                    continue
                cfg = TrainConfig.from_dict(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
                data = build_dataset(cfg).to("cuda")
                torch.manual_seed(0)
                model = build_model(cfg, data).to("cuda")
                state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cuda", weights_only=False)
                model.load_state_dict(state["state_dict"])  # src/utils/checkpointer.py:57
                model.eval()
                with torch.no_grad():
                    logits = model(data).logits
                f1 = {split: compute_classification_metrics(logits[mask], data.target_y[mask])["macro_f1"]
                      for split, mask in (("train", data.target_train_mask), ("val", data.target_val_mask),
                                          ("test", data.target_test_mask))}
                best = json.loads((run_dir / "best_test_metrics.json").read_text(encoding="utf-8"))
                rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
                record = {"rung": rung, "dataset": dataset, "seed": seed, "best_epoch": best["best_epoch"],
                          "train_macro_f1": f1["train"], "val_macro_f1": best["val_macro_f1"],
                          "test_macro_f1": best["test_macro_f1"], "epochs_run": len(rows),
                          "train_macro_f1_trainmode_logged": rows[best["best_epoch"]]["train"]["macro_f1"],
                          "consistency_val_abs_diff": abs(f1["val"] - best["val_macro_f1"]),
                          "consistency_test_abs_diff": abs(f1["test"] - best["test_macro_f1"])}
                out.write_text(json.dumps(record, indent=1), encoding="utf-8")
                _log(f"evaltrain {rung} {dataset} s{seed}: train {100 * f1['train']:.2f} "
                     f"val/test diff {record['consistency_val_abs_diff']:.2e}/{record['consistency_test_abs_diff']:.2e}")
                del model, data
                torch.cuda.empty_cache()


def _stats(rung: str, dataset: str, key: str) -> tuple[float, float, int]:
    values = [100 * json.loads(p.read_text(encoding="utf-8"))[key] for p in sorted(RUNS.glob(f"{rung}_{dataset}_s*.json"))]
    if not values:
        return float("nan"), float("nan"), 0
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else float("nan")), len(values)


def cmd_summarize(_: argparse.Namespace) -> None:
    rungs = ["L0", "L1", "L2raw", "L2row", "L3", "L4"]
    lines = ["| rung | dataset | n | test macro-F1 | val macro-F1 | train macro-F1 (eval mode) | best epoch |",
             "|---|---|---|---|---|---|---|"]
    m: dict[tuple[str, str], dict] = {}
    for rung in rungs:
        for dataset in DATASETS:
            t, ts, n = _stats(rung, dataset, "test_macro_f1")
            v, _, _ = _stats(rung, dataset, "val_macro_f1")
            tr, _, _ = _stats(rung, dataset, "train_macro_f1")
            epochs = [json.loads(p.read_text(encoding="utf-8"))["best_epoch"] for p in RUNS.glob(f"{rung}_{dataset}_s*.json")]
            m[(rung, dataset)] = {"test": t, "std": ts, "val": v, "train": tr, "n": n}
            lines.append(f"| {rung} | {dataset} | {n} | {t:.2f} ± {ts:.2f} | {v:.2f} | {tr:.2f} | "
                         f"{statistics.fmean(epochs) if epochs else float('nan'):.0f} |")
    rules = []
    for dataset in DATASETS:
        g = lambda r, k="test": m[(r, dataset)][k]  # noqa: E731
        lo = BANDS["L1"][dataset][0] - 2 * BANDS["L1"][dataset][1]
        r1 = g("L2row") > g("L2raw") + 2
        r2 = abs(g("L3") - g("L0")) <= 2 and g("L4") < g("L3") - 2
        r3 = g("L3") < g("L0") - 2
        label = ""
        if r3:
            if g("L3", "train") < 90:
                label = "underfitting (train < 90)"
            elif g("L3", "train") >= 98 and g("L3", "val") < g("L0", "val") - 2:
                label = "overfitting (train >= 98, val more than 2 below L0's)"
            else:
                label = "unlabelled (train in [90, 98) or val within 2 of L0)"
        r4 = g("L1") < lo
        rules += [f"{dataset}: (i) L2row > L2raw + 2: {r1} ({g('L2row'):.2f} vs {g('L2raw'):.2f})",
                  f"{dataset}: (ii) |L3 - L0| <= 2 and L4 < L3 - 2: {r2} (L0 {g('L0'):.2f}, L3 {g('L3'):.2f}, L4 {g('L4'):.2f})",
                  f"{dataset}: (iii) L3 < L0 - 2: {r3} {label} (L3 train {g('L3', 'train'):.2f}, val {g('L3', 'val'):.2f}; L0 val {g('L0', 'val'):.2f})",
                  f"{dataset}: (iv) L1 < band lower edge {lo:.2f}: {r4} (L1 {g('L1'):.2f})"]
    text = "\n".join(["## G0b ladder (test macro-F1 at best val macro-F1; mean ± sample std over seeds)", *lines,
                      "", "## Reading rules (g0b_preregistration.md)", *rules])
    (EVIDENCE / "g0b_summary.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "evaltrain", "summarize"])
    args = parser.parse_args()
    {"run": cmd_run, "evaltrain": cmd_evaltrain, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    main()
