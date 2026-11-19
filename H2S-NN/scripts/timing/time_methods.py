"""Card 7 V7 (ablation plan v2 Q5; spec 1.27 C5, §6 "Timing and analysis-only settings"): timing of one main-table
method on one dataset (ogbn-mag: 8,192 training targets per epoch; IMDB). No metric is read or written.

Per (method, dataset): time per epoch = the median of epochs 2-4 after one warm-up epoch (`--epochs 4`); peak GPU
memory (`torch.cuda.max_memory_allocated` over the timed run); parameter count. Configuration: the method's tuned
one (H2S-NN: `--h2snn_config`, the main table's tuned file; a baseline: the best trial of its seed-0 sweep under
`--baseline_output`, as the final-seed launcher selects it), else the card 2 central point (flagged
`configuration: central`).

Code reused (card 7 V7):
- H2S-NN, full batch (IMDB): `scripts/v3b_cost_smoke.py` `run` (epoch = the WISP training draw + the training step);
- H2S-NN, mini-batch (ogbn-mag): the trainer itself (`build_experiment_components`, `trainer.fit`), each
  `train_epoch` call timed (the pattern of the card 2 dry run, `docs/dryrun_evidence/d2_dryrun.py` `wrap`);
- the central points: card 2 D6 (`docs/dryrun_evidence/d6_dryrun.py`: `central_point` for the baselines,
  `patched_h2snn_raw` for H2S-NN);
- baselines: the D6 fit path (`campaign.build_raw_base` + `fit_cell` with the per-epoch `report` callback); an
  epoch's training time runs from the previous epoch report to the first `.eval()` call after it (validation
  excluded), its full time from report to report.

    python scripts/timing/time_methods.py --method h2snn --dataset imdb --epochs 4 --out <json> \
        [--h2snn_config <tuned yaml>] [--baseline_output experiments/baselines] [--device cuda]
    python scripts/timing/time_methods.py --list --dataset ogbn-mag
Run from `H2S-NN/` (the card 2 dry-run modules resolve the repository from the working directory).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
DRYRUN = PROJECT_ROOT.parent / "docs" / "dryrun_evidence"

TASK = "node_classification"
DATASETS: tuple[str, ...] = ("ogbn-mag", "imdb")
WARMUP_EPOCHS = 1
TIMED_EPOCHS: tuple[int, ...] = (2, 3, 4)          # 1-based: the median of epochs 2-4
FIELDS: tuple[str, ...] = ("method", "dataset", "configuration", "flagged", "point", "timing_path",
                           "epoch_definition", "epochs_requested", "epochs_run", "per_epoch",
                           "time_per_epoch_seconds", "peak_gpu_memory_mb", "parameters", "gpu", "host", "job",
                           "head", "device")


def dryrun_module():
    """The card 2 D6 module (imports `d2_dryrun`; both resolve the repository from the working directory)."""
    if str(DRYRUN) not in sys.path:
        sys.path.insert(0, str(DRYRUN))
    import d6_dryrun

    return d6_dryrun


def main_table_methods(dataset: str) -> list[str]:
    """H2S-NN and every baseline the registry runs on this node-classification dataset."""
    from src.baselines.registry import models_for_task, unsupported_reason

    return ["h2snn"] + [spec.name for spec in models_for_task(TASK) if not unsupported_reason(spec.name, TASK, dataset)]


def time_per_epoch(per_epoch: list[dict[str, Any]], key: str = "train_seconds") -> float | None:
    """The median of epochs 2-4 (1-based) after the warm-up; None when fewer epochs ran."""
    if len(per_epoch) < max(TIMED_EPOCHS):
        return None
    return float(statistics.median(per_epoch[e - 1][key] for e in TIMED_EPOCHS))


def finish(record: dict[str, Any]) -> dict[str, Any]:
    record["epochs_run"] = len(record["per_epoch"])
    record["time_per_epoch_seconds"] = time_per_epoch(record["per_epoch"])
    missing = [f for f in FIELDS if f not in record]
    if missing:
        raise KeyError(f"timing record lacks {missing}")
    return {f: record[f] for f in FIELDS} | {k: v for k, v in record.items() if k not in FIELDS}


def environment(device: str) -> dict[str, Any]:
    import torch

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    return {"gpu": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "cpu",
            "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", ""), "head": head, "device": device}


# ------------------------------------------------------------------------------------------------ H2S-NN
def h2snn_raw(dataset: str, config: Path | None, out: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    import yaml

    if config is not None:
        return yaml.safe_load(config.read_text(encoding="utf-8")), "tuned", {"config": str(config)}
    d6 = dryrun_module()
    d6.ARGS["point"], d6.ARGS["set"] = "central", {}
    raw, point = d6.patched_h2snn_raw(dataset, TASK, out)
    return raw, "central", point


def time_h2snn(dataset: str, config: Path | None, epochs: int, device: str, out: Path) -> dict[str, Any]:
    import torch

    from src.configs.config import TrainConfig

    raw, source, point = h2snn_raw(dataset, config, out)
    raw.update(epochs=epochs, device=device, use_wandb=False, output_dir=str(out / "runs"),
               exp_name=f"timing_h2snn_{dataset}")
    for key in ("wandb_project", "wandb_group", "wandb_tags"):
        raw.pop(key, None)
    cfg = TrainConfig.from_dict(raw)
    record: dict[str, Any] = {"method": "h2snn", "dataset": dataset, "configuration": source,
                              "flagged": source == "central", "point": point, "epochs_requested": epochs,
                              **environment(device)}
    if cfg.data.training_mode != "minibatch":
        import v3b_cost_smoke

        smoke = v3b_cost_smoke.run(cfg, epochs)
        record.update(timing_path="scripts/v3b_cost_smoke.py run", epoch_definition="WISP training draw + "
                      "one full-batch training step (forward, backward, optimizer step); validation excluded",
                      per_epoch=[{"epoch": e["epoch"] + 1, "train_seconds": e["epoch_seconds"],
                                  "epoch_seconds": e["epoch_seconds"]} for e in smoke["epochs"]],
                      peak_gpu_memory_mb=max((e["peak_memory_mb"] for e in smoke["epochs"]), default=None),
                      parameters=int(smoke["num_parameters"]))
        return finish(record)
    from src.builders import build_experiment_components

    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    times: list[float] = []
    original = trainer.train_epoch

    def timed(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    trainer.train_epoch = timed
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        trainer.fit(data)
    finally:
        logger.close()
    record.update(timing_path="trainer.fit, train_epoch timed (card 2 dry-run pattern)",
                  epoch_definition=f"one training epoch of the trainer ({cfg.data.epoch_targets} training targets in "
                                   f"mini-batches of {cfg.data.batch_size}); validation excluded",
                  per_epoch=[{"epoch": i + 1, "train_seconds": t, "epoch_seconds": t} for i, t in enumerate(times)],
                  peak_gpu_memory_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                                      if torch.cuda.is_available() else None),
                  parameters=int(sum(p.numel() for p in model.parameters())))
    return finish(record)


# ------------------------------------------------------------------------------------------------ baselines
def baseline_params(model: str, dataset: str, baseline_output: Path | None) -> tuple[dict[str, Any], str, dict]:
    """(params, configuration, point rule): the best trial of the seed-0 sweep, else the D6 central point."""
    trials = None if baseline_output is None else baseline_output / f"{model}_{TASK}_{dataset}" / "tuning_trials_run0.json"
    if trials is not None and trials.exists():
        from run_baseline_tuning import select_best_trial
        from src.baselines.campaign import monitor_for

        best = select_best_trial(json.loads(trials.read_text(encoding="utf-8")), monitor_for(TASK))
        return dict(best["sweep_params"]), "tuned", {"selected_from": str(trials), "trial": best.get("trial_name")}
    from src.baselines.registry import get_adapter

    d6 = dryrun_module()
    adapter = get_adapter(model)
    published = adapter.published_params(TASK, dataset)
    default = dict(getattr(published, "values", {}) or {}) if published is not None else {}
    values, rule = d6.central_point(adapter.search_space(TASK, dataset), default)
    return values, "central", {"rule": rule}


def time_baseline(model: str, dataset: str, epochs: int, device: str, baseline_output: Path | None
                  ) -> dict[str, Any]:
    import torch

    from src.baselines.campaign import Cell, build_raw_base, fit_cell
    from src.utils.wandb_campaign import CampaignUnit

    params, source, rule = baseline_params(model, dataset, baseline_output)
    record: dict[str, Any] = {"method": model, "dataset": dataset, "configuration": source,
                              "flagged": source == "central", "point": {"values": params, **rule},
                              "epochs_requested": epochs, **environment(device)}
    unit = CampaignUnit(run_index=0, num_runs=1, seed=0, label="run")
    raw_base = build_raw_base(Cell(model, TASK, dataset), unit, device=device, epochs=epochs)
    evals: list[float] = []
    reports: list[float] = []
    counted: list[int] = []
    original_eval = torch.nn.Module.eval
    original_optimizer_init = torch.optim.Optimizer.__init__

    def marked_eval(self):
        evals.append(time.perf_counter())
        return original_eval(self)

    def counting_init(self, params_, *args, **kwargs):   # the first optimizer's trainable parameters
        original_optimizer_init(self, params_, *args, **kwargs)
        if not counted:
            seen = {id(p): p for group in self.param_groups for p in group["params"]}
            counted.append(int(sum(p.numel() for p in seen.values())))

    def report(*, epoch, monitor, running_best, metrics):   # a timestamp only; metric values are never read
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reports.append(time.perf_counter())

    torch.nn.Module.eval = marked_eval
    torch.optim.Optimizer.__init__ = counting_init
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        fit_cell(raw_base, params, source="tuned", device=torch.device(device), report=report)
    finally:
        torch.nn.Module.eval = original_eval
        torch.optim.Optimizer.__init__ = original_optimizer_init
    per_epoch = []
    for k, end in enumerate(reports):
        begin = reports[k - 1] if k else start
        first_eval = next((t for t in evals if begin < t <= end), end)
        per_epoch.append({"epoch": k + 1, "train_seconds": first_eval - begin, "epoch_seconds": end - begin})
    record.update(timing_path="card 2 D6 fit path (campaign.fit_cell with the per-epoch report)",
                  epoch_definition="from the previous epoch report to the first .eval() call (training; the first "
                                   "epoch includes the bundle build and is the warm-up)",
                  per_epoch=per_epoch,
                  peak_gpu_memory_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                                      if torch.cuda.is_available() else None),
                  parameters=counted[0] if counted else None)
    return finish(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--method", default=None)
    parser.add_argument("--list", action="store_true", help="print the main-table methods of the dataset")
    parser.add_argument("--epochs", type=int, default=WARMUP_EPOCHS + len(TIMED_EPOCHS))
    parser.add_argument("--h2snn_config", type=Path, default=None)
    parser.add_argument("--baseline_output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.list:
        print("\n".join(main_table_methods(args.dataset)))
        return
    if args.method is None or args.out is None:
        raise SystemExit("--method and --out are required")
    work = args.out.parent / f"{args.out.stem}_work"
    work.mkdir(parents=True, exist_ok=True)
    if args.method == "h2snn":
        record = time_h2snn(args.dataset, args.h2snn_config, args.epochs, args.device, work)
    else:
        record = time_baseline(args.method, args.dataset, args.epochs, args.device, args.baseline_output)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("method", "dataset", "configuration", "epochs_run",
                                             "time_per_epoch_seconds", "peak_gpu_memory_mb", "parameters")}))


if __name__ == "__main__":
    main()
