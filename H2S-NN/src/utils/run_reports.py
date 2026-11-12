from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.configs.config import TrainConfig
from src.utils.misc import dataclass_to_dict


def default_monitor_for_task(task_type: str) -> str:
    if task_type == "node_classification":
        return "macro_f1"
    if task_type == "graph_classification":
        # Macro-F1, matching the reported metric. See the graph-classification
        # protocol note in src/data/graph_classification_dataset.py.
        return "macro_f1"
    if task_type == "link_prediction":
        return "auroc"
    raise ValueError(f"Unsupported task_type '{task_type}'.")


def report_metrics_for_task(task_type: str) -> tuple[str, ...]:
    if task_type == "node_classification":
        return ("test_macro_f1", "test_micro_f1", "test_acc")
    if task_type == "link_prediction":
        # HGB's two test sets (spec v1.16 K1)
        return ("test_2hop_auroc", "test_2hop_mrr", "test_random_auroc", "test_random_mrr")
    if task_type == "graph_classification":
        # Macro-F1 first: it is the headline metric for this task.
        return ("test_macro_f1", "test_micro_f1", "test_auroc", "test_acc")
    raise ValueError(f"Unsupported task_type '{task_type}'.")


def validation_metric_name(metric: str) -> str:
    return metric if metric.startswith("val_") else f"val_{metric}"


def metric_score(record: dict[str, Any], metric: str) -> float:
    return float(record.get(validation_metric_name(metric), 0.0))


def enrich_run_record(
    record: dict[str, Any],
    cfg: TrainConfig,
    *,
    run_index: int,
    num_runs: int,
    monitor: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched: dict[str, Any] = {
        "run_index": run_index,
        "num_runs": num_runs,
        "seed": cfg.seed,
        "dataset": cfg.data.dataset_name,
        "task_type": cfg.task_type,
        "monitor": validation_metric_name(monitor),
        "checkpoint_mode": cfg.checkpoint_mode,
        "exp_name": cfg.exp_name,
        "exp_dir": str(cfg.exp_dir),
        **record,
    }
    if "best_epoch" not in enriched and "epoch" in enriched:
        enriched["best_epoch"] = enriched["epoch"]
    if cfg.task_type == "graph_classification":
        enriched["split_protocol"] = cfg.data.split_protocol
        enriched["fold_index"] = cfg.data.fold_index
        enriched["num_folds"] = cfg.data.num_folds
        enriched["repetition_index"] = cfg.data.repetition_index
        enriched["num_repetitions"] = cfg.data.num_repetitions
        enriched["split_seed"] = cfg.data.split_seed
        enriched["train_ratio"] = cfg.data.train_ratio
        enriched["val_ratio"] = cfg.data.val_ratio
        enriched["test_ratio"] = cfg.data.test_ratio
    if cfg.task_type == "link_prediction":
        enriched["decoder_type"] = cfg.data.decoder_type
        enriched["target_relation"] = list(cfg.data.target_relation)
    if extra:
        enriched.update(extra)
    return enriched


def summarize_records(
    records: list[dict[str, Any]],
    metric_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        present = [record.get(metric, 0.0) for record in records]
        values = [float(value) for value in present if value is not None]
        summary[metric] = {
            "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else (0.0 if values else None),
        }
        if not values:  # an empty LP test set: None, never a number (spec v1.16 K1)
            reasons = {record.get(metric.replace("_auroc", "_reason").replace("_mrr", "_reason"))
                       for record in records} - {None}
            summary[metric]["reason"] = "; ".join(sorted(reasons)) or "no value"
    return summary


def write_summary_files(
    summary_dir: Path,
    *,
    records: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    config: TrainConfig,
    filename_prefix: str = "runs_summary",
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": config.data.dataset_name,
        "task_type": config.task_type,
        "num_runs": len(records),
        "metrics": summarize_records(records, metric_names),
    }
    if extra_summary:
        summary.update(extra_summary)

    payload = {
        "runs": records,
        "summary": summary,
        "config": dataclass_to_dict(config),
    }
    with (summary_dir / f"{filename_prefix}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    fieldnames = [
        "metric",
        "mean",
        "std",
        "reason",  # set when a metric has no value (an empty LP test set, spec v1.16 K1)
    ]
    with (summary_dir / f"{filename_prefix}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, values in summary["metrics"].items():
            writer.writerow({"metric": metric, **values})
    return payload


def write_run_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
