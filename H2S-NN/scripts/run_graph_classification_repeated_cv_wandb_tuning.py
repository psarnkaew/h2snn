from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_h2snn_wandb_tuning import (
    _apply_base_cli,
    _build_sweep_for_run,
    apply_sweep_overrides,
    canonical_fingerprint,
    load_raw_config,
    load_tuning_state,
    resolve_campaign_state,
    run_seed_sweep,
    save_tuning_state,
    validate_resume_state,
)
from src.configs.config import TrainConfig
from src.utils.misc import dataclass_to_dict
from src.utils.run_reports import default_monitor_for_task, report_metrics_for_task

EVALUATION_PROTOCOL = "graph_classification_repeated_10x10_kfold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated KFold graph-classification W&B tuning for H2SNN"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    # Five, matching the manuscript and the 'fifty runs' reading of the TUDataset
    # paper. Their code uses ten; pass --num_repetitions 10 for that.
    parser.add_argument("--num_repetitions", type=int, default=5)
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--unit_start", type=int, default=0)
    parser.add_argument("--unit_count", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--d_values", type=int, nargs="+", default=None)
    parser.add_argument("--sheafcov_feat_dims", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--pooling_type",
        type=str,
        default=None,
        choices=["meanpool", "grampool", "s2pool", "sheafpool", "sheafcov", "sheafconv"],
    )
    parser.add_argument("--decoder_type", type=str, default=None)
    parser.add_argument("--include_pooling_type", action="store_true")
    parser.add_argument("--include_decoder_type", action="store_true")
    parser.add_argument("--include_discovery", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="h2snn")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--hyperband_min_iter", type=int, default=20)
    parser.add_argument("--hyperband_eta", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sweep_id", type=str, default="")
    parser.add_argument("--state_dir", type=Path, default=None)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def unit_coordinates(unit_index: int, num_folds: int) -> tuple[int, int]:
    if num_folds <= 0:
        raise ValueError(f"num_folds must be positive, got {num_folds}.")
    if unit_index < 0:
        raise ValueError(f"unit_index must be non-negative, got {unit_index}.")
    return divmod(unit_index, num_folds)


def unit_name(repetition_index: int, fold_index: int) -> str:
    return f"rep{repetition_index:02d}_fold{fold_index:02d}"


def build_unit_raw_config(
    raw_base: dict[str, Any],
    *,
    unit_index: int,
    num_repetitions: int,
    num_folds: int,
    base_seed: int,
    split_seed: int,
) -> dict[str, Any]:
    repetition_index, fold_index = unit_coordinates(unit_index, num_folds)
    if repetition_index >= num_repetitions:
        raise ValueError(
            f"unit_index {unit_index} exceeds a {num_repetitions}x{num_folds} campaign."
        )
    raw = copy.deepcopy(raw_base)
    raw["seed"] = int(base_seed) + unit_index
    data = raw.setdefault("data", {})
    data.update(
        {
            "split_protocol": "repeated_kfold",
            "num_repetitions": num_repetitions,
            "repetition_index": repetition_index,
            "num_folds": num_folds,
            "fold_index": fold_index,
            "split_seed": int(split_seed),
            "num_runs": num_repetitions * num_folds,
            "run_index": unit_index,
        }
    )
    return raw


def _population_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    value_mean = sum(values) / len(values)
    variance = sum((value - value_mean) ** 2 for value in values) / len(values)
    return {"mean": value_mean, "std": math.sqrt(variance)}


def summarize_repeated_cv_records(
    records: list[dict[str, Any]],
    *,
    num_repetitions: int,
    num_folds: int,
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    total_units = num_repetitions * num_folds
    by_unit: dict[int, dict[str, Any]] = {}
    for record in records:
        unit_index = int(record["unit_index"])
        if not 0 <= unit_index < total_units:
            raise ValueError(f"Record has out-of-range unit_index {unit_index}.")
        if unit_index in by_unit:
            raise ValueError(f"Duplicate repeated-CV record for unit {unit_index}.")
        by_unit[unit_index] = record

    repetition_summaries: list[dict[str, Any]] = []
    for repetition_index in range(num_repetitions):
        fold_records = [
            by_unit[repetition_index * num_folds + fold_index]
            for fold_index in range(num_folds)
            if repetition_index * num_folds + fold_index in by_unit
        ]
        repetition_summaries.append(
            {
                "repetition_index": repetition_index,
                "completed_folds": sorted(int(item["fold_index"]) for item in fold_records),
                "metrics": {
                    metric: {
                        **_population_stats([float(item[metric]) for item in fold_records]),
                        "fold_values": [float(item[metric]) for item in fold_records],
                    }
                    for metric in metric_names
                },
            }
        )

    metric_summary: dict[str, Any] = {}
    for metric in metric_names:
        all_fold_values = [float(by_unit[index][metric]) for index in sorted(by_unit)]
        repetition_means = [
            float(item["metrics"][metric]["mean"])
            for item in repetition_summaries
            if len(item["completed_folds"]) == num_folds
        ]
        repetition_stats = _population_stats(repetition_means)
        fold_stats = _population_stats(all_fold_values)
        metric_summary[metric] = {
            "mean": repetition_stats["mean"],
            "std_across_repetition_means": repetition_stats["std"],
            "std_across_all_folds": fold_stats["std"],
            "num_complete_repetitions": len(repetition_means),
            "num_fold_scores": len(all_fold_values),
        }

    completed_units = sorted(by_unit)
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "num_repetitions": num_repetitions,
        "num_folds": num_folds,
        "total_units": total_units,
        "completed_units": completed_units,
        "missing_units": [index for index in range(total_units) if index not in by_unit],
        "metrics": metric_summary,
        "repetitions": repetition_summaries,
    }


def write_repeated_cv_summary(
    summary_dir: Path,
    *,
    records: list[dict[str, Any]],
    config: TrainConfig,
    num_repetitions: int,
    num_folds: int,
) -> dict[str, Any]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_repeated_cv_records(
        records,
        num_repetitions=num_repetitions,
        num_folds=num_folds,
        metric_names=report_metrics_for_task(config.task_type),
    )
    payload = {
        "records": sorted(records, key=lambda item: int(item["unit_index"])),
        "summary": summary,
        "config": dataclass_to_dict(config),
    }
    json_path = summary_dir / "repeated_cv_tuning_summary.json"
    save_tuning_state(json_path, payload)
    csv_path = summary_dir / "repeated_cv_tuning_summary.csv"
    csv_tmp_path = csv_path.with_name(f"{csv_path.name}.{os.getpid()}.tmp")
    with csv_tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "metric",
            "mean",
            "std_across_repetition_means",
            "std_across_all_folds",
            "num_complete_repetitions",
            "num_fold_scores",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, values in summary["metrics"].items():
            writer.writerow({"metric": metric, **values})
    csv_tmp_path.replace(csv_path)
    return payload


def _validate_unit_setup(
    raw_unit: dict[str, Any],
    *,
    args: argparse.Namespace,
    cfg: TrainConfig,
    campaign_name: str,
    monitor: str,
    state_path: Path,
    unit_index: int,
    unit_token: str,
) -> None:
    sweep = _build_sweep_for_run(
        args=args,
        base_cfg=cfg,
        base_exp_name=campaign_name,
        monitor=monitor,
        run_index=unit_index,
        unit_label="unit",
        unit_name=unit_token,
    )
    allowed = set(sweep["parameters"])
    for parameter_name, specification in sweep["parameters"].items():
        candidates = specification.get("values")
        if candidates is None:
            candidates = [specification["min"], specification["max"]]
        for candidate in candidates:
            TrainConfig.from_dict(
                apply_sweep_overrides(
                    raw_unit,
                    {parameter_name: candidate},
                    allowed_parameters=allowed,
                )
            )

    base_fingerprint = canonical_fingerprint(dataclass_to_dict(cfg))
    sweep_fingerprint = canonical_fingerprint(sweep)
    existing = load_tuning_state(state_path)
    resolved_state_path = state_path
    resume_action = "fresh"
    if args.resume:
        if args.sweep_id:
            validate_resume_state(
                existing,
                state_path=state_path,
                base_config_fingerprint=base_fingerprint,
                sweep_config_fingerprint=sweep_fingerprint,
            )
            resume_action = "resume"
        else:
            resolved_state_path, _, resume_action = resolve_campaign_state(
                state_path,
                base_config_fingerprint=base_fingerprint,
                sweep_config_fingerprint=sweep_fingerprint,
            )
    elif existing is not None:
        raise ValueError(
            f"Fresh repeated-CV validation found existing state at {state_path}. "
            "Use --resume or choose a new experiment/state directory."
        )
    print(
        json.dumps(
            {
                "unit_index": unit_index,
                "repetition_index": cfg.data.repetition_index,
                "fold_index": cfg.data.fold_index,
                "model_seed": cfg.seed,
                "split_seed": cfg.data.split_seed,
                "state_path": str(resolved_state_path),
                "resume_action": resume_action,
                "base_config_fingerprint": base_fingerprint,
                "sweep_config_fingerprint": sweep_fingerprint,
                "parameter_names": sorted(sweep["parameters"]),
            },
            indent=2,
        )
    )


def _collect_records(summary_dir: Path, num_repetitions: int, num_folds: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for repetition_index in range(num_repetitions):
        for fold_index in range(num_folds):
            path = summary_dir / f"best_tuned_{unit_name(repetition_index, fold_index)}.json"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if isinstance(record, dict):
                records.append(record)
    return records


def main() -> None:
    args = parse_args()
    if args.num_repetitions <= 0:
        raise ValueError("--num_repetitions must be positive.")
    if args.num_folds < 2:
        raise ValueError("--num_folds must be at least 2.")
    if args.n_trials <= 0:
        raise ValueError("--n_trials must be positive.")
    if args.unit_start < 0:
        raise ValueError("--unit_start must be non-negative.")

    raw_base = _apply_base_cli(load_raw_config(args.config), args)
    initial_cfg = TrainConfig.from_dict(raw_base)
    if initial_cfg.task_type != "graph_classification":
        raise ValueError(
            "Repeated KFold tuning requires a graph-classification dataset config."
        )
    if args.device is None and initial_cfg.device == "cuda" and not torch.cuda.is_available():
        raw_base["device"] = "cpu"
        initial_cfg = TrainConfig.from_dict(raw_base)

    campaign_name = initial_cfg.exp_name
    total_units = args.num_repetitions * args.num_folds
    unit_count = total_units - args.unit_start if args.unit_count is None else args.unit_count
    if unit_count <= 0:
        raise ValueError("--unit_count must be positive.")
    if args.unit_start + unit_count > total_units:
        raise ValueError(
            "The selected unit partition exceeds the campaign: "
            f"unit_start={args.unit_start}, unit_count={unit_count}, total={total_units}."
        )
    if args.sweep_id and (not args.resume or unit_count != 1):
        raise ValueError(
            "--sweep_id requires --resume and exactly one selected campaign unit."
        )

    base_seed = initial_cfg.seed if args.base_seed is None else args.base_seed
    split_seed = base_seed if args.split_seed is None else args.split_seed
    if base_seed < 0:
        raise ValueError("--base_seed must be non-negative.")
    if split_seed < 0:
        raise ValueError("--split_seed must be non-negative.")
    monitor = (args.monitor or default_monitor_for_task(initial_cfg.task_type)).removeprefix("val_")
    summary_dir = initial_cfg.output_dir / campaign_name
    state_dir = args.state_dir or (summary_dir / "tuning_state")

    manifest = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "dataset": initial_cfg.data.dataset_name,
        "num_repetitions": args.num_repetitions,
        "num_folds": args.num_folds,
        "total_units": total_units,
        "base_seed": base_seed,
        "split_seed": split_seed,
        "unit_partition": {"start": args.unit_start, "count": unit_count},
        "estimated_model_fits": total_units * args.n_trials,
    }

    for unit_index in range(args.unit_start, args.unit_start + unit_count):
        repetition_index, fold_index = unit_coordinates(unit_index, args.num_folds)
        token = unit_name(repetition_index, fold_index)
        raw_unit = build_unit_raw_config(
            raw_base,
            unit_index=unit_index,
            num_repetitions=args.num_repetitions,
            num_folds=args.num_folds,
            base_seed=base_seed,
            split_seed=split_seed,
        )
        cfg = TrainConfig.from_dict(raw_unit)
        state_path = state_dir / f"tuning_state_{token}.json"
        unit_args = copy.copy(args)
        unit_args.wandb_group = args.wandb_group or f"{campaign_name}_{token}"
        unit_args.wandb_tags = list(
            dict.fromkeys(
                [
                    *(args.wandb_tags or []),
                    "graph-classification",
                    "repeated-kfold",
                    f"repetition-{repetition_index}",
                    f"fold-{fold_index}",
                ]
            )
        )
        metadata = {
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "unit_index": unit_index,
            "repetition_index": repetition_index,
            "num_repetitions": args.num_repetitions,
            "fold_index": fold_index,
            "num_folds": args.num_folds,
            "split_seed": split_seed,
            "model_seed": cfg.seed,
        }
        if args.validate_only:
            _validate_unit_setup(
                raw_unit,
                args=unit_args,
                cfg=cfg,
                campaign_name=campaign_name,
                monitor=monitor,
                state_path=state_path,
                unit_index=unit_index,
                unit_token=token,
            )
            continue
        run_seed_sweep(
            raw_unit,
            args=unit_args,
            run_index=unit_index,
            num_runs=total_units,
            seed=cfg.seed,
            base_exp_name=campaign_name,
            base_cfg=cfg,
            monitor=monitor,
            state_path=state_path,
            run_count=unit_count,
            unit_label="unit",
            unit_name=token,
            unit_metadata=metadata,
        )

    if args.validate_only:
        print(json.dumps(manifest, indent=2))
        return

    summary_dir.mkdir(parents=True, exist_ok=True)
    save_tuning_state(summary_dir / "repeated_cv_manifest.json", manifest)
    records = _collect_records(summary_dir, args.num_repetitions, args.num_folds)
    summary_cfg = TrainConfig.from_dict(
        build_unit_raw_config(
            raw_base,
            unit_index=0,
            num_repetitions=args.num_repetitions,
            num_folds=args.num_folds,
            base_seed=base_seed,
            split_seed=split_seed,
        )
    )
    write_repeated_cv_summary(
        summary_dir,
        records=records,
        config=summary_cfg,
        num_repetitions=args.num_repetitions,
        num_folds=args.num_folds,
    )


if __name__ == "__main__":
    main()
