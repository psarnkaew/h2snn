from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.builders import build_experiment_components
from src.configs.config import TrainConfig
from src.utils.misc import save_config, seed_everything
from src.utils.run_reports import (
    default_monitor_for_task,
    enrich_run_record,
    report_metrics_for_task,
    write_run_record,
    write_summary_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated-run H2SNN experiment driver")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--num_folds", type=int, default=None, help="Compatibility alias for --num_runs")
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--pooling_type",
        type=str,
        default=None,
        choices=["meanpool", "grampool", "s2pool", "sheafpool", "sheafcov", "sheafconv"],
    )
    parser.add_argument("--decoder_type", type=str, default=None, choices=["dot", "distmult", "concat_mlp"])
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_log_artifacts", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return TrainConfig.from_dict(payload)


def _apply_common_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.device is not None:
        cfg.device = args.device
    elif cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    if args.pooling_type is not None:
        cfg.data.pooling_type = args.pooling_type
    if args.decoder_type is not None:
        cfg.data.decoder_type = args.decoder_type
    if args.train_ratio is not None:
        cfg.data.train_ratio = args.train_ratio
    if args.val_ratio is not None:
        cfg.data.val_ratio = args.val_ratio
    if args.test_ratio is not None:
        cfg.data.test_ratio = args.test_ratio
    if args.monitor is not None:
        cfg.checkpoint_metric = args.monitor.removeprefix("val_")
    else:
        cfg.checkpoint_metric = default_monitor_for_task(cfg.task_type)
    cfg.checkpoint_mode = "max"
    if args.patience is not None:
        cfg.early_stopping_patience = args.patience
    if args.use_wandb:
        cfg.use_wandb = True
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        cfg.wandb_entity = args.wandb_entity
    if args.wandb_mode is not None:
        cfg.wandb_mode = args.wandb_mode
    if args.wandb_group is not None:
        cfg.wandb_group = args.wandb_group
    if args.wandb_tags is not None:
        cfg.wandb_tags = tuple(args.wandb_tags)
    if args.wandb_log_artifacts:
        cfg.wandb_log_artifacts = True
    return cfg


def clone_run_config(
    config_path: Path,
    args: argparse.Namespace,
    *,
    run_index: int,
    num_runs: int,
    base_exp_name: str,
    base_seed: int,
    wandb_group: str,
) -> TrainConfig:
    cfg = _apply_common_overrides(load_config(config_path), args)
    cfg.exp_name = f"{base_exp_name}_run{run_index}"
    cfg.seed = base_seed + run_index
    cfg.data.run_index = run_index
    cfg.data.num_runs = num_runs
    cfg.checkpoint_mode = "max"
    if cfg.use_wandb:
        cfg.wandb_group = wandb_group
        cfg.wandb_tags = tuple(dict.fromkeys((*cfg.wandb_tags, cfg.task_type, cfg.data.dataset_name)))
    return cfg


def run_single_experiment(
    cfg: TrainConfig,
    *,
    run_index: int,
    num_runs: int,
    monitor: str,
) -> dict[str, Any]:
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.exp_dir / "config.yaml")
    print(
        "Resolved run: "
        f"device={cfg.device} orthogonal_map={cfg.model.orthogonal_map} "
        f"d={cfg.model.d} exp_dir={cfg.exp_dir}",
        flush=True,
    )
    seed_everything(cfg.seed, cfg.device)

    data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
    try:
        logger.info(
            "Dataset: %s | Task: %s | Run: %d/%d | Seed: %d | Monitor: %s",
            cfg.data.dataset_name,
            cfg.task_type,
            run_index,
            num_runs,
            cfg.seed,
            monitor,
        )
        if cfg.task_type == "graph_classification":
            logger.info(
                "Pooling: %s | Split: %.2f/%.2f/%.2f",
                cfg.data.pooling_type,
                cfg.data.train_ratio,
                cfg.data.val_ratio,
                cfg.data.test_ratio,
            )
        elif cfg.task_type == "link_prediction":
            logger.info("Target relation: %s | Decoder: %s", cfg.data.target_relation, cfg.data.decoder_type)
        else:
            logger.info("Target node type: %s", cfg.data.target_node_type)
        logger.info("Model parameters: %d", sum(parameter.numel() for parameter in model.parameters()))
        trainer.fit(data)
        raw_record = getattr(trainer, "_best_test_metrics", None)
        if raw_record is None:
            raise RuntimeError(f"Run {run_index} did not produce best metrics.")
        record = enrich_run_record(
            raw_record,
            cfg,
            run_index=run_index,
            num_runs=num_runs,
            monitor=monitor,
        )
        out_path = cfg.exp_dir / f"best_test_metrics_run{run_index}.json"
        write_run_record(out_path, record)
        logger.log_summary(record, prefix="best_run")
        return record
    finally:
        logger.close()


def log_wandb_aggregate(base_cfg: TrainConfig, payload: dict[str, Any], group: str) -> None:
    if not base_cfg.use_wandb:
        return
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("use_wandb=True requires the 'wandb' package.") from exc

    init_kwargs: dict[str, Any] = {
        "project": base_cfg.wandb_project,
        "name": f"{base_cfg.exp_name}_summary",
        "mode": base_cfg.wandb_mode,
        "group": group,
        "tags": list(dict.fromkeys((*base_cfg.wandb_tags, "summary", base_cfg.task_type))),
        "config": payload.get("config", {}),
        "dir": str(base_cfg.output_dir / base_cfg.exp_name),
    }
    if base_cfg.wandb_entity:
        init_kwargs["entity"] = base_cfg.wandb_entity
    run = wandb.init(**init_kwargs)
    try:
        aggregate_metrics = {}
        for metric, values in payload["summary"]["metrics"].items():
            aggregate_metrics[f"aggregate/{metric}_mean"] = values["mean"]
            aggregate_metrics[f"aggregate/{metric}_std"] = values["std"]
            run.summary[f"aggregate/{metric}_mean"] = values["mean"]
            run.summary[f"aggregate/{metric}_std"] = values["std"]
        wandb.log(aggregate_metrics)
    finally:
        wandb.finish()


def main() -> None:
    args = parse_args()
    base_cfg = _apply_common_overrides(load_config(args.config), args)
    base_exp_name = base_cfg.exp_name
    num_runs = args.num_runs or args.num_folds or base_cfg.data.num_runs
    base_seed = args.base_seed if args.base_seed is not None else base_cfg.seed
    monitor = base_cfg.checkpoint_metric.removeprefix("val_")
    wandb_group = base_cfg.wandb_group or base_exp_name

    records: list[dict[str, Any]] = []
    for run_index in range(num_runs):
        cfg = clone_run_config(
            args.config,
            args,
            run_index=run_index,
            num_runs=num_runs,
            base_exp_name=base_exp_name,
            base_seed=base_seed,
            wandb_group=wandb_group,
        )
        records.append(
            run_single_experiment(
                cfg,
                run_index=run_index,
                num_runs=num_runs,
                monitor=monitor,
            )
        )

    summary_dir = base_cfg.output_dir / base_exp_name
    payload = write_summary_files(
        summary_dir,
        records=records,
        metric_names=report_metrics_for_task(base_cfg.task_type),
        config=base_cfg,
    )
    log_wandb_aggregate(base_cfg, payload, wandb_group)


if __name__ == "__main__":
    main()
