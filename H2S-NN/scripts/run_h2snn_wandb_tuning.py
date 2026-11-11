from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig, reject_never_tuned_settings
from src.baselines import search_grids as shared_grids
from src.utils.misc import dataclass_to_dict, save_config, seed_everything
from src.utils.run_reports import (
    default_monitor_for_task,
    enrich_run_record,
    metric_score,
    report_metrics_for_task,
    write_run_record,
    write_summary_files,
)

# Sweep orchestration lives in src/utils/wandb_campaign.py so the baseline campaign
# (scripts/run_baseline_tuning.py) drives the identical state/resume/worker machinery
# instead of forking it. Only the H2SNN-specific pieces stay here: the search space
# (build_sweep_config), the params->config mapping (apply_sweep_overrides), and the
# per-trial fit (train_trial). These names are re-exported for existing importers.
from src.utils.wandb_campaign import (  # noqa: F401  (canonical_fingerprint and the
    # state helpers are re-exported for tests/ and for
    # run_graph_classification_repeated_cv_wandb_tuning.py)
    TUNING_STATE_VERSION,
    CampaignConfig,
    CampaignUnit,
    canonical_fingerprint,
    load_tuning_state,
    resolve_campaign_state,
    run_campaign_worker,
    run_unit_sweep,
    save_tuning_state,
    validate_resume_state,
)

# The current model falls back to a single learned identity embedding when every
# node type is featureless.  In that branch ModelConfig.encoder_dropout is not
# consumed, so sweeping it would create duplicate trials.  These are the currently
# supported datasets with that representation (see H2SNN.__init__, which takes the
# identity branch when ModelDims.in_channels == 0).
#
# Verified against the resolved loaders rather than assumed: youtube (2000, 0) and
# lastfm (20612, 0) carry no node features, while pubmed (200) and amazon (1156) do.
# youtube was missing here, so its sweeps were searching a dead encoder_dropout axis.
# Spec v1.15 §4.1 (experiment_settings.html): encoder dropout is absent where the dataset has no
# features: Freebase (NC; LP freebase_bb, every type featureless after F17), LastFM, and graph
# classification (constant feature 1).
FEATURELESS_DATASETS = frozenset({"lastfm", "amazonbook", "youtube", "freebase", "freebase_bb",
                                  "mutag", "proteins", "enzymes", "nci1"})

# Spec v1.15 §6 Tuning (J2, J3; card Part M, M10): H2S-NN's space holds only components of the
# method, with the shared lr / weight-decay / dropout ranges of every method.
# The shared axes are those of every baseline (src/baselines/search_grids.py: LR, WEIGHT_DECAY,
# DROPOUT), one definition for every method.
SPACE_D: list[int] = [16, 32, 64]
SPACE_LR: dict[str, Any] = shared_grids.LR
SPACE_WEIGHT_DECAY: dict[str, Any] = shared_grids.WEIGHT_DECAY
SPACE_DROPOUT: dict[str, Any] = shared_grids.DROPOUT
SPACE_WISP_MAX_LENGTH: dict[str, Any] = {"values": [2, 3, 4, 5]}
SPACE_WISP_WALKS_PER_ANCHOR: dict[str, Any] = {"values": [8, 16, 32, 64]}
SPACE_GRAPH_POOLING: dict[str, Any] = {"values": ["mean", "max", "add", "attention"]}
# Tuning on seed 0 only, then the best configuration on seeds 0-4 (spec §6).
TUNING_SEED = 0
FINAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

PRIMARY_PARAMETER_NAMES = frozenset(
    {
        "d",
        "lr",
        "weight_decay",
        "wisp_max_length",
        "wisp_walks_per_anchor",
    }
)
CONDITIONAL_PARAMETER_NAMES = frozenset(
    {
        "encoder_dropout",
        "classifier_dropout",
        "graph_pooling",
        "decoder_type",
    }
)
KNOWN_PARAMETER_NAMES = PRIMARY_PARAMETER_NAMES | CONDITIONAL_PARAMETER_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W&B plain Bayesian tuning of H2SNN on seed 0 (spec v1.15 §6), then the best "
                    "configuration on seeds 0-4 with --final_seeds")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument(
        "--run_start",
        type=int,
        default=0,
        help="Zero-based campaign run index at which this process starts.",
    )
    parser.add_argument(
        "--run_count",
        type=int,
        default=None,
        help="Number of campaign run indices to execute; defaults to --num_runs.",
    )
    parser.add_argument("--n_trials", type=int, default=100)  # spec §6: 100 trials
    parser.add_argument(
        "--d_values",
        type=int,
        nargs="+",
        default=None,
        help="Optional positive stalk dimensions; default 16/32/64 (spec v1.15 §6).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--decoder_type", type=str, default=None, choices=["dot", "distmult", "concat_mlp"])
    parser.add_argument("--include_decoder_type", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="h2snn")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument(
        "--final_seeds",
        type=int,
        nargs="+",
        default=None,
        help="Instead of tuning: train the best configuration of the seed-0 sweep "
        "(best_tuned_run0.json) on these seeds (spec §6: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed-trial accounting from the local tuning state JSON.",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default="",
        help="Existing W&B sweep id to verify while resuming a single-run fingerprinted state.",
    )
    parser.add_argument(
        "--state_dir",
        type=Path,
        default=None,
        help="Directory for per-run tuning_state_run*.json files. Defaults to output_dir/exp_name.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate the resolved config, sweep search space, and resume state without "
        "contacting W&B or starting training.",
    )
    parser.add_argument("--worker_context", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_raw_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    reject_never_tuned_settings(raw)  # fact_layer: untied is never tuned (plan §9.1)
    return raw


def load_config(path: Path) -> TrainConfig:
    return TrainConfig.from_dict(load_raw_config(path))


def default_d_values(task_type: str) -> list[int]:
    """`d` in {16, 32, 64} for every task (spec v1.15 §6)."""
    return list(SPACE_D)


def validate_d_values(values: list[int] | tuple[int, ...]) -> list[int]:
    if not values:
        raise ValueError("At least one --d_values entry is required.")
    resolved: list[int] = []
    for value in values:
        dimension = int(value)
        if dimension <= 0:
            raise ValueError(f"All --d_values entries must be positive, got {dimension}.")
        if dimension not in resolved:
            resolved.append(dimension)
    return resolved


def validate_positive_int_values(
    values: list[int] | tuple[int, ...],
    *,
    option_name: str,
) -> list[int]:
    if not values:
        raise ValueError(f"At least one {option_name} entry is required.")
    resolved: list[int] = []
    for value in values:
        dimension = int(value)
        if dimension <= 0:
            raise ValueError(
                f"All {option_name} entries must be positive, got {dimension}."
            )
        if dimension not in resolved:
            resolved.append(dimension)
    return resolved


def validate_trial_params(
    params: dict[str, Any],
    *,
    allowed_parameters: set[str] | frozenset[str] | None = None,
) -> None:
    """Reject stale/foreign W&B parameters before applying a trial.

    W&B can resume a remote sweep whose parameter schema predates the local
    script.  Exact allowed names from the fingerprinted sweep are therefore
    used in workers.  The public fallback accepts every parameter supported by
    this script, while still rejecting legacy keys rather than ignoring them.
    """

    allowed = set(KNOWN_PARAMETER_NAMES if allowed_parameters is None else allowed_parameters)
    unsupported_allowed = sorted(allowed - KNOWN_PARAMETER_NAMES)
    if unsupported_allowed:
        raise ValueError(
            "Local sweep declared unsupported tuning parameter name(s): "
            f"{', '.join(unsupported_allowed)}."
        )
    unknown = sorted(set(params) - allowed)
    if unknown:
        rendered = ", ".join(f"{key}={params[key]!r}" for key in unknown)
        raise ValueError(f"W&B proposed unsupported tuning parameter(s): {rendered}.")

    for key in ("d", "wisp_max_length", "wisp_walks_per_anchor"):
        if key in params and int(params[key]) <= 0:
            raise ValueError(
                f"W&B proposed invalid {key}={params[key]!r}; {key} must be positive."
            )
    for key in ("lr",):
        if key in params and float(params[key]) <= 0.0:
            raise ValueError(
                f"W&B proposed invalid {key}={params[key]!r}; {key} must be positive."
            )
    if "weight_decay" in params and float(params["weight_decay"]) < 0.0:
        raise ValueError(
            "W&B proposed invalid weight_decay="
            f"{params['weight_decay']!r}; weight_decay must be non-negative."
        )
    if "graph_pooling" in params and str(params["graph_pooling"]) not in {
            "mean", "max", "add", "attention"}:
        raise ValueError(f"W&B proposed invalid graph_pooling={params['graph_pooling']!r}.")
    for key in ("encoder_dropout", "classifier_dropout"):
        if key in params and not 0.0 <= float(params[key]) < 1.0:
            raise ValueError(
                f"W&B proposed invalid {key}={params[key]!r}; {key} must be in [0, 1)."
            )
    if "decoder_type" in params and str(params["decoder_type"]) not in {
        "dot",
        "distmult",
        "concat_mlp",
    }:
        raise ValueError(f"W&B proposed invalid decoder_type={params['decoder_type']!r}.")


def _apply_base_cli(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = copy.deepcopy(raw)
    if args.exp_name is not None:
        payload["exp_name"] = args.exp_name
    if args.output_dir is not None:
        payload["output_dir"] = str(args.output_dir)
    if args.device is not None:
        payload["device"] = args.device
    if args.patience is not None:
        payload["early_stopping_patience"] = args.patience
    if args.epochs is not None:
        payload["epochs"] = args.epochs
    payload.setdefault("data", {})
    if args.decoder_type is not None:
        payload["data"]["decoder_type"] = args.decoder_type
    payload["use_wandb"] = True
    return payload


def build_sweep_config(
    *,
    task_type: str,
    metric: str,
    epochs: int,
    decoder_type: str | None = None,
    dataset_name: str = "",
    include_decoder_type: bool = False,
    d_values: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """The W&B sweep of H2S-NN (spec v1.15 §6 Tuning; card Part M, M10): plain Bayes, no
    Hyperband or other early termination; the space holds only components of the method:
    d, lr, weight_decay, encoder/classifier dropout (shared ranges), l_max, r, and GC pooling;
    LP's decoder_type only with --include_decoder_type (open). `epochs` is not used by the
    sweep (no early termination); it is kept for the callers' signature."""
    if include_decoder_type and task_type != "link_prediction":
        raise ValueError("--include_decoder_type is valid only for link prediction.")

    resolved_d_values = validate_d_values(
        d_values if d_values is not None else default_d_values(task_type)
    )
    parameters: dict[str, Any] = {
        "d": {"values": resolved_d_values},
        "lr": dict(SPACE_LR),
        "weight_decay": {"values": list(SPACE_WEIGHT_DECAY["values"])},
        "wisp_max_length": {"values": list(SPACE_WISP_MAX_LENGTH["values"])},
        "wisp_walks_per_anchor": {"values": list(SPACE_WISP_WALKS_PER_ANCHOR["values"])},
    }
    # encoder_dropout is unconsumed without node features (spec §4.1: absent for featureless
    # datasets), so it is omitted rather than generating duplicate trials.
    if dataset_name.lower() not in FEATURELESS_DATASETS:
        parameters["encoder_dropout"] = dict(SPACE_DROPOUT)

    # The classifier dropout exists in the NC head and in LP's concat_mlp decoder; dot and
    # DistMult have none, and the GC readout is pooling + one Linear (no dropout, M5).
    classifier_dropout_active = task_type == "node_classification" or (
        task_type == "link_prediction" and not include_decoder_type
        and decoder_type == "concat_mlp"
    )
    if classifier_dropout_active:
        parameters["classifier_dropout"] = dict(SPACE_DROPOUT)
    if task_type == "graph_classification":
        parameters["graph_pooling"] = {"values": list(SPACE_GRAPH_POOLING["values"])}

    if include_decoder_type:
        parameters["decoder_type"] = {"values": ["dot", "distmult", "concat_mlp"]}

    reject_never_tuned_settings(None, parameters)
    return {
        "method": "bayes",
        "metric": {"name": f"best/{metric.removeprefix('val_')}", "goal": "maximize"},
        "parameters": parameters,
    }


def apply_sweep_overrides(
    raw: dict[str, Any],
    params: dict[str, Any],
    *,
    allowed_parameters: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    validate_trial_params(params, allowed_parameters=allowed_parameters)
    reject_never_tuned_settings(raw, params)
    payload = copy.deepcopy(raw)
    model = payload.setdefault("model", {})
    optim = payload.setdefault("optim", {})
    data = payload.setdefault("data", {})

    for key in ("encoder_dropout", "classifier_dropout"):
        if key in params:
            model[key] = params[key]
    if "d" in params:
        model["d"] = int(params["d"])
    if "graph_pooling" in params:
        model["graph_pooling"] = str(params["graph_pooling"])
    for key in ("wisp_max_length", "wisp_walks_per_anchor"):
        if key in params:
            data[key] = int(params[key])

    for key in ("lr", "weight_decay"):
        if key in params:
            optim[key] = params[key]
    if "decoder_type" in params:
        data["decoder_type"] = params["decoder_type"]
    return payload


def train_trial(
    raw_base: dict[str, Any],
    *,
    params: dict[str, Any],
    allowed_parameters: set[str] | frozenset[str] | None = None,
    run_index: int,
    num_runs: int,
    seed: int,
    base_exp_name: str,
    output_dir: Path,
    monitor: str,
    trial_name: str,
    unit_label: str = "run",
    unit_name: str | None = None,
    record_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_trial_params(params, allowed_parameters=allowed_parameters)
    from src.builders import build_experiment_components

    raw = apply_sweep_overrides(
        raw_base,
        params,
        allowed_parameters=allowed_parameters,
    )
    unit_token = unit_name or f"{unit_label}{run_index}"
    raw["exp_name"] = f"{base_exp_name}_tune_{unit_token}_{trial_name}"
    raw["output_dir"] = str(output_dir)
    raw["seed"] = seed
    raw["checkpoint_metric"] = monitor.removeprefix("val_")
    raw["checkpoint_mode"] = "max"
    raw["use_wandb"] = False
    raw.setdefault("data", {})
    raw["data"]["run_index"] = run_index
    raw["data"]["num_runs"] = num_runs

    cfg = TrainConfig.from_dict(raw)
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.exp_dir / "config.yaml")
    print(
        "Resolved trial: "
        f"device={cfg.device} orthogonal_map={cfg.model.orthogonal_map} "
        f"d={cfg.model.d} exp_dir={cfg.exp_dir}",
        flush=True,
    )
    seed_everything(cfg.seed, cfg.device)

    data, _, _, _, logger, _, trainer = build_experiment_components(cfg)
    try:
        monitor_key = monitor.removeprefix("val_")
        best_seen = float("-inf")

        def _epoch_callback(
            epoch: int,
            train_metrics: dict[str, float],
            val_metrics: dict[str, float],
            test_metrics: dict[str, float],
        ) -> None:
            nonlocal best_seen
            best_seen = max(best_seen, float(val_metrics.get(monitor_key, 0.0)))
            try:
                import wandb  # type: ignore[import-not-found]

                payload = {
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                    **{f"test/{key}": value for key, value in test_metrics.items()},
                }
                payload[f"best/{monitor_key}"] = best_seen
                wandb.log(payload, step=epoch)
            except ImportError:
                return

        trainer.fit(data, epoch_callback=_epoch_callback)
        raw_record = getattr(trainer, "_best_test_metrics", None)
        if raw_record is None:
            raise RuntimeError(f"Tuning trial {trial_name} did not produce best metrics.")
        return enrich_run_record(
            raw_record,
            cfg,
            run_index=run_index,
            num_runs=num_runs,
            monitor=monitor,
            extra={
                "trial_name": trial_name,
                "sweep_params": params,
                **(record_extra or {}),
            },
        )
    finally:
        logger.close()


def summarize_tuned_seed(
    records: list[dict[str, Any]],
    *,
    monitor: str,
    out_path: Path,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("No completed tuning records to summarize.")
    best = max(records, key=lambda record: metric_score(record, monitor))
    write_run_record(out_path, best)
    return best


def _default_state_path(base_cfg: TrainConfig, base_exp_name: str, run_index: int) -> Path:
    return base_cfg.output_dir / base_exp_name / f"tuning_state_run{run_index}.json"


def _build_sweep_for_run(
    *,
    args: argparse.Namespace,
    base_cfg: TrainConfig,
    base_exp_name: str,
    monitor: str,
    run_index: int,
    unit_label: str = "run",
    unit_name: str | None = None,
) -> dict[str, Any]:
    sweep_config = build_sweep_config(
        task_type=base_cfg.task_type,
        metric=monitor,
        epochs=base_cfg.epochs,
        decoder_type=base_cfg.data.decoder_type,
        dataset_name=base_cfg.data.dataset_name,
        include_decoder_type=args.include_decoder_type,
        d_values=args.d_values,
    )
    unit_token = unit_name or f"{unit_label}{run_index}"
    sweep_config["name"] = f"{base_exp_name}_{unit_token}_sweep"
    return sweep_config


def _campaign_config(args: argparse.Namespace) -> CampaignConfig:
    """Project the H2SNN CLI namespace onto the engine's campaign settings."""

    return CampaignConfig(
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_group=args.wandb_group,
        wandb_tags=tuple(args.wandb_tags or ()),
        n_trials=args.n_trials,
        resume=args.resume,
        sweep_id=args.sweep_id,
    )


def run_agent_worker(context_path: Path) -> None:
    """Run one H2SNN trial in this worker process.

    `train_trial` is looked up on the module at call time, so patching it in tests
    still takes effect through this delegation.
    """

    run_campaign_worker(
        context_path,
        train_trial=train_trial,
        known_parameters=KNOWN_PARAMETER_NAMES,
        validate_params=lambda params, allowed: validate_trial_params(
            params, allowed_parameters=allowed
        ),
        validate_raw_base=lambda raw_base: TrainConfig.from_dict(raw_base),
    )


def run_seed_sweep(
    raw_base: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_index: int,
    num_runs: int,
    seed: int,
    base_exp_name: str,
    base_cfg: TrainConfig,
    monitor: str,
    state_path: Path,
    run_count: int = 1,
    unit_label: str = "run",
    unit_name: str | None = None,
    unit_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune one unit (a seed, or a repetition/fold pair) via the shared campaign engine.

    Signature is unchanged for existing callers, including
    scripts/run_graph_classification_repeated_cv_wandb_tuning.py.
    """

    sweep_config = _build_sweep_for_run(
        args=args,
        base_cfg=base_cfg,
        base_exp_name=base_exp_name,
        monitor=monitor,
        run_index=run_index,
        unit_label=unit_label,
        unit_name=unit_name,
    )
    # Fingerprint the fully resolved dataclass so defaults cannot disappear merely
    # because a YAML key was omitted.
    base_config_fingerprint = canonical_fingerprint(dataclass_to_dict(base_cfg))

    return run_unit_sweep(
        raw_base,
        campaign=_campaign_config(args),
        unit=CampaignUnit(
            run_index=run_index,
            num_runs=num_runs,
            seed=seed,
            label=unit_label,
            name=unit_name,
            metadata=dict(unit_metadata or {}),
        ),
        sweep_config=sweep_config,
        base_exp_name=base_exp_name,
        output_dir=base_cfg.output_dir,
        task_type=base_cfg.task_type,
        base_config_fingerprint=base_config_fingerprint,
        monitor=monitor,
        state_path=state_path,
        # Each trial re-spawns this script with --worker_context.
        worker_entrypoint=Path(__file__).resolve(),
        summarize=lambda records, monitor_name, out_path: summarize_tuned_seed(
            records, monitor=monitor_name, out_path=out_path
        ),
        run_count=run_count,
    )


def validate_tuning_setup(
    raw_config: dict[str, Any],
    *,
    args: argparse.Namespace,
    base_cfg: TrainConfig,
    base_exp_name: str,
    monitor: str,
    state_dir: Path,
    num_runs: int,
    run_start: int = 0,
    run_count: int | None = None,
) -> None:
    base_fingerprint = canonical_fingerprint(dataclass_to_dict(base_cfg))
    effective_run_count = num_runs if run_count is None else run_count
    for run_index in range(run_start, run_start + effective_run_count):
        state_path = state_dir / f"tuning_state_run{run_index}.json"
        sweep_config = _build_sweep_for_run(
            args=args,
            base_cfg=base_cfg,
            base_exp_name=base_exp_name,
            monitor=monitor,
            run_index=run_index,
        )
        allowed_parameters = set(sweep_config["parameters"])
        for parameter_name, specification in sweep_config["parameters"].items():
            candidates = specification.get("values")
            if candidates is None:
                candidates = [specification["min"], specification["max"]]
            for candidate in candidates:
                trial_params = {parameter_name: candidate}
                resolved_trial = apply_sweep_overrides(
                    raw_config,
                    trial_params,
                    allowed_parameters=allowed_parameters,
                )
                TrainConfig.from_dict(resolved_trial)
        sweep_fingerprint = canonical_fingerprint(sweep_config)
        existing_state = load_tuning_state(state_path)
        resume_action = "fresh"
        resolved_state_path = state_path
        if args.resume:
            if args.sweep_id:
                validate_resume_state(
                    existing_state,
                    state_path=state_path,
                    base_config_fingerprint=base_fingerprint,
                    sweep_config_fingerprint=sweep_fingerprint,
                )
                resume_action = "resume"
            else:
                resolved_state_path, existing_state, resume_action = resolve_campaign_state(
                    state_path,
                    base_config_fingerprint=base_fingerprint,
                    sweep_config_fingerprint=sweep_fingerprint,
                )
        elif existing_state is not None:
            raise ValueError(
                f"Fresh-sweep validation found existing state at {state_path}. "
                "Set --resume only for the exact same sweep, or choose a new state directory."
            )
        print(
            json.dumps(
                {
                    "run_index": run_index,
                    "state_path": str(resolved_state_path),
                    "resume": bool(args.resume),
                    "resume_action": resume_action,
                    "base_config_fingerprint": base_fingerprint,
                    "sweep_config_fingerprint": sweep_fingerprint,
                    "parameter_names": sorted(sweep_config["parameters"]),
                    "search_space": sweep_config["parameters"],
                    "d_values": sweep_config["parameters"]["d"]["values"],
                },
                indent=2,
            )
        )


def run_final_seeds(
    raw_config: dict[str, Any],
    *,
    args: argparse.Namespace,
    base_cfg: TrainConfig,
    base_exp_name: str,
) -> list[dict[str, Any]]:
    """Spec v1.15 §6: the best configuration of the seed-0 sweep (`best_tuned_run0.json`, its
    `sweep_params`) trained on each of `args.final_seeds` (0-4), one record per seed, then a
    summary. No W&B sweep; the configuration is not re-tuned per seed."""
    summary_dir = base_cfg.output_dir / base_exp_name
    best_path = summary_dir / "best_tuned_run0.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} is missing: run the seed-0 sweep first.")
    with best_path.open("r", encoding="utf-8") as handle:
        best = json.load(handle)
    params = dict(best["sweep_params"])
    monitor = (args.monitor or default_monitor_for_task(base_cfg.task_type)).removeprefix("val_")
    seeds = list(args.final_seeds)
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        record = train_trial(
            raw_config, params=params, run_index=index, num_runs=len(seeds), seed=seed,
            base_exp_name=base_exp_name, output_dir=summary_dir / "final_seeds",
            monitor=monitor, trial_name=f"final_seed{seed}", unit_name=f"final_seed{seed}",
            record_extra={"selected_from": str(best_path), "final_seed": seed},
        )
        write_run_record(summary_dir / f"final_seed{seed}.json", record)
        records.append(record)
    write_summary_files(
        summary_dir, records=records, metric_names=report_metrics_for_task(base_cfg.task_type),
        config=base_cfg, filename_prefix="final_seeds_summary",
        extra_summary={"monitor": f"val_{monitor}", "seeds": seeds, "sweep_params": params})
    return records


def main() -> None:
    args = parse_args()
    if args.worker_context is not None:
        run_agent_worker(args.worker_context)
        return
    if args.config is None:
        raise ValueError("--config is required unless --worker_context is used.")
    if args.n_trials <= 0:
        raise ValueError(f"--n_trials must be positive, got {args.n_trials}.")
    if args.run_start < 0:
        raise ValueError(f"--run_start must be non-negative, got {args.run_start}.")
    if args.sweep_id and not args.resume:
        raise ValueError(
            "--sweep_id is accepted only together with --resume and an existing "
            "fingerprinted local tuning state."
        )
    raw_config = _apply_base_cli(load_raw_config(args.config), args)
    base_cfg = TrainConfig.from_dict(raw_config)
    if args.device is None and base_cfg.device == "cuda" and not torch.cuda.is_available():
        raw_config["device"] = "cpu"
        base_cfg = TrainConfig.from_dict(raw_config)
    base_exp_name = base_cfg.exp_name
    if args.final_seeds is not None:
        run_final_seeds(raw_config, args=args, base_cfg=base_cfg, base_exp_name=base_exp_name)
        return
    # Spec v1.15 §6: tuning on seed 0 only (one run index, seed 0) unless named explicitly.
    num_runs = args.num_runs or 1
    run_count = num_runs if args.run_count is None else args.run_count
    if run_count <= 0:
        raise ValueError(f"--run_count must be positive, got {run_count}.")
    if args.run_start + run_count > num_runs:
        raise ValueError(
            "The selected run partition exceeds the campaign: "
            f"run_start={args.run_start}, run_count={run_count}, num_runs={num_runs}."
        )
    if args.sweep_id and run_count != 1:
        raise ValueError("--sweep_id is only unambiguous when --run_count 1 is used.")
    base_seed = args.base_seed if args.base_seed is not None else TUNING_SEED
    monitor = (args.monitor or default_monitor_for_task(base_cfg.task_type)).removeprefix("val_")
    state_dir = args.state_dir or (base_cfg.output_dir / base_exp_name)

    if args.validate_only:
        validate_tuning_setup(
            raw_config,
            args=args,
            base_cfg=base_cfg,
            base_exp_name=base_exp_name,
            monitor=monitor,
            state_dir=state_dir,
            num_runs=num_runs,
            run_start=args.run_start,
            run_count=run_count,
        )
        return

    best_records: list[dict[str, Any]] = []
    for run_index in range(args.run_start, args.run_start + run_count):
        state_path = state_dir / f"tuning_state_run{run_index}.json"
        best_records.append(
            run_seed_sweep(
                raw_config,
                args=args,
                run_index=run_index,
                num_runs=num_runs,
                seed=base_seed + run_index,
                base_exp_name=base_exp_name,
                base_cfg=base_cfg,
                monitor=monitor,
                state_path=state_path,
                run_count=run_count,
            )
        )

    summary_records: list[dict[str, Any]] = []
    summary_dir = base_cfg.output_dir / base_exp_name
    for run_index in range(num_runs):
        best_path = summary_dir / f"best_tuned_run{run_index}.json"
        if best_path.exists():
            with best_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if isinstance(record, dict):
                summary_records.append(record)
    if not summary_records:
        summary_records = best_records

    write_summary_files(
        summary_dir,
        records=summary_records,
        metric_names=report_metrics_for_task(base_cfg.task_type),
        config=base_cfg,
        filename_prefix="tuning_summary",
        extra_summary={
            "monitor": f"val_{monitor}",
            "tuning": "wandb_plain_bayes_seed0",
            "run_partition": {
                "start": args.run_start,
                "count": run_count,
                "total": num_runs,
            },
        },
    )


if __name__ == "__main__":
    main()
