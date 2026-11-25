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

from src.configs.config import (H2GB_NC_DATASETS, MINIBATCH_BATCH_SIZES, TrainConfig, arm_search_space,
                               default_training_mode, regularisation_pack_keys_set, reject_never_tuned_settings,
                               without_default_pack_keys)
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
from scripts.slurm.grid_points import grid_size, grid_sweep, print_grid, print_peak, record_phases, reset_peak  # noqa: E402
from src.utils.wandb_campaign import (  # noqa: F401  (canonical_fingerprint and the
    # state helpers are re-exported for tests/ and for
    # run_graph_classification_repeated_cv_wandb_tuning.py)
    TUNING_STATE_VERSION,
    CampaignConfig,
    CampaignUnit,
    load_tuning_state,
    resolve_campaign_state,
    run_campaign_worker,
    run_unit_sweep,
    sweep_objective,
    save_tuning_state,
    validate_resume_state,
)
from src.utils.wandb_campaign import canonical_fingerprint as _campaign_fingerprint  # noqa: E402


def canonical_fingerprint(payload: Any) -> str:
    """`wandb_campaign.canonical_fingerprint` of `payload` without the card-23 keys at their off values
    (`without_default_pack_keys`; a no-op for payloads that do not carry them, e.g. sweep configs). A
    resolved configuration without the regularisation pack keeps its `b0f13fb` base fingerprint, so
    existing H2S-NN sweeps (and the GC repeated-CV sweeps, which import this name) resume unchanged."""
    if isinstance(payload, dict):
        payload = without_default_pack_keys(payload)
    return _campaign_fingerprint(payload)


def base_config_fingerprint(cfg: TrainConfig) -> str:
    """Fingerprint of the fully resolved dataclass, so defaults cannot disappear merely because a YAML
    key was omitted (card-23 keys at their off values excepted, see `canonical_fingerprint`)."""
    return canonical_fingerprint(dataclass_to_dict(cfg))

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
# Spec v1.22 W2: H2S-NN's LP decoder is tuned in {dot, DistMult}, the decoders of the LP head shared
# with every baseline (`src/baselines/heads.py` `build_link_decoder`); `concat_mlp` is removed (from
# tuning, runs, the config and the model; M3a, M3a-fix).
SPACE_DECODER_TYPE: dict[str, Any] = {"values": ["dot", "distmult"]}
# Card 23 (author 2026-11-26; spec 1.33): the space of the re-tuned cells, used when the base config sets a
# key of the regularisation pack (`regularisation_pack_keys_set`; the cells fix data.eval_draws: 8 and
# checkpoint_smoothing: 5): encoder dropout to 0.99 (C2), lr to 3e-2 and d up to 128 on NC (C3), input
# dropout where encoder_dropout is tuned (C1), and the EMA decay on full-batch NC (C5). H2S-NN only: the
# shared `search_grids` ranges and every baseline space are unchanged. Without the pack: the space above.
SPACE_PACK_ENCODER_DROPOUT: dict[str, Any] = {"distribution": "uniform", "min": 0.0, "max": 0.99}
SPACE_PACK_LR: dict[str, Any] = {"distribution": "log_uniform_values", "min": 1e-4, "max": 3e-2}
SPACE_PACK_D_NC: list[int] = [16, 32, 64, 128]
SPACE_PACK_INPUT_DROPOUT: dict[str, Any] = {"distribution": "uniform", "min": 0.0, "max": 0.9}
SPACE_PACK_EMA_DECAY: dict[str, Any] = {"values": [0.0, 0.9, 0.99, 0.999]}
# Card 23 review, item 2: the fixed settings of the re-tuned cells; with the pack space on, a base config with
# other values is refused before the sweep is created.
PACK_EVAL_DRAWS = 8
PACK_CHECKPOINT_SMOOTHING = 5
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
        # Card 2 D6.7 (spec 1.17 Q4, author decision 4 of the card 2 review): the H2GB NC datasets only
        "batch_size",
        # Card 23: the regularisation-pack space only (C1, C5)
        "input_dropout",
        "ema_decay",
    }
)
KNOWN_PARAMETER_NAMES = PRIMARY_PARAMETER_NAMES | CONDITIONAL_PARAMETER_NAMES
#: Card 2 D3: names that may never appear in a sweep (the --dry_run flag and the epoch budget).
DRY_RUN_KEYS: tuple[str, ...] = ("dry_run", "epochs")


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
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Card 2 D3: epoch budget 1 for every trial and final seed (a check that the launcher works end "
        "to end, never a result); W&B runs tagged dry-run. A command-line flag only: refused as a sweep "
        "parameter and as a base-config key.",
    )
    parser.add_argument(
        "--grid_point",
        choices=["min", "max"],
        default=None,
        help="Card 10 L2 (probe lines only, with --dry_run and --n_trials 1): one fixed point of the search space: "
        "every axis at its smallest point (min, as card 8 C5), or every size axis (d, l_max, r, batch size) at its "
        "largest value, the LP decoder and GC pooling at their most memory-hungry value (distmult, attention; card 10 "
        "A3.1) and every other axis at its central value (max), the lab fit check; never a result.",
    )
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--decoder_type", type=str, default=None,
                        choices=list(SPACE_DECODER_TYPE["values"]))
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

    # Card 2 D3: the dry-run flag and the epoch budget are command-line settings, never sweep values.
    for name in DRY_RUN_KEYS:
        if name in params or name in (allowed_parameters or ()):
            raise ValueError(f"{name!r} is not a tuning parameter: the dry run is the launcher's --dry_run "
                             "flag (card 2 D3) and the epoch budget is the protocol's (spec §6).")
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
    for key in ("encoder_dropout", "classifier_dropout", "input_dropout", "ema_decay"):
        if key in params and not 0.0 <= float(params[key]) < 1.0:
            raise ValueError(
                f"W&B proposed invalid {key}={params[key]!r}; {key} must be in [0, 1)."
            )
    if "decoder_type" in params and str(params["decoder_type"]) not in SPACE_DECODER_TYPE["values"]:
        raise ValueError(f"W&B proposed invalid decoder_type={params['decoder_type']!r}; "
                         f"the LP decoder is tuned in {SPACE_DECODER_TYPE['values']} (spec v1.22 W2).")


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
    if getattr(args, "dry_run", False):  # card 2 D3: the explicit dry-run flag (one epoch)
        if args.epochs is not None and args.epochs != 1:
            raise ValueError("--dry_run sets the epoch budget to 1; do not combine it with --epochs.")
        payload["epochs"] = 1
        args.wandb_tags = list(dict.fromkeys([*(args.wandb_tags or ()), "dry-run"]))
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
    d_values: list[int] | tuple[int, ...] | None = None,
    regularisation_pack: bool = False,
    training_mode: str = "",
) -> dict[str, Any]:
    """The W&B sweep of H2S-NN (spec v1.15 §6 Tuning; card Part M, M10): plain Bayes, no
    Hyperband or other early termination; the space holds only components of the method:
    d, lr, weight_decay, encoder/classifier dropout (shared ranges), l_max, r, and GC pooling;
    on every LP task decoder_type in {dot, distmult} (spec v1.22 W2; never on NC or GC); on the eight
    H2GB NC datasets batch_size in {32, 256, 1024} (spec 1.17 Q4; card 2 D6.7; nowhere else).
    `decoder_type` is the base config's decoder: `concat_mlp` is refused (W2), any other value is
    replaced per trial by the tuned one. `epochs` is not used by the sweep (no early termination);
    it is kept for the callers' signature.
    `regularisation_pack` (card 23): the re-tuned cells' space (SPACE_PACK_*); `training_mode` ("" = the
    dataset's) decides full-batch NC, where the EMA decay is tuned."""
    if decoder_type is not None and decoder_type not in SPACE_DECODER_TYPE["values"]:
        raise ValueError(
            f"data.decoder_type: {decoder_type} is not an LP decoder of the tuned model: H2S-NN's "
            f"decoder is tuned in {SPACE_DECODER_TYPE['values']}, the LP head shared with every "
            "baseline (spec v1.22 W2; concat_mlp is removed).")

    if d_values is None and regularisation_pack and task_type == "node_classification":
        d_values = SPACE_PACK_D_NC  # card 23 C3
    resolved_d_values = validate_d_values(
        d_values if d_values is not None else default_d_values(task_type)
    )
    parameters: dict[str, Any] = {
        "d": {"values": resolved_d_values},
        "lr": dict(SPACE_PACK_LR if regularisation_pack else SPACE_LR),  # card 23 C3
        "weight_decay": {"values": list(SPACE_WEIGHT_DECAY["values"])},
        "wisp_max_length": {"values": list(SPACE_WISP_MAX_LENGTH["values"])},
        "wisp_walks_per_anchor": {"values": list(SPACE_WISP_WALKS_PER_ANCHOR["values"])},
    }
    # encoder_dropout is unconsumed without node features (spec §4.1: absent for featureless
    # datasets), so it is omitted rather than generating duplicate trials.
    if dataset_name.lower() not in FEATURELESS_DATASETS:
        parameters["encoder_dropout"] = dict(SPACE_PACK_ENCODER_DROPOUT if regularisation_pack
                                             else SPACE_DROPOUT)  # card 23 C2
        if regularisation_pack:  # card 23 C1: under the condition of encoder_dropout
            parameters["input_dropout"] = dict(SPACE_PACK_INPUT_DROPOUT)

    # The classifier dropout exists in the NC head only: LP's dot and DistMult decoders have none
    # (`classifier_dropout` reached LP only through the removed concat_mlp MLP; M3a-A0), and the
    # GC readout is pooling + one Linear (no dropout, M5).
    if task_type == "node_classification":
        parameters["classifier_dropout"] = dict(SPACE_DROPOUT)
    if task_type == "graph_classification":
        parameters["graph_pooling"] = {"values": list(SPACE_GRAPH_POOLING["values"])}

    if task_type == "link_prediction":
        parameters["decoder_type"] = {"values": list(SPACE_DECODER_TYPE["values"])}

    # Spec 1.17 Q4 (card 2 D6.7): the batch size is tuned on the eight H2GB NC datasets, {32, 256, 1024},
    # the range of every method (MB2); nowhere else (full batch / per graph / LP positives per step).
    if task_type == "node_classification" and dataset_name.lower() in H2GB_NC_DATASETS:
        parameters["batch_size"] = {"values": list(MINIBATCH_BATCH_SIZES)}
    # Card 23 C5: the EMA decay on full-batch NC only (the trainer that implements C4-C6).
    if (regularisation_pack and task_type == "node_classification"
            and (training_mode or default_training_mode(task_type, dataset_name)) == "full_batch"):
        parameters["ema_decay"] = {"values": list(SPACE_PACK_EMA_DECAY["values"])}

    # the dataset name lets the check accept batch_size on H2GB NC and refuse it elsewhere (config.py MB2)
    reject_never_tuned_settings({"data": {"dataset_name": dataset_name}} if dataset_name else None, parameters)
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

    for key in ("encoder_dropout", "classifier_dropout", "input_dropout"):
        if key in params:
            model[key] = params[key]
    if "ema_decay" in params:  # card 23 C5
        optim["ema_decay"] = params["ema_decay"]
    if "d" in params:
        model["d"] = int(params["d"])
    if "graph_pooling" in params:
        model["graph_pooling"] = str(params["graph_pooling"])
    for key in ("wisp_max_length", "wisp_walks_per_anchor", "batch_size"):
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
    import time

    from src.builders import build_experiment_components

    started = time.time()

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
    reset_peak()  # card 10 A2.2: the peak of this trial covers data, training and evaluation

    data, _, _, _, logger, _, trainer = build_experiment_components(cfg)
    # Card 23 review, item 1: with C6 (w > 1) the trial objective is the smoothed validation score at the
    # selected epoch, the quantity the selection maximises; w = 1: the running maximum, as before.
    smoothed = cfg.checkpoint_smoothing > 1
    trained = {"epoch": False}  # card 10 A2.2: a training epoch reached the epoch callback
    stopped = {"epoch": None}  # card 16 part 2: the last epoch the trainer reported = the epoch the trial stopped at
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
            if smoothed:
                best_seen = float((trainer._best_test_metrics or {}).get("val_smoothed_score", best_seen))
            else:
                best_seen = max(best_seen, float(val_metrics.get(monitor_key, 0.0)))
            stopped["epoch"] = int(epoch)
            trained["epoch"] |= bool(train_metrics)
            try:
                import wandb  # type: ignore[import-not-found]

                # Card 2 D3: outside a sweep worker (the --final_seeds path) there is no W&B run, and
                # wandb.log raised "You must call wandb.init() before wandb.log()", so every seed run failed.
                if getattr(wandb, "run", None) is None:
                    return
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
        if smoothed:  # card 23 review, item 1: the objective field holds the smoothed score (what the sweep ranks
            # by, `sweep_objective`); the raw value at the selected epoch stays as `val_<monitor>_raw`
            objective_key = f"val_{monitor_key}"
            raw_record = {**raw_record, f"{objective_key}_raw": raw_record[objective_key],
                          objective_key: raw_record["val_smoothed_score"]}
        # the whole validation and test splits are evaluated into the record (after training under B1a, whose
        # epoch callback sees only the validation subset), and the peak is read after that evaluation
        val_done, test_done = record_phases(raw_record, monitor)
        print_peak(train=trained["epoch"], val=val_done, test=test_done)
        record = enrich_run_record(
            raw_record,
            cfg,
            run_index=run_index,
            num_runs=num_runs,
            monitor=monitor,
            extra={
                "trial_name": trial_name,
                "sweep_params": params,
                **({} if stopped["epoch"] is None else {"stopped_epoch": stopped["epoch"]}),
                **(record_extra or {}),
            },
        )
        try:  # card 8 C7: summary of the active run (a sweep trial or a final seed): resolved config, timing
            import wandb  # type: ignore[import-not-found]

            from src.baselines.wandb_runs import write_summary

            write_summary(getattr(wandb, "run", None), record, config={"resolved": cfg, "params": dict(params)},
                          started=started)
        except ImportError:
            pass
        return record
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
    # Card 5 U1 (decisions log 2026-11-18 "Card 1f U5 review", option A): one criterion through the
    # whole tuning, the sweep objective (the B1a subset Macro-F1 on the H2GB NC datasets; a record of
    # such a dataset without it is refused), as the baselines' `select_best_trial`.
    best = max(records, key=lambda record: sweep_objective(record, monitor))
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
    pack_keys = regularisation_pack_keys_set(base_cfg)
    if pack_keys and (base_cfg.data.eval_draws, base_cfg.checkpoint_smoothing) != (
            PACK_EVAL_DRAWS, PACK_CHECKPOINT_SMOOTHING):
        # Card 23 review, item 2: any pack key switches the card-23 space on; it is tuned only with the
        # re-tuned cells' fixed settings.
        raise ValueError(
            f"{', '.join(pack_keys)} switch(es) on the card-23 search space, which needs data.eval_draws: "
            f"{PACK_EVAL_DRAWS} and checkpoint_smoothing: {PACK_CHECKPOINT_SMOOTHING} in the base config "
            f"(got {base_cfg.data.eval_draws} and {base_cfg.checkpoint_smoothing}).")
    sweep_config = build_sweep_config(
        task_type=base_cfg.task_type,
        metric=monitor,
        epochs=base_cfg.epochs,
        decoder_type=base_cfg.data.decoder_type,
        dataset_name=base_cfg.data.dataset_name,
        d_values=args.d_values,
        regularisation_pack=bool(pack_keys),  # card 23
        training_mode=base_cfg.data.training_mode,
    )
    if base_cfg.tuning.arm:  # spec 1.24 Y4: an arm-tuning sweep drops the arm's inert keys
        sweep_config["parameters"] = arm_search_space(sweep_config["parameters"], base_cfg.tuning.arm)
    if getattr(args, "grid_point", None) is not None:  # card 10 L2: a probe line, one fixed point (grid of one)
        sweep_config = grid_sweep(sweep_config, args.grid_point)
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
    base_fingerprint = base_config_fingerprint(base_cfg)

    campaign = _campaign_config(args)
    if getattr(args, "grid_point", None) is not None:  # card 10 A3.1: one trial per point of the probe's grid
        campaign.n_trials = grid_size(sweep_config)
        print_grid(campaign.n_trials)
    return run_unit_sweep(
        raw_base,
        campaign=campaign,
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
        base_config_fingerprint=base_fingerprint,
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
    base_fingerprint = base_config_fingerprint(base_cfg)
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
    from src.baselines.wandb_runs import finish, open_run

    for index, seed in enumerate(seeds):
        # card 8 C7 (amendment 6): one W&B run per final seed, opened here (no src/ change); the trainer's epoch
        # callback logs train/val/test into it, `train_trial` writes the summary
        run = open_run(project=getattr(args, "wandb_project", None), entity=getattr(args, "wandb_entity", None),
                       mode=getattr(args, "wandb_mode", None), kind="final_seed",
                       token=base_exp_name, seed=int(seed),
                       config={"params": params, "selected_from": str(best_path)})
        try:
            record = train_trial(
                raw_config, params=params, run_index=index, num_runs=len(seeds), seed=seed,
                base_exp_name=base_exp_name, output_dir=summary_dir / "final_seeds",
                monitor=monitor, trial_name=f"final_seed{seed}", unit_name=f"final_seed{seed}",
                record_extra={"selected_from": str(best_path), "final_seed": seed},
            )
        finally:
            finish(run)
        write_run_record(summary_dir / f"final_seed{seed}.json", record)
        records.append(record)
    write_summary_files(
        summary_dir, records=records, metric_names=report_metrics_for_task(base_cfg.task_type),
        config=base_cfg, filename_prefix="final_seeds_summary",
        extra_summary={"monitor": f"val_{monitor}", "seeds": seeds, "sweep_params": params})
    return records


def _merge_counts(counts) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in counts:
        for key, value in (item or {}).items():
            merged[key] = merged.get(key, 0) + int(value)
    return dict(sorted(merged.items()))


def main() -> None:
    args = parse_args()
    if args.worker_context is not None:
        run_agent_worker(args.worker_context)
        return
    if args.config is None:
        raise ValueError("--config is required unless --worker_context is used.")
    if args.grid_point is not None and (not args.dry_run or args.n_trials != 1):  # card 10 L2: probe lines only
        raise SystemExit("--grid_point is for probe lines: use it with --dry_run and --n_trials 1.")
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
            # failed trials (stop condition, T_max, non-finite loss; spec v1.16 K5)
            "failed_trials": sum(int(r.get("failed_trials", 0)) for r in best_records),
            "failed_by_condition": _merge_counts(r.get("failed_by_condition") for r in best_records),
            "run_partition": {
                "start": args.run_start,
                "count": run_count,
                "total": num_runs,
            },
        },
    )


if __name__ == "__main__":
    main()
