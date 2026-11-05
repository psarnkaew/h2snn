"""Hyperparameter tuning for H2SNN using Optuna + Hyperband pruner.

Usage (from repo root):
    python scripts/tune_h2snn.py --config configs/experiments/acm_h2snn.yaml \\
        --n_trials 50 --metric val_macro_f1

Resume a crashed study (same --study_name reuses the SQLite DB automatically):
    python scripts/tune_h2snn.py --config configs/experiments/acm_h2snn.yaml \\
        --n_trials 50 --study_name acm_tune

The study is stored in a SQLite file at experiments/optuna_<study_name>.db so it
survives crashes and can be inspected with the Optuna dashboard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import optuna
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.builders import build_experiment_components
from src.configs.config import TrainConfig
from src.utils.misc import seed_everything


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

def _suggest_overrides(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    """Build a raw config dict with trial-suggested values merged in.

    Uses the raw YAML dict as the base so that TrainConfig.from_dict() (which
    calls DataConfig.resolved()) applies the new face_sample_k correctly to
    the metapath templates.

    d_r and d_h are derived from d so they scale together; MLP hidden dims
    are set to 2×d to avoid an extra search dimension.
    """
    d = trial.suggest_categorical("d", [16, 32, 64, 128])

    model_overrides: dict[str, Any] = {
        # Architecture
        "d": d,
        "d_r": d // 2,
        "d_h": d,
        "K": trial.suggest_int("K", 2, 6),
        "eta_0": trial.suggest_float("eta_0", 0.005, 0.3, log=True),
        "eta_1": trial.suggest_float("eta_1", 0.005, 0.3, log=True),
        "learnable_eta": trial.suggest_categorical("learnable_eta", [True, False]),
        "layer_dropout": trial.suggest_categorical(
            "layer_dropout", [0.0, 0.1, 0.2, 0.3]
        ),
        "use_delta0_init": trial.suggest_categorical("use_delta0_init", [True, False]),
        # Regularisation
        "encoder_dropout": trial.suggest_categorical(
            "encoder_dropout", [0.0, 0.1, 0.2, 0.3]
        ),
        "classifier_dropout": trial.suggest_categorical(
            "classifier_dropout", [0.0, 0.1, 0.2, 0.3]
        ),
        # MLP dims scale with d
        "edge_mlp_hidden_dim": 2 * d,
        "edge_init_hidden_dim": 2 * d,
        "face_mlp_hidden_dim": 2 * d,
        "classifier_hidden_dim": 2 * d,
    }

    optim_overrides: dict[str, Any] = {
        "lr": trial.suggest_categorical(
            "lr", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
        ),
        "scheduler": trial.suggest_categorical(
            "scheduler", ["cosine", "none", "reduce_on_plateau"]
        ),
    }

    data_overrides: dict[str, Any] = {
        "face_sample_k": trial.suggest_categorical("face_sample_k", [20, 50, 100]),
    }

    return {
        **base,
        "model": {**base.get("model", {}), **model_overrides},
        "optim": {**base.get("optim", {}), **optim_overrides},
        "data": {**base.get("data", {}), **data_overrides},
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_objective(
    base_dict: dict[str, Any],
    metric: str,
    output_dir: Path,
    exp_name_prefix: str,
    device: str,
    base_epochs: int,
) -> optuna.ObjectiveFuncType:
    # Strip 'val_' prefix so we can look up the key in val_metrics directly
    metric_key = metric.removeprefix("val_")

    def objective(trial: optuna.Trial) -> float:
        raw = _suggest_overrides(trial, base_dict)
        raw["exp_name"] = f"{exp_name_prefix}_trial{trial.number}"
        raw["output_dir"] = str(output_dir)
        raw["device"] = device
        raw["use_wandb"] = True
        raw["epochs"] = base_epochs

        cfg = TrainConfig.from_dict(raw)
        seed_everything(cfg.seed)
        cfg.exp_dir.mkdir(parents=True, exist_ok=True)

        data, _, _, _, _, _, trainer = build_experiment_components(cfg)

        def _epoch_cb(epoch: int, val_metrics: dict[str, float]) -> None:
            value = val_metrics.get(metric_key, 0.0)
            trial.report(value, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            trainer.fit(data, epoch_callback=_epoch_cb)
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"Trial {trial.number} failed: {exc}", flush=True)
            return float("nan")

        best = trainer._best_test_metrics  # noqa: SLF001
        if best is None:
            return 0.0
        return float(best.get(f"val_{metric_key}", best.get(metric_key, 0.0)))

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna + Hyperband tuning for H2SNN")
    parser.add_argument("--config", type=Path, required=True, help="Base YAML config")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument(
        "--metric",
        type=str,
        default="val_macro_f1",
        help=(
            "Validation metric to maximise. Must be a key in val_metrics "
            "(with or without 'val_' prefix). Default: val_macro_f1"
        ),
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name. Defaults to '<config_stem>_tune'.",
    )
    parser.add_argument(
        "--exp_name_prefix",
        type=str,
        default=None,
        help="Prefix for per-trial experiment dirs. Defaults to '<config_stem>'.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Root output dir for trial dirs and results. Defaults to config output_dir.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel workers (>1 requires shared DB storage).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        base_dict: dict[str, Any] = yaml.safe_load(handle)

    # Resolve CLI overrides that affect config-level fields
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    study_name = args.study_name or f"{args.config.stem}_tune"
    exp_name_prefix = args.exp_name_prefix or args.config.stem
    base_cfg = TrainConfig.from_dict(base_dict)
    output_dir = args.output_dir or base_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_path = output_dir / f"optuna_{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=20,
        max_resource=base_cfg.epochs,
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    objective = _build_objective(
        base_dict=base_dict,
        metric=args.metric,
        output_dir=output_dir,
        exp_name_prefix=exp_name_prefix,
        device=device,
        base_epochs=base_cfg.epochs,
    )

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    # --- Summary ---
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial: #{best.number}  |  {args.metric}: {best.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    results = {
        "study_name": study_name,
        "metric": args.metric,
        "best_value": best.value,
        "best_trial": best.number,
        "best_params": best.params,
        "n_trials_completed": len(completed),
        "n_trials_pruned": len(pruned),
    }
    results_path = output_dir / "tune_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Results saved to {results_path}")
    print(f"Full study DB:   {storage_path}")


if __name__ == "__main__":
    main()
