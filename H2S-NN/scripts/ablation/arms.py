"""Card 4 K6 (1): the ablation arms of pre-registration §2 (`docs/ablation_analysis_preregistration_draft.md`;
spec 1.23 X-6, X-11) and configurations (a): `full`'s tuned configuration file + the arm's one change.

Per the card (amended 2026-11-16): the fixed-literature-schema row is NOT an ablation configuration (a
main-table row, seeds 0-4, Welch); the appendix pair is two configurations (`full` and `endpoints`, each
with `path_weight_context: true`; seeds 1-10, no claim).

    python scripts/ablation/arms.py --full_config <full tuned yaml> --dataset acm --out_dir <dir>

writes `<dir>/<dataset>/<arm>.yaml` (the configuration the seed-run launcher reads) for every arm and
checks each with `TrainConfig.from_dict`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.configs.config import TrainConfig  # noqa: E402

# arm -> {section: {key: value}}: the arm's one change from `full` (S0: no_path_cells with maps = I).
ARM_CHANGES: dict[str, dict[str, dict[str, Any]]] = {
    "full": {},
    "endpoints": {"ablation": {"path_weight_input": "endpoints"}},
    "schema": {"ablation": {"path_weight_input": "schema"}},
    "uniform": {"ablation": {"path_weight_input": "uniform"}},
    "no_path_cells": {"model": {"arm": "no_path_cells"}},
    "closure_only": {"model": {"arm": "closure_only"}},
    "instance_features": {"model": {"arm": "instance_features"}},
    "hub_A": {"model": {"arm": "hub_A"}},
    "covering_only": {"model": {"arm": "covering_only"}},
    "S2": {"ablation": {"fact_maps_identity": True}},
    "S0": {"model": {"arm": "no_path_cells"}, "ablation": {"fact_maps_identity": True}},
    "direction_blind_maps": {"ablation": {"direction_blind_maps": True}},
    "free_cell_maps": {"model": {"arm": "free_cell_maps"}},
    "geometry_context_off": {"model": {"geometry_context": False}},
    "h0_only": {"ablation": {"h0_only_readout": True}},
    "appendix_full": {"ablation": {"path_weight_context": True}},
    "appendix_endpoints": {"ablation": {"path_weight_input": "endpoints",
                                        "path_weight_context": True}},
}
# Arms with configuration (b) (tuned on their own; B-2, B-4) and their `tuning.arm` name.
TUNED_ARMS: tuple[str, ...] = ("endpoints", "closure_only", "instance_features", "no_path_cells",
                               "S2")
# Seeds of every ablation run (B-1); `full`'s seeds 1-4 are the main table's (looked up).
SEEDS: tuple[int, ...] = tuple(range(1, 11))
REUSED_FULL_SEEDS: tuple[int, ...] = (1, 2, 3, 4)


def apply_change(full_raw: dict[str, Any], arm: str) -> dict[str, Any]:
    raw = copy.deepcopy(full_raw)
    for section, values in ARM_CHANGES[arm].items():
        raw[section] = {**(raw.get(section) or {}), **values}
    return raw


def flatten(raw: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def emit_configurations_a(full_config: Path, dataset: str, out_dir: Path) -> dict[str, Path]:
    full_raw = yaml.safe_load(full_config.read_text(encoding="utf-8"))
    name = str(full_raw.get("data", {}).get("dataset_name", "")).lower()
    if name != dataset.lower():
        raise ValueError(f"{full_config} is a {name!r} configuration, not {dataset!r}.")
    if str((full_raw.get("model") or {}).get("arm", "full")) != "full" or full_raw.get("tuning"):
        raise ValueError("the input must be full's (main-table) configuration.")
    base_name = str(full_raw.get("exp_name", f"{dataset}_h2snn"))
    paths = {}
    for arm in ARM_CHANGES:
        raw = apply_change(full_raw, arm)
        raw["exp_name"] = f"{base_name}__ablation_a_{arm}"
        TrainConfig.from_dict(copy.deepcopy(raw))          # every emitted configuration builds
        path = out_dir / dataset / f"{arm}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        paths[arm] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    paths = emit_configurations_a(args.full_config, args.dataset, args.out_dir)
    print(json.dumps({arm: str(path) for arm, path in paths.items()}, indent=1))


if __name__ == "__main__":
    main()
