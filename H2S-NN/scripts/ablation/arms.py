"""Card 7 V1 (ablation plan v2, `docs/ablation_plan_v2_2026-11-19.html`; author scope rule 2026-11-19): the 7
rows of the ablation table and their configurations (a): `full`'s tuned configuration file + the row's one change.

Rows: `full`, `no_path_cells` (C2-cells), `S2` (maps = I; C15), `S0` (`no_path_cells` + maps = I; 2x2, no claim),
`endpoints` (C2, kill rule), `instance_features` (C14), and `literature` (the fixed-literature-schema row,
`path_source: schemas`, reported without a claim). Card 7 makes the literature row a configuration (a) with seeds
1-10 (it supersedes the card 4 note that called it a main-table row; spec 1.27 will record the change); it is built
by the existing `fixed_literature_schema_config` (only `data.path_source` and the run name change). Every other arm
of the old pre-registration (`closure_only`, `hub_A`, `covering_only`, `schema`, `uniform`, `direction_blind_maps`,
`free_cell_maps`, `ego_head`, `h0_only_readout`) is out of scope: it stays in the frozen `src/` and is in no list
here.

    python scripts/ablation/arms.py --full_config <full tuned yaml> --dataset acm --out_dir <dir>

writes `<dir>/<dataset>/<row>.yaml` (the configuration the seed-run launcher reads) for every row and checks each
with `TrainConfig.from_dict`.
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

# row -> {section: {key: value}}: the row's one change from `full` (S0: no_path_cells with maps = I). The
# literature row's change is applied by `fixed_literature_schema_config` (`LITERATURE_ROW`), not by a raw merge.
LITERATURE_ROW = "literature"
ARM_CHANGES: dict[str, dict[str, dict[str, Any]]] = {
    "full": {},
    "no_path_cells": {"model": {"arm": "no_path_cells"}},
    "S2": {"ablation": {"fact_maps_identity": True}},
    "S0": {"model": {"arm": "no_path_cells"}, "ablation": {"fact_maps_identity": True}},
    "endpoints": {"ablation": {"path_weight_input": "endpoints"}},
    "instance_features": {"model": {"arm": "instance_features"}},
    LITERATURE_ROW: {"data": {"path_source": "schemas"}},
}
# Arms with configuration (b) (tuned on their own: 100 trials, seed 0; plan v2) and their `tuning.arm` name.
TUNED_ARMS: tuple[str, ...] = ("endpoints", "no_path_cells", "instance_features", "S2")
# Seeds of every ablation run (plan v2); `full`'s seeds 1-4 are the main table's (looked up).
SEEDS: tuple[int, ...] = tuple(range(1, 11))
REUSED_FULL_SEEDS: tuple[int, ...] = (1, 2, 3, 4)


def apply_change(full_raw: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm == LITERATURE_ROW:
        return literature_row(full_raw)
    raw = copy.deepcopy(full_raw)
    for section, values in ARM_CHANGES[arm].items():
        raw[section] = {**(raw.get(section) or {}), **values}
    return raw


def literature_row(full_raw: dict[str, Any]) -> dict[str, Any]:
    """The fixed-literature-schema row of `full_raw` through the project's own builder
    (`fixed_literature_schema_config`: the resolved configuration with `data.path_source: schemas`; it raises if any
    other setting differs). Returned as the saved-configuration dict (`dataclass_to_dict`), the form a run's
    `config.yaml` takes."""
    from src.configs.config import fixed_literature_schema_config
    from src.utils.misc import dataclass_to_dict

    row = fixed_literature_schema_config(TrainConfig.from_dict(copy.deepcopy(full_raw)))
    return dataclass_to_dict(row)


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
