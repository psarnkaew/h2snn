"""Generate the five-variant AMPSS ablation protocol."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig
from src.utils.ampss_ablation_protocol import (
    AMPSS_ABLATIONS,
    build_ablation_config,
)


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(args: argparse.Namespace) -> None:
    auto_base = _read(args.auto_config)
    manual_base = _read(args.manual_config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for spec in AMPSS_ABLATIONS:
        for seed in args.seeds:
            payload = build_ablation_config(
                auto_base=auto_base,
                manual_base=manual_base,
                spec=spec,
                training_seed=seed,
                output_dir=str(args.output_dir),
                exp_name_prefix=args.exp_name_prefix,
            )
            # Resolve now so malformed or legacy configs fail before experiments start.
            TrainConfig.from_dict(payload)
            path = out_dir / f"{payload['exp_name']}.yaml"
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            rows.append(
                {
                    "ablation_index": spec.index,
                    "ablation": spec.name,
                    "seed": seed,
                    "frozen_stage1": spec.frozen_stage1,
                    "config": str(path),
                }
            )
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} configs and {manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-config", type=Path, required=True)
    parser.add_argument("--manual-config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ampss_ablations"))
    parser.add_argument("--exp-name-prefix", default="ampss")
    main(parser.parse_args())
