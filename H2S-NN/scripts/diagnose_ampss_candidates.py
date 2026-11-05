"""Inspect the task-aligned AMPSS Stage-I report for a resolved experiment config.

The active selector enumerates the complete identity set before applying the
``max_schema_identities`` and ``max_schema_candidates`` fail-fast guards. This
command runs the same task builder used by training and prints its immutable
``schema_conditioned_v1`` report. Link reports include endpoint-vs-positive-pair
coverage diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.data.graph_classification_dataset import build_graph_classification_dataset
from src.data.link_prediction_dataset import build_link_prediction_dataset


def _build_report(cfg: TrainConfig):
    if cfg.task_type == "graph_classification":
        return build_graph_classification_dataset(cfg).selector_report
    if cfg.task_type == "link_prediction":
        return build_link_prediction_dataset(cfg).selector_report
    return build_dataset(cfg).selector_report


def main(config_path: Path, output: Path | None) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["device"] = "cpu"
    cfg = TrainConfig.from_dict(payload)
    if not (cfg.discovery.enabled and cfg.discovery.mode == "auto"):
        raise ValueError("The diagnostic requires discovery.enabled=true and mode='auto'.")

    torch.set_grad_enabled(False)
    report = _build_report(cfg)
    if report is None:
        result = {
            "protocol_version": cfg.discovery.protocol_version,
            "status": "no_selected_materialized_faces",
        }
    else:
        result = report.to_dict()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.config, args.output)
