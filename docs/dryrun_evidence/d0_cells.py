"""Card 2 D0: the cell list (no runs, no data read). Run from H2S-NN/ in a SLURM step.

Writes d0_cells.json: every registry cell (runnable or not, with reason), and for each H2S-NN cell the
configuration source (experiment YAML or dataset preset) and whether it passes TrainConfig validation.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path.cwd()))
from src.baselines.registry import DATASETS, iter_cells  # noqa: E402
from src.configs.config import TrainConfig, H2GB_NC_DATASETS, LARGE_LP_DATASETS  # noqa: E402

OUT = Path(sys.argv[1])
EXPERIMENTS = Path("configs/experiments")

rows = iter_cells()
baseline = [dict(r, kind="baseline") for r in rows]

h2snn = []
for task, datasets in DATASETS.items():
    for dataset in datasets:
        yaml_path = EXPERIMENTS / f"{dataset}_h2snn_v3b.yaml"
        if yaml_path.exists():
            source = str(yaml_path)
            raw = yaml.safe_load(yaml_path.read_text())
        else:
            source = "dataset preset only ({'data': {'dataset_name': ...}})"
            raw = {"data": {"dataset_name": dataset}}
        try:
            cfg = TrainConfig.from_dict(raw)
            status, error = "resolves", ""
            resolved = {"task_type": cfg.task_type, "training_mode": cfg.data.training_mode,
                        "wisp_walks_per_anchor": cfg.data.wisp_walks_per_anchor,
                        "epochs": cfg.epochs}
        except Exception as exc:  # recorded, not silenced: this is the D0 inventory
            status, error = "does not resolve", f"{type(exc).__name__}: {exc}"
            resolved = {}
            traceback.print_exc()
        h2snn.append({"kind": "h2snn", "model": "h2snn", "task": task, "dataset": dataset,
                      "config_source": source, "config_status": status, "error": error,
                      "resolved": resolved,
                      "h2gb_nc": dataset in H2GB_NC_DATASETS, "large_lp": dataset in LARGE_LP_DATASETS})

OUT.write_text(json.dumps({"baseline_cells": baseline, "h2snn_cells": h2snn}, indent=1, default=str))
print("baseline cells", len(baseline), "runnable", sum(r["runnable"] for r in baseline))
for r in h2snn:
    print(f"h2snn {r['task']:22s} {r['dataset']:12s} {r['config_status']:18s} {r['config_source'][:40]:40s} {r['error'][:150]}")
