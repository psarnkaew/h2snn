"""Load hyperparameters the baseline authors published for our datasets.

Each file under ``configs/baselines/<model>/<dataset>.yaml`` carries a ``source:`` field
naming the exact file or README line it was transcribed from, so a results table can state
which cells inherited published settings and a reader can check the transcription.

Cells with no published values return ``None`` and are tuned instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.baselines.protocol import BaselineParams
from src.baselines.runtime import PROJECT_ROOT

CONFIG_ROOT = PROJECT_ROOT / "configs" / "baselines"


def published_params_path(model: str, task: str, dataset: str) -> Path:
    """Layout: ``configs/baselines/<model>/<task>/<dataset>.yaml``."""

    return CONFIG_ROOT / model / task / f"{dataset}.yaml"


def load_published_params(model: str, task: str, dataset: str) -> BaselineParams | None:
    path = published_params_path(model, task, dataset)
    if not path.exists():
        return None

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Published parameter file must be a mapping: {path}")

    values: dict[str, Any] = dict(payload.get("params", {}))
    if not values:
        raise ValueError(
            f"{path} has no 'params:' block; delete the file rather than shipping an "
            "empty published-hyperparameter cell."
        )
    if not payload.get("source"):
        raise ValueError(
            f"{path} is missing 'source:'. Published values must name where they came "
            "from so the transcription can be checked."
        )
    return BaselineParams(values=values, source="published")


#: Keys of a source's own optimizer and step rule (card 8 C1): accepted only for `source="published"`; a tuned run
#: (the sweep) refuses them, so the tuned cells keep the shared protocol (AdamW, no scheduler, no clipping, one step per
#: batch, LP batches of 8,192 positives).
SOURCE_SCHEDULE_KEYS: frozenset[str] = frozenset({
    "optimizer", "scheduler", "clip_grad_norm", "batch_accumulation", "lp_batch_size",
})


def source_schedule_keys(values: dict[str, Any]) -> list[str]:
    """The source-schedule keys of `values` (every H2GB `optim.*` key counts)."""
    return sorted(key for key in values if key in SOURCE_SCHEDULE_KEYS or key.startswith("optim."))


def refuse_source_schedule(values: dict[str, Any], source: str) -> None:
    """Raise when a non-published parameter set carries a source's optimizer, scheduler, clipping, accumulation or
    LP step size (card 8 C1: the tuner keeps refusing them)."""
    keys = source_schedule_keys(values)
    if keys and source != "published":
        raise ValueError(f"source-schedule keys {keys} are accepted only for published runs (source={source!r}).")


def available_published_cells() -> dict[str, list[tuple[str, str]]]:
    """Map model -> [(task, dataset)] for every transcribed cell on disk."""

    found: dict[str, list[tuple[str, str]]] = {}
    if not CONFIG_ROOT.is_dir():
        return found
    for model_dir in sorted(p for p in CONFIG_ROOT.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for config in sorted(task_dir.glob("*.yaml")):
                found.setdefault(model_dir.name, []).append(
                    (task_dir.name, config.stem)
                )
    return found
