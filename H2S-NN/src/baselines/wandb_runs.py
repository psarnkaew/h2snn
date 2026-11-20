"""W&B runs for every run of the campaign (card 8 C7, amendment 6).

Tuning trials already run inside a W&B run (the sweep agent, `src/utils/wandb_campaign.py`); final seeds of both
launchers and published runs open one here: project as the line's, group `<cell token>_final` /
`<cell token>_published`, tags `final_seed` / `published`, name `<cell token>_seed<s>`. At the end of every run the
summary gets the selected epoch, the record's `val_*` and `test_*`, the wall time, the GPU name, `git rev-parse HEAD`
and the fully resolved configuration (`run.config.update`). W&B never fails a run: every call here logs its error and
returns.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return ""


def json_safe(value: Any) -> Any:
    """A JSON-serialisable copy of a configuration (dataclasses, paths, tuples, sets)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    return json.loads(json.dumps(value, default=lambda v: sorted(v) if isinstance(v, set) else str(v)))


def run_name(token: str, seed: int) -> str:
    return f"{token}_seed{seed}"


def open_run(*, project: str | None, kind: str, token: str, seed: int, config: dict[str, Any],
             entity: str | None = "", mode: str | None = "online", extra_tags: tuple[str, ...] = ()) -> Any | None:
    """A W&B run for one final-seed (`kind` "final_seed") or published (`kind` "published") run, or None. No run (a
    no-op) when no project is given or W&B is disabled (`mode` "disabled", or `WANDB_MODE=disabled`)."""
    import os

    if kind not in ("final_seed", "published"):
        raise ValueError(f"unknown run kind {kind!r}")
    if not project or (mode or "online") == "disabled" or os.environ.get("WANDB_MODE") == "disabled":
        return None
    mode = mode or "online"
    group = f"{token}_{'final' if kind == 'final_seed' else 'published'}"
    try:
        import wandb  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {"project": project, "mode": mode, "group": group,
                                  "tags": list(dict.fromkeys([kind, *extra_tags])), "name": run_name(token, seed),
                                  "config": json_safe(config), "reinit": True}
        if entity:
            kwargs["entity"] = entity
        return wandb.init(**kwargs)
    except Exception as exc:  # noqa: BLE001 - W&B never fails a run
        print(f"wandb.init failed ({kind} {token} seed {seed}): {type(exc).__name__}: {exc}", flush=True)
        return None


def summary_values(record: dict[str, Any], *, started: float) -> dict[str, Any]:
    """The end-of-run summary of a record (card 8 C7)."""
    values: dict[str, Any] = {"selected_epoch": record.get("best_epoch")}
    values.update({k: v for k, v in record.items()
                   if (k.startswith("val_") or k.startswith("test_")) and isinstance(v, (int, float, str, bool))})
    values.update({"wall_time_s": time.time() - started, "gpu_name": gpu_name(), "git_head": git_head()})
    return values


def write_summary(run: Any | None, record: dict[str, Any], *, config: dict[str, Any], started: float) -> None:
    """Summary and resolved configuration of a finished run (a final seed, a published run or a tuning trial)."""
    if run is None:
        return
    try:
        for key, value in summary_values(record, started=started).items():
            run.summary[key] = value
        run.config.update(json_safe(config), allow_val_change=True)
    except Exception as exc:  # noqa: BLE001
        print(f"wandb summary failed: {type(exc).__name__}: {exc}", flush=True)


def finish(run: Any | None) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # noqa: BLE001
        print(f"wandb finish failed: {type(exc).__name__}: {exc}", flush=True)


__all__ = ["finish", "git_head", "gpu_name", "json_safe", "open_run", "run_name", "summary_values", "write_summary"]
