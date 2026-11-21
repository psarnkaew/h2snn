"""Card 10 L2 (Amendment 1, option A): one fixed point of a launcher's own search space, and the peak-GPU-memory line
every trial prints (A2.2). Used by `scripts/run_baseline_tuning.py` and `scripts/run_h2snn_wandb_tuning.py`
(`--grid_point {min,max}`, probe lines only: `--dry_run --n_trials 1`); read by `scripts/slurm/fit_check.py`.

`--grid_point max`: every **size axis** at its largest value, every other axis at its central value (card 10 L2).
`--grid_point min`: every axis at its smallest point (card 8 C5, `src.baselines.campaign.smallest_grid_point`).
Every search key must be classified below: a key in neither set raises, so a new size axis is never silently put at
its central value. The classification covers every key of every runnable baseline cell
(`docs/card10_evidence/search_space_keys.txt`) and H2S-NN's space (`build_sweep_config`).
"""

from __future__ import annotations

import json
import math
from typing import Any

#: widths, layers (and hops / polynomial order / sequence length / bases), heads, d, l_max, r, batch size
SIZE_AXES = frozenset({
    # widths
    "hidden", "hidden_dim", "hidden_channels", "hidden_l", "hidden_x", "emb_dim", "embed_dim", "embed_size",
    "attn_vec_dim",
    # layers, hops, propagation order, sequence length, bases
    "layers", "layers_l", "layers_x", "n_layers", "n_layers_2", "n_task_layers", "num_layers", "gt_layers",
    "n_fp_layers", "num_gnns", "K", "num_hops", "num_label_hops", "len_seq", "n_bases",
    # heads
    "num_heads", "attn_heads",
    # stalk dimension, H2S-NN l_max and r (walks per anchor), batch size
    "d", "wisp_max_length", "wisp_walks_per_anchor", "batch_size",
})
#: every other search key (rates, dropouts, regularisers, temperatures, categorical choices)
OTHER_AXES = frozenset({
    "a", "b", "add_hp", "add_lp", "alpha", "alpha_list", "attn_dropout", "beta", "beta_lr", "beta_wd", "conformal",
    "decoder_type", "dprate", "dropout", "feats_type", "gnn_type", "input_drop", "input_dropout", "label_feats", "lr",
    "mask_lp", "metapath_pe", "orth", "p", "poly", "prop_lr", "prop_wd", "r", "residual", "sheaf", "sheaf_model",
    "temp_list_lr", "temp_list_wd", "temp_lr", "temp_wd", "temperature", "weight_decay",
    # H2S-NN
    "encoder_dropout", "classifier_dropout", "graph_pooling",
})
GRID_POINTS = ("min", "max")
PEAK_TAG = "FIT_PEAK"


def _numeric(values: list[Any]) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def largest(spec: dict[str, Any]) -> Any:
    """The largest value of a size axis (a numeric value list or a fixed value)."""
    if "value" in spec:
        return spec["value"]
    values = list(spec.get("values", []))
    if not values or not _numeric(values):
        raise ValueError(f"size axis without a numeric value list: {spec!r}")
    return max(values)


def central(spec: dict[str, Any]) -> Any:
    """The central value of an axis: the lower median of a numeric list, the middle entry of a categorical list
    (in its listed order), the midpoint of a uniform range, the geometric midpoint of a log-uniform range."""
    if "value" in spec:
        return spec["value"]
    if "values" in spec:
        values = list(spec["values"])
        ordered = sorted(values) if _numeric(values) else values
        return ordered[(len(ordered) - 1) // 2]
    low, high, kind = spec.get("min"), spec.get("max"), spec.get("distribution", "uniform")
    if low is None or high is None:
        raise ValueError(f"no central value for search axis {spec!r}")
    if kind in ("uniform", "q_uniform"):
        return (float(low) + float(high)) / 2
    if kind == "int_uniform":
        return int(round((low + high) / 2))
    if kind in ("log_uniform_values", "q_log_uniform_values"):
        return math.sqrt(float(low) * float(high))
    raise ValueError(f"no central value for distribution {kind!r}")


def fixed_point(parameters: dict[str, Any], which: str) -> dict[str, dict[str, Any]]:
    """The W&B `parameters` block of one fixed point (`{"key": {"value": v}}`), for a `grid` sweep of one trial."""
    if which not in GRID_POINTS:
        raise ValueError(f"grid_point must be one of {GRID_POINTS}, got {which!r}")
    unknown = sorted(set(parameters) - SIZE_AXES - OTHER_AXES)
    if unknown:
        raise ValueError(f"search keys not classified as size / other axes (scripts/slurm/grid_points.py): {unknown}")
    if which == "min":
        from src.baselines.campaign import smallest_grid_point  # card 8 C5: the same smallest point

        return {key: {"value": smallest_grid_point(spec)} for key, spec in parameters.items()}
    return {key: {"value": largest(spec) if key in SIZE_AXES else central(spec)} for key, spec in parameters.items()}


def grid_sweep(sweep: dict[str, Any], which: str) -> dict[str, Any]:
    """A launcher's sweep config turned into a one-point grid sweep (name and metric kept)."""
    out = dict(sweep)
    out["parameters"] = fixed_point(sweep["parameters"], which)
    out["method"] = "grid"
    return out


def reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_payload(*, train: bool, val: bool, test: bool) -> dict[str, Any]:
    """Peak GPU memory of this trial process (A2.2). `peak_bytes` = the allocator's peak reserved memory plus the
    memory the device holds outside the allocator at the end (CUDA context, library workspaces): an upper estimate
    of the device peak. The phases say which of training, validation and test evaluation ran before the reading."""
    payload: dict[str, Any] = {"phases": {"train": bool(train), "val": bool(val), "test": bool(test)},
                               "cuda": False, "peak_bytes": None, "total_bytes": None, "gpu_name": None}
    try:
        import torch
    except ImportError:
        return payload
    if not torch.cuda.is_available():
        return payload
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    reserved_now = torch.cuda.memory_reserved(device)
    outside = max(0, (total - free) - reserved_now)
    payload.update({"cuda": True, "gpu_name": torch.cuda.get_device_name(device), "total_bytes": int(total),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "outside_allocator_bytes": int(outside),
                    "peak_bytes": int(torch.cuda.max_memory_reserved(device) + outside)})
    return payload


def print_peak(*, train: bool, val: bool, test: bool) -> None:
    print(f"{PEAK_TAG} {json.dumps(peak_payload(train=train, val=val, test=test), sort_keys=True)}", flush=True)


def record_phases(record: dict[str, Any], monitor: str) -> tuple[bool, bool]:
    """(val, test) evaluated: the trial's record holds the validation monitor and at least one test metric."""
    key = f"val_{monitor.removeprefix('val_')}"
    return key in record, any(str(k).startswith("test_") for k in record)
