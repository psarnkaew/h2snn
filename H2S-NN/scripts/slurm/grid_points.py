"""Card 10 L2 (Amendment 1, option A): one fixed point of a launcher's own search space, and the peak-GPU-memory line
every trial prints (A2.2). Used by `scripts/run_baseline_tuning.py` and `scripts/run_h2snn_wandb_tuning.py`
(`--grid_point {min,max}`, probe lines only: `--dry_run --n_trials 1`); read by `scripts/slurm/fit_check.py`.

`--grid_point max`: every **size axis** at its largest value (card 10 L2), every **memory-relevant categorical** key at its
most memory-hungry value (card 10 A3.1; a grid over its values where the code does not say which is the largest:
one trial per value, the cell's peak is the largest), every other axis at its central value.
`--grid_point min`: every axis at its smallest point (card 8 C5, `src.baselines.campaign.smallest_grid_point`).
Every search key must be classified below: a key in none of the three sets raises, so a new size axis is never silently put at
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
#: card 10 A3.1: non-size keys whose value changes model or activation memory. "value": the most memory-hungry value;
#: "values": the code does not say which is the largest, so the probe runs once per listed value (a grid; values the
#: cell's space does not have are dropped). Each with its reason (file:line under H2S-NN/).
MEMORY_AXES: dict[str, dict[str, Any]] = {
    "feats_type": {"values": [0, 2, 3], "why": "0 keeps dense native attributes, 2/3 give a type a sparse identity of "
                   "width = its node count (src/baselines/bundles.py:202-230): which is larger depends on the dataset"},
    "label_feats": {"value": True, "why": "builds the propagated label features (src/baselines/adapters/sehgnn.py:146-160,"
                    "300)"},
    "residual": {"value": True, "why": "adds res_fc and its input (vendor/SeHGNN/hgb/model.py:142-143,218-219)"},
    "sheaf_model": {"values": ["BundleSheaf", "GeneralSheaf"], "why": "Diag learns d values per edge (vendor/neural-"
                    "sheaf-diffusion/models/disc_models.py:39); General learns d x d (231) and inverts d x d degree "
                    "blocks by SVD (laplacian_builders.py:280-286), Bundle adds orthogonal-map intermediates per edge "
                    "(orthogonal.py:45-64): Bundle vs General not clear"},
    "sheaf": {"values": [f"{family}/{learner}" for family in ("bundle", "general") for learner in (
        "local_concat", "type_concat", "type_ensemble", "edge_type_concat", "node_type_concat", "node_type",
        "edge_type", "types_only", "attention", "attention_type_concat", "attention_type_ensemble",
        "attention_edge_encoding", "attention_node_encoding", "attention_types_only", "attention_node_type",
        "attention_edge_type")],
        "why": "HetSheaf: diag is the smallest family (d values per edge, as NSD disc_models.py:39); bundle vs general and "
               "the 16 learners are not clear (src/baselines/adapters/hetsheaf.py:105-123)"},
    "orth": {"value": "cayley", "why": "cayley learns d(d+1)/2 values per map vs d(d-1)/2 (vendor/neural-sheaf-diffusion/"
             "models/disc_models.py:146-150; vendor/Cooperative-Sheaf-NN/model.py:322-326) and solves a d x d system "
             "per edge (orthogonal.py:61-64); euler builds Q from 1 or 3 values and only exists for d <= 3 (orthogonal.py:69-74)"},
    "add_lp": {"value": True, "why": "adds one stalk dimension (vendor/neural-sheaf-diffusion/models/sheaf_base.py:21-26)"},
    "add_hp": {"value": True, "why": "adds one stalk dimension (vendor/neural-sheaf-diffusion/models/sheaf_base.py:21-26)"},
    "conformal": {"value": True, "why": "one more sheaf-learner output per node (vendor/Cooperative-Sheaf-NN/model.py:523)"},
    "gnn_type": {"value": "GAT", "why": "GATConv adds per-edge attention coefficients to the per-edge messages GCN/SAGE "
                 "also build (vendor/Cooperative-Sheaf-NN/model.py:577-583)"},
    "metapath_pe": {"value": True, "why": "adds the Hetero_Metapath channel (src/baselines/adapters/h2gformer.py:20-22,"
                    "229-231)"},
    "mask_lp": {"value": True, "why": "adds the label-embedding input (src/baselines/adapters/hetero2net.py:269-273)"},
    "r": {"values": [0.3, 0.5], "why": "Hetero2Net drop_metapath: r both starts random walks and drops their paths "
          "(vendor/Hetero2Net/hetero2net/metapath.py:75-125): not clear"},
    "poly": {"values": ["GPR_GPR", "GPR_legendre", "GPR_jaco"], "why": "three polynomial bases with different "
             "recurrences (vendor/H2SGNN/model.py:80-90,129): not clear"},
    "attn_dropout": {"value": 0.2, "why": "nn.Dropout on the attention weights (vendor/H2GB/H2GB/layer/gt_layer.py:203) "
                     "stores a mask only for p > 0 (torch dropout returns its input at p = 0); the central value of "
                     "[0.0, 0.2] is 0.0"},
    "decoder_type": {"value": "distmult", "why": "relation-specific bilinear: per-relation weights and one more product "
                     "per scored pair (src/models/link_readout.py:46,65; src/baselines/heads.py:65)"},
    "graph_pooling": {"value": "attention", "why": "a gate network and per-node scores (src/models/graph_readout.py:55)"},
}
#: every other search key: rates, regularisers, temperatures and dropouts whose central value is already > 0
OTHER_AXES = frozenset({
    "a", "b", "alpha", "alpha_list", "beta", "beta_lr", "beta_wd", "dprate", "dropout", "input_drop", "input_dropout",
    "lr", "p", "prop_lr", "prop_wd", "temp_list_lr", "temp_list_wd", "temp_lr", "temp_wd", "temperature",
    "weight_decay",
    # H2S-NN
    "encoder_dropout", "classifier_dropout",
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
    unknown = sorted(set(parameters) - SIZE_AXES - set(MEMORY_AXES) - OTHER_AXES)
    if unknown:
        raise ValueError(f"search keys not classified as size / memory / other axes (scripts/slurm/grid_points.py): "
                         f"{unknown}")
    if which == "min":
        from src.baselines.campaign import smallest_grid_point  # card 8 C5: the same smallest point

        return {key: {"value": smallest_grid_point(spec)} for key, spec in parameters.items()}
    point: dict[str, dict[str, Any]] = {}
    for key, spec in parameters.items():
        if key in SIZE_AXES:
            point[key] = {"value": largest(spec)}
        elif key in MEMORY_AXES:
            point[key] = hungriest(key, spec)
        else:
            point[key] = {"value": central(spec)}
    return point


def hungriest(key: str, spec: dict[str, Any]) -> dict[str, Any]:
    """The most memory-hungry value of a memory-relevant key, or the grid of its candidate values (A3.1)."""
    rule = MEMORY_AXES[key]
    offered = list(spec["values"]) if "values" in spec else [spec.get("value")]
    if "value" in rule:
        if rule["value"] not in offered:
            raise ValueError(f"{key}: most memory-hungry value {rule['value']!r} is not in the search space {offered}")
        return {"value": rule["value"]}
    candidates = [v for v in rule["values"] if v in offered]
    if not candidates:
        raise ValueError(f"{key}: none of {rule['values']} is in the search space {offered}")
    return {"value": candidates[0]} if len(candidates) == 1 else {"values": candidates}


def grid_size(sweep: dict[str, Any]) -> int:
    """Trials of a fixed-point grid: one per combination of the grid keys' values."""
    size = 1
    for spec in sweep["parameters"].values():
        size *= len(spec["values"]) if "values" in spec else 1
    return size


def print_grid(points: int) -> None:
    """Announces how many trials (FIT_PEAK lines) the probe runs; fit_check needs every one of them."""
    print(f"FIT_GRID {json.dumps({'points': int(points)})}", flush=True)


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
