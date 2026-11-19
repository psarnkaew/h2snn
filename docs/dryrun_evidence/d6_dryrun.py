"""Card 2 D6: dry run of one cell on real data, as D2 (`d2_dryrun.py`, reused unchanged), at a stated point.

    python ../docs/dryrun_evidence/d6_dryrun.py --cell 'h2snn|node_classification|ogbn-mag' --out <dir> \
        --attempt <label> [--point max|central] [--set axis=value,axis=value]          (from H2S-NN/)

--point max      D2's largest memory point (d2_dryrun.baseline_point / h2snn_raw).
--point central  D6.4 (decision B2a): each numeric grid axis at the middle value of its grid (the lower middle for an
                 even count), log-uniform axes at the geometric middle, uniform axes at the middle, categoricals at the
                 adapter's default (its published value for the cell where it has one, else the grid's first value;
                 H2S-NN: the experiment YAML's value). H2S-NN on H2GB NC: batch size 256 (middle of {32, 256, 1024}).
--set            D6.3 step-down ladder: the given axes replaced; every value must be a value of the axis's grid and
                 not above the point's value (one grid step at a time is the ladder's rule, d6_ladders.txt).
Adds to D2's record: `point_kind`, `val_targets` (validation targets of one full validation pass) and, for the
baselines' H2GB mini-batch loop, the validation / test pass times (`minibatch.evaluate_split` timed as phases).
No metric value is read, stored or printed (card 2 Rule).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.cwd()))

import d2_dryrun as d2  # noqa: E402

ARGS: dict[str, Any] = {"point": "max", "set": {}}
EXTRA: dict[str, Any] = {}
H2GB_NC = d2.H2GB_NC

import time as _time  # noqa: E402

_T0 = _time.perf_counter()


def mark(what: str) -> None:
    """D6b: a log line with the elapsed time, so a run killed at its time limit still gives build / train times."""
    print(f"D6MARK {what} t={_time.perf_counter() - _T0:.1f}s", flush=True)


def parse_value(text: str) -> Any:
    if text in ("True", "False"):
        return text == "True"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def middle(options: list[Any]) -> Any:
    ordered = sorted(options)
    return ordered[(len(ordered) - 1) // 2]


def is_numeric_grid(options: list[Any]) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in options)


def central_point(space: dict[str, Any], default: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    values, rule = {}, {}
    for name, spec in space.items():
        if "values" in spec:
            options = list(spec["values"])
            if is_numeric_grid(options):
                values[name], rule[name] = middle(options), "middle of the grid"
            elif name in default and default[name] in options:
                values[name], rule[name] = default[name], "categorical: adapter default (published value)"
            else:
                values[name], rule[name] = options[0], "categorical: adapter default (first grid value)"
        elif "value" in spec:
            values[name], rule[name] = spec["value"], "fixed"
        elif spec.get("distribution") == "log_uniform_values":
            values[name], rule[name] = math.sqrt(spec["min"] * spec["max"]), "geometric middle"
        elif spec.get("distribution") in ("uniform", "q_uniform"):
            values[name], rule[name] = (spec["min"] + spec["max"]) / 2, "middle"
        else:
            raise ValueError(f"unknown search-space entry {name}: {spec}")
    return values, rule


def apply_set(values: dict[str, Any], grids: dict[str, list[Any]], rule: dict[str, str]) -> None:
    for name, value in ARGS["set"].items():
        if name not in grids:
            raise ValueError(f"step-down axis {name} is not a grid axis of this cell ({sorted(grids)})")
        grid = list(grids[name])
        if value not in grid:
            raise ValueError(f"{name}={value} is not a value of its grid {grid}")
        current = values[name]
        if isinstance(current, bool):
            if value is True and current is False:
                raise ValueError(f"{name}: True is heavier than the point's False")
        elif value > current:
            raise ValueError(f"{name}={value} is above the point's {current}")
        values[name] = value
        rule[name] = "D6.3 step-down"


# ------------------------------------------------------------------------------------------- baselines
_orig_baseline_point = d2.baseline_point


def patched_baseline_point(space: dict[str, Any]):
    cell = EXTRA["cell"]
    if ARGS["point"] == "central":
        from src.baselines.registry import get_adapter

        model, task, dataset = cell.split("|")
        published = get_adapter(model).published_params(task, dataset)
        default = dict(getattr(published, "values", {}) or {}) if published is not None else {}
        values, rule = central_point(space, default)
    else:
        values, rule = _orig_baseline_point(space)
    apply_set(values, {k: s["values"] for k, s in space.items() if "values" in s}, rule)
    patch_metric_timestamps()
    return values, rule


def patch_metric_timestamps() -> None:
    """Timestamps of the metric calls (values never read): with the first `.eval()` they bound the validation pass
    of loops that evaluate in an inner function (`minibatch.train_index_minibatch`, Hetero2Net's and SeHGNN's
    own loops): validation = first eval -> 1st metric call, test = 1st -> 2nd metric call."""
    import time

    for name, module in list(sys.modules.items()):
        if not name.startswith("src.baselines") or module is None:
            continue
        original = getattr(module, "classification_metrics", None)
        if original is None or getattr(original, "_d6", False):
            continue

        def stamped(*a, _orig=original, **k):
            EXTRA.setdefault("metric_calls", []).append(time.perf_counter())
            return _orig(*a, **k)

        stamped._d6 = True
        setattr(module, "classification_metrics", stamped)


# ------------------------------------------------------------------------------------------- H2S-NN
_orig_h2snn_raw = d2.h2snn_raw
H2SNN_CENTRAL = {"d": 32, "wisp_max_length": 3, "wisp_walks_per_anchor": 16, "lr": 1e-3, "weight_decay": 5e-4,
                 "encoder_dropout": 0.45, "classifier_dropout": 0.45}
H2SNN_KEYS = {"model.d": ("model", "d"), "data.wisp_max_length": ("data", "wisp_max_length"),
              "data.wisp_walks_per_anchor": ("data", "wisp_walks_per_anchor"), "data.batch_size": ("data", "batch_size")}


def patched_h2snn_raw(dataset: str, task: str, out: Path):
    raw, point = _orig_h2snn_raw(dataset, task, out)
    if ARGS["point"] == "central":
        import yaml
        from scripts.run_h2snn_wandb_tuning import apply_sweep_overrides, build_sweep_config

        sweep = build_sweep_config(task_type=task, metric="macro_f1", epochs=1, dataset_name=dataset)["parameters"]
        base = yaml.safe_load((d2.REPO / "configs" / "experiments" / f"{dataset}_h2snn_v3b.yaml").read_text())
        params = {k: v for k, v in H2SNN_CENTRAL.items() if k in sweep}
        # middle of every numeric grid of the generated sweep (checks H2SNN_CENTRAL against the code)
        for k, spec in sweep.items():
            if "values" in spec and is_numeric_grid(list(spec["values"])) and k in params:
                assert params[k] == middle(list(spec["values"])), (k, params[k], spec)
        central = apply_sweep_overrides(base, params)
        for keep in ("epochs", "device", "exp_name", "output_dir", "use_wandb", "wandb_project", "wandb_group",
                     "wandb_tags"):
            central[keep] = raw[keep]
        point = {"model.d": 32, "data.wisp_max_length": 3, "data.wisp_walks_per_anchor": 16, "optim.lr": 1e-3,
                 "optim.weight_decay": 5e-4, "model.encoder_dropout": 0.45, "model.classifier_dropout": 0.45}
        if task == "graph_classification":
            from src.configs.config import TrainConfig

            point["model.graph_pooling"] = TrainConfig.from_dict(central).model.graph_pooling  # YAML or default
        if dataset in H2GB_NC:
            central["data"]["batch_size"] = 256
            point["data.batch_size"] = 256
        raw = central
    grids = dict(d2.H2SNN_GRID)
    if dataset in H2GB_NC:
        grids["data.batch_size"] = [32, 256, 1024]
    values = {k: raw[s][f] for k, (s, f) in H2SNN_KEYS.items() if k in grids and f in raw.get(s, {})}
    rule: dict[str, str] = {}
    apply_set(values, grids, rule)
    for k in rule:
        section, field = H2SNN_KEYS[k]
        raw[section][field] = point[k] = values[k]
    raw["wandb_group"] = "d6"
    raw["wandb_tags"] = ["dry-run", "d6", task]
    return raw, point


# ------------------------------------------------------------------------------------------- target counts
def count_val_targets(obj: Any) -> int | None:
    for path in (("data", "target_val_mask"), ("target_val_mask",), ("data", "val_mask"), ("val_mask",)):
        cur = obj
        try:
            for p in path:
                cur = getattr(cur, p)
            return int(cur.sum())
        except AttributeError:
            continue
    for path in (("fold", "val_loader"), ("val_loader",)):
        cur = obj
        try:
            for p in path:
                cur = getattr(cur, p)
            return len(cur.dataset)
        except AttributeError:
            continue
    return None


def install_counters() -> None:
    import wandb

    import src.baselines.bundles as bundles
    import src.baselines.minibatch as minibatch
    import src.builders as builders

    orig_components = builders.build_experiment_components

    def components(cfg):
        out = orig_components(cfg)
        EXTRA["val_targets"] = count_val_targets(out[0])
        return out

    builders.build_experiment_components = components

    orig_bundle = bundles.build_bundle

    def bundle(*a, **k):
        out = orig_bundle(*a, **k)
        EXTRA.setdefault("val_targets", count_val_targets(out))
        mark(f"bundle_built val_targets={EXTRA['val_targets']}")
        return out

    bundles.build_bundle = bundle

    orig_eval = minibatch.evaluate_split
    calls = {"n": 0}

    def evaluate_split(loader, forward_seeds, target_type, ids, *a, **k):
        phase = "val_pass" if calls["n"] == 0 else ("test_pass" if calls["n"] == 1 else None)
        calls["n"] += 1
        if calls["n"] == 1:
            EXTRA["val_targets"] = int(ids.numel())
        if phase is None:
            return orig_eval(loader, forward_seeds, target_type, ids, *a, **k)
        with d2.Phase(phase):
            return orig_eval(loader, forward_seeds, target_type, ids, *a, **k)

    minibatch.evaluate_split = evaluate_split

    import time

    orig_module_eval = torch_module_eval = __import__("torch").nn.Module.eval

    def marked_eval(self):
        if "first_eval" not in EXTRA:
            EXTRA["first_eval"] = time.perf_counter()
            mark("first_eval")
        return orig_module_eval(self)

    __import__("torch").nn.Module.eval = marked_eval

    orig_init = wandb.init

    def init(*a, **k):
        k["group"] = "d6"
        k["tags"] = ["dry-run", "d6"] + [t for t in (k.get("tags") or []) if t not in ("d2", "dry-run")]
        return orig_init(*a, **k)

    wandb.init = init


def main() -> None:
    argv = sys.argv[1:]
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--point":
            ARGS["point"] = argv[i + 1]
            i += 2
        elif argv[i] == "--set":
            ARGS["set"] = {k: parse_value(v) for k, v in (kv.split("=") for kv in argv[i + 1].split(",") if kv)}
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    if ARGS["point"] not in ("max", "central"):
        raise SystemExit(f"--point must be max or central, got {ARGS['point']}")
    EXTRA["cell"] = rest[rest.index("--cell") + 1]
    import os

    if os.environ.get("D6_ANOMALY") == "1":  # D6.5: the forward stack of the failing backward node (to stderr)
        import torch

        torch.autograd.set_detect_anomaly(True, check_nan=False)
    orig_classify = d2.classify

    def classify(exc):  # D6.5: keep the whole chained traceback (D2 keeps its last 4,000 characters)
        import traceback

        out = orig_classify(exc)
        out["traceback_full"] = "".join(traceback.format_exception(exc))
        return out

    d2.classify = classify
    d2.baseline_point = patched_baseline_point
    d2.h2snn_raw = patched_h2snn_raw
    install_counters()
    # D2's main writes the record; add the D6 fields to it by wrapping the JSON write
    import json

    orig_dumps = json.dumps

    def dumps(obj, *a, **k):
        if isinstance(obj, dict) and obj.get("cell") == EXTRA["cell"] and "status" in obj:
            obj["point_kind"] = ARGS["point"]
            obj["set"] = ARGS["set"]
            obj["val_targets"] = EXTRA.get("val_targets")
            calls, first = EXTRA.get("metric_calls", []), EXTRA.get("first_eval")
            if first is not None and len(calls) >= 1:
                obj["t_eval_to_metric1_s"] = calls[0] - first
            if len(calls) >= 2:
                obj["t_metric1_to_metric2_s"] = calls[1] - calls[0]
            obj["card_unit"] = "D6"
        return orig_dumps(obj, *a, **k)

    d2.json.dumps = dumps
    sys.argv = [sys.argv[0]] + rest
    d2.main()
    print(f"D6 {EXTRA['cell']} point={ARGS['point']} set={ARGS['set']} val_targets={EXTRA.get('val_targets')}",
          flush=True)


if __name__ == "__main__":
    main()
