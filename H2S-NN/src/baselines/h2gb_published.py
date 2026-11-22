"""Published H2GB configurations run exactly as the yaml states (card 8 C1; decisions log 2026-11-20 "Published tuned
configurations" and "Card 8 amendment 1").

A published H2GB cell (`H2GB_PUBLISHED_CELLS`) carries, in `configs/baselines/<model>/node_classification/<dataset>.yaml`,
every key of H2GB's `configs/<folder>/<folder>-<Method>.yaml` except the data, evaluation and logging keys of
`EXCLUDED_KEYS` (kept from our protocol: files, splits, metrics, B1a selection, the single whole-split test; A2), as
dotted H2GB cfg keys with the yaml line of each. At run time:

* the cfg is set as H2GB's main sets it (`configure_from_yaml`: `set_cfg`, `merge_from_file`, the published keys
  applied, `assert_cfg`), so H2GB's own model (`cfg.model.type`: `HomoGNNModel` for GCN / GAT / R-GCN,
  `HeteroGNNModel` for HAN, `GTModel` for H2G-former) is built from it unchanged;
* `resolved_run_arguments` reads the keys the shared loop uses (`RUN_ARGUMENTS`: optimizer, learning rate, weight
  decay, epochs, scheduler and its warm-up, clipping and its value, accumulation, batch size, iterations per epoch,
  evaluation period, sampler, fan-out) plus the model-block keys, and every run setting comes from that dict;
* optimizer and scheduler are H2GB's (`graphgym/optimizer.py: create_optimizer, create_scheduler`, with
  `H2GB/utils.py: new_optimizer_config, new_scheduler_config`, as `H2GB/main.py:108-110`); the step rule (accumulation,
  clipping, one scheduler step per epoch) is H2GB's `custom_train.py:106-159, 288-293`
  (`minibatch.SourceSchedule`).

Values are decoded as yacs decodes a yaml value (`ast.literal_eval` of a string, else the string), so `1e-5` is the
float H2GB trains with.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Any

import yaml

from src.baselines.protocol import NODE_CLASSIFICATION

#: Card 8 C0 + amendment 1: the published cells that run an H2GB yaml (model, dataset).
_H2GB_SIX = ("mag-year", "oag-cs", "oag-eng", "oag-chem", "h-pokec", "pdns")
H2GB_PUBLISHED_CELLS: frozenset[tuple[str, str]] = frozenset({
    *(("gcn", d) for d in _H2GB_SIX),
    *(("gat", d) for d in _H2GB_SIX if d != "h-pokec"),  # Pokec-GAT.yaml:61 says layer_type GCN: tuned (A1)
    *(("han", d) for d in _H2GB_SIX),
    *(("rgcn", d) for d in _H2GB_SIX),
    *(("h2gformer", d) for d in ("ogbn-mag",) + _H2GB_SIX),
})

#: yaml keys not transcribed (prefix match on the dotted key), with the reason: they are data, evaluation or logging,
#: which stay our protocol (card 8 C1 "Kept from our protocol"; amendment 1 A2).
EXCLUDED_KEYS: dict[str, str] = {
    "out_dir": "logging",
    "metric_best": "selection is ours (B1a validation subset, macro-F1)",
    "seed": "seeds 0-4 are ours",
    "wandb": "logging",
    "num_threads": "hardware",
    "num_workers": "hardware",
    "dataset.dir": "data files are ours",
    "dataset.format": "data files are ours",
    "dataset.name": "data files are ours",
    "dataset.task": "task set by our bundle",
    "dataset.task_type": "task set by our bundle",
    "dataset.task_entity": "target type set by our loader",
    "dataset.transductive": "split is ours",
    "train.mode": "the shared H2GB NC loop (minibatch.train_minibatch_node_classification)",
    "train.ckpt_period": "logging",
    "train.tqdm": "logging",
    "train.persistent_workers": "hardware",
    "train.pin_memory": "hardware",
    "val": "evaluation is ours: every target of the split once (MB3) and the B1a subset",
    "model.edge_decoding": "link-task key, unused on node classification",
    "model.graph_pooling": "graph-task key, unused on node classification",
}

#: The keys the shared loop reads from the resolved arguments (`run_settings`); the model reads its own blocks
#: (`MODEL_SECTIONS`) from H2GB's cfg.
RUN_ARGUMENTS: tuple[str, ...] = (
    "optim.optimizer", "optim.base_lr", "optim.weight_decay", "optim.max_epoch",
    "optim.scheduler", "optim.num_warmup_epochs",
    "optim.clip_grad_norm", "optim.clip_grad_norm_value", "optim.batch_accumulation",
    "train.batch_size", "train.iter_per_epoch", "train.eval_period", "train.sampler",
    "train.neighbor_sizes", "train.neighbor_sizes_dict",
)
#: cfg sections that configure H2GB's network (read by the network itself).
MODEL_SECTIONS: tuple[str, ...] = ("gnn", "gt", "model", "dataset", "posenc_Hetero_Metapath", "posenc_Hetero_Label")


def is_excluded(key: str) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in EXCLUDED_KEYS)


def decode(value: Any) -> Any:
    """A yaml value as yacs decodes it (`yacs.config._decode_cfg_value`)."""
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, dotted + "."))
        else:
            out[dotted] = decode(value)
    return out


def yaml_published_values(model: str, dataset: str) -> dict[str, Any]:
    """Every transcribed key of H2GB's yaml of `model` on `dataset` (dotted, decoded), in file order."""
    from src.baselines.minibatch import h2gb_config_file

    raw = yaml.safe_load(h2gb_config_file(model, dataset).read_text(encoding="utf-8"))
    return {key: value for key, value in flatten(raw).items() if not is_excluded(key)}


def yaml_key_lines(path: Path) -> dict[str, int]:
    """1-based line of every `section.key` (and top-level `key`) of an H2GB yaml (two levels)."""
    lines: dict[str, int] = {}
    section = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            key = line.split(":", 1)[0].strip()
            section = key
            lines[key] = number
        elif line.startswith("  ") and not line.startswith("   "):
            lines[f"{section}.{line.split(':', 1)[0].strip()}"] = number
    return lines


def published_overrides(values: dict[str, Any]) -> dict[str, Any]:
    """The dotted H2GB cfg keys of a published parameter set (the keys `configure_from_yaml` applies)."""
    return {key: value for key, value in values.items() if "." in key and not is_excluded(key)}


def is_h2gb_published(model: str, task: str, dataset: str, source: str) -> bool:
    return source == "published" and task == NODE_CLASSIFICATION and (model, dataset) in H2GB_PUBLISHED_CELLS


def _cfg_value(cfg: Any, key: str) -> Any:
    node = cfg
    for part in key.split("."):
        node = getattr(node, part)
    return node


def resolved_run_arguments(cfg: Any, values: dict[str, Any]) -> dict[str, Any]:
    """The arguments the run uses, read from H2GB's cfg after `configure_from_yaml`: `RUN_ARGUMENTS` (the loop) and
    the published model-block keys (`MODEL_SECTIONS`, read by H2GB's network)."""
    out = {key: _cfg_value(cfg, key) for key in RUN_ARGUMENTS}
    for key in values:
        if key.split(".", 1)[0] in MODEL_SECTIONS:
            out[key] = _cfg_value(cfg, key)
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class H2GBRunSettings:
    """What the shared loop takes from the resolved arguments."""

    batch_size: int
    iter_per_epoch: int
    epochs: int
    eval_period: int
    accumulation: int
    clip_value: float | None


def run_settings(args: dict[str, Any]) -> H2GBRunSettings:
    if args["optim.scheduler"] in ("reduce_on_plateau", "plateau"):
        raise ValueError("a plateau scheduler steps on the validation loss of H2GB's evaluation; not supported.")
    return H2GBRunSettings(
        batch_size=int(args["train.batch_size"]), iter_per_epoch=int(args["train.iter_per_epoch"]),
        epochs=int(args["optim.max_epoch"]), eval_period=int(args["train.eval_period"]),
        accumulation=int(args["optim.batch_accumulation"]),
        clip_value=float(args["optim.clip_grad_norm_value"]) if args["optim.clip_grad_norm"] else None)


def published_plan(dataset: str, settings: H2GBRunSettings) -> Any:
    """The batch plan of the yaml: its batch size and iterations per epoch must be one epoch of the shared plan
    (8,192 targets), its epochs and evaluation period replace the defaults."""
    from src.baselines.minibatch import minibatch_plan

    plan = minibatch_plan(dataset, settings.batch_size)
    if settings.iter_per_epoch != plan.steps_per_epoch:
        raise ValueError(f"{dataset}: yaml iter_per_epoch {settings.iter_per_epoch} x batch {settings.batch_size} is "
                         f"not one epoch of {plan.epoch_targets} targets.")
    # card 16 part 2: a published cell keeps H2GB's rule, no early stopping
    return dataclasses.replace(plan, epochs=settings.epochs, eval_period=settings.eval_period, patience=None)


def h2gb_optimizer_and_schedule(model: Any, cfg: Any, args: dict[str, Any], settings: H2GBRunSettings
                                ) -> tuple[Any, Any]:
    """H2GB's optimizer and scheduler (`H2GB/main.py:108-110`) from the resolved arguments, and the step rule.
    Call inside `pushd_sys_path("h2gb")`."""
    from H2GB.graphgym.optimizer import OptimizerConfig, create_optimizer, create_scheduler  # type: ignore
    from H2GB.utils import new_scheduler_config  # type: ignore[import-not-found]

    from src.baselines.minibatch import SourceSchedule

    optimizer = create_optimizer(model.named_parameters(), OptimizerConfig(
        optimizer=args["optim.optimizer"], base_lr=args["optim.base_lr"], weight_decay=args["optim.weight_decay"],
        momentum=cfg.optim.momentum))
    scheduler_cfg = dataclasses.replace(new_scheduler_config(cfg), scheduler=args["optim.scheduler"],
                                        max_epoch=args["optim.max_epoch"],
                                        num_warmup_epochs=args["optim.num_warmup_epochs"])
    scheduler = create_scheduler(optimizer, scheduler_cfg)
    return optimizer, SourceSchedule(scheduler=scheduler, clip_value=settings.clip_value,
                                     accumulation=settings.accumulation)


def build_h2gb_network(data: Any, dataset: str, num_classes: int, cfg: Any) -> Any:
    """H2GB's network named by `cfg.model.type` (`HomoGNNModel` or `HeteroGNNModel`) under the current cfg."""
    from src.baselines.adapters._h2gb_network import DatasetShim, build_hetero_gnn

    if cfg.model.type == "HeteroGNNModel":
        return build_hetero_gnn(data, dataset, num_classes)
    if cfg.model.type == "HomoGNNModel":
        from H2GB.network.homo_gnn_model import HomoGNNModel  # type: ignore[import-not-found]

        return HomoGNNModel(cfg.share["dim_in"], num_classes, DatasetShim(data, dataset))
    raise ValueError(f"no published path for H2GB model type {cfg.model.type!r}.")


def fanout_from_args(args: dict[str, Any], node_types: tuple[str, ...]) -> dict[str, list[int]]:
    """HGTLoader's `num_samples` from the resolved fan-out keys, H2GB's own expression (`custom_sampler.py:323-326`)."""
    if len(args["train.neighbor_sizes_dict"]) > 0:
        sizes = eval(args["train.neighbor_sizes_dict"])  # noqa: S307 - H2GB's own line (custom_sampler.py:324)
    else:
        sizes = {key: args["train.neighbor_sizes"] for key in node_types}
    return {str(key): [int(v) for v in value] for key, value in sizes.items()}


def resolve_published(model_name: str, dataset: str, data: Any, target: str, device: Any,
                      values: dict[str, Any], extra: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """(cfg, resolved arguments) of a published H2GB cell: H2GB's cfg set from the yaml and the published keys
    (`extra`: cfg keys that are not hyperparameters, e.g. H2G-former's `share.dim_out`). Call inside
    `pushd_sys_path("h2gb")`; the cfg stays set (configure it last before building the network)."""
    from src.baselines.adapters._h2gb_network import configure_from_yaml

    cfg = configure_from_yaml(model_name, dataset, data, target, device, {**published_overrides(values), **(extra or {})})
    return cfg, dict(resolved_run_arguments(cfg, values))


def fit_h2gb_published(bundle: Any, params: Any, *, model_name: str, seed: int, device: Any) -> dict[str, float]:
    """A published H2GB cell with a `HomoGNNModel` or `HeteroGNNModel` (GCN, GAT, R-GCN, HAN): H2GB's network and
    cfg from the yaml, on H2GB's `hgt` batches with the yaml's fan-out, the shared loop with H2GB's optimizer,
    scheduler, clipping and accumulation; B1a selection and the whole-split test as every H2GB NC cell."""
    import torch

    from src.baselines.adapters._h2gb_network import seed_predictions
    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.minibatch import (eval_loader, h2gb_graph, hgt_loader, train_minibatch_node_classification,
                                         val_subset_loader)
    from src.baselines.runtime import pushd_sys_path
    from src.data.h2gb_nc import ALL_H2GB_NC_SPECS
    from src.utils.misc import seed_everything

    values = dict(params.values)
    seed_everything(seed, device)
    data = h2gb_graph(bundle)
    target = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
    labels = data[target].y.long()
    num_classes = int(labels.max()) + 1
    ensure_pyg_submodules()
    with pushd_sys_path("h2gb"):
        _, args = resolve_published(model_name, bundle.dataset, data, target, device, values)
    if args["model.loss_fun"] != "cross_entropy":
        raise ValueError(f"loss {args['model.loss_fun']!r}: the shared loop trains with cross-entropy.")
    settings = run_settings(args)
    plan = published_plan(bundle.dataset, settings)
    fanout = fanout_from_args(args, tuple(data.node_types))
    loaders = (hgt_loader(data, fanout, target, "train", plan.batch_size, shuffle=True, steps=plan.steps_per_epoch),
               eval_loader(data, fanout, target, "val", plan.batch_size),
               eval_loader(data, fanout, target, "test", plan.batch_size))
    subset_loader, subset_ids = val_subset_loader(data, fanout, target, bundle.dataset, plan.batch_size,
                                                  run_seed=bundle.metadata.get("seed"))
    with pushd_sys_path("h2gb"):  # the cfg is configured last: H2GB's model reads it in its forward
        cfg, args_again = resolve_published(model_name, bundle.dataset, data, target, device, values)
        if args_again != args:
            raise RuntimeError("H2GB's cfg changed between resolution and model construction.")
        torch.manual_seed(seed)
        model = build_h2gb_network(data, bundle.dataset, num_classes, cfg).to(device)
        optimizer, schedule = h2gb_optimizer_and_schedule(model, cfg, args, settings)
    record = train_minibatch_node_classification(
        model=model, forward_seeds=lambda batch: seed_predictions(model, batch.to(device), target),
        train_loader=loaders[0], val_loader=loaders[1], test_loader=loaders[2],
        val_ids=data[target].val_mask.nonzero().view(-1), test_ids=data[target].test_mask.nonzero().view(-1),
        val_subset_loader=subset_loader, val_subset_ids=subset_ids, target_type=target, optimizer=optimizer,
        criterion=torch.nn.functional.cross_entropy, plan=plan, report=getattr(params, "report", None),
        labels=labels, schedule=schedule)
    record["params_source"] = "published"
    record["h2gb_model_type"] = str(cfg.model.type)
    record["batch_size"] = float(plan.batch_size)
    record["sampler_depth"] = float(max(len(v) for v in fanout.values()))
    return record


__all__ = [
    "EXCLUDED_KEYS", "H2GBRunSettings", "H2GB_PUBLISHED_CELLS", "MODEL_SECTIONS", "RUN_ARGUMENTS",
    "build_h2gb_network", "decode", "fanout_from_args", "fit_h2gb_published", "flatten", "h2gb_optimizer_and_schedule",
    "is_excluded", "is_h2gb_published", "published_overrides", "published_plan", "resolve_published",
    "resolved_run_arguments",
    "run_settings", "yaml_key_lines", "yaml_published_values",
]
