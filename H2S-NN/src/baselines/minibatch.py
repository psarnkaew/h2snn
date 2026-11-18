"""Mini-batch infrastructure shared by the baseline adapters on the H2GB node-classification datasets
(card `docs/H2S-NN_baselines_h2gb_card.md` C1; spec v1.22 §6; `docs/minibatch_design_v3B.md` MB2, MB3).

* **Batch plan** (`minibatch_plan`): the batch size is one of the budget table's range ({32, 256, 1024},
  `src/configs/config.py: BATCH_BUDGETS`); one epoch = 8,192 training targets = 8,192 / B optimizer steps; the loss
  is the mean over the B targets of a step; 500 epochs, no early stopping, the best validation epoch is kept.
* **Fan-out** (`h2gb_sample_sizes`): read from H2GB's own YAML of that method and dataset, merged into H2GB's cfg as
  H2GB's `main.py` does (`set_cfg`, `merge_from_file`) and turned into HGTLoader's `num_samples` with H2GB's own
  expression (`H2GB/sampler/custom_sampler.py:323-326`); never retyped.
* **Sampler** (`hgt_loader`): H2GB's `get_HGTloader` (`custom_sampler.py:287-347`, PyG `HGTLoader` inside H2GB's
  `LoaderWrapper`), called with H2GB's cfg set from the arguments. The number of HGTLoader iterations is the length of
  the per-type lists passed (`torch_geometric/sampler/hgt_sampler.py:48`). Which depth an adapter passes is open
  (C0 stop S1, `docs/baselines_h2gb_evidence/c0_preconditions.md`); this module takes `sample_sizes` as given and
  `repeat_fanout` builds the card's rule (one per-hop value repeated to a depth) where a single value exists.
* **Evaluation** (`evaluate_split`): every target of the split exactly once, in a fixed order (HGTLoader without
  shuffling over the split mask, `ceil(n / B)` batches, the last partial batch included); the metric over the whole
  split (MB3; H2GB itself evaluates at most `val.iter_per_epoch` x B shuffled targets, `custom_sampler.py:20-55`, a
  stated difference). Every `eval_period` epochs by H2GB's rule (`H2GB/graphgym/utils/epoch.py:9-12`: epoch 0,
  every `eval_period`-th, and the last).
* **B1a** (card 1f, author decision; `src/evaluation/validation_subset.py`): during training, an evaluation epoch
  scores the fixed validation subset only (the same ids for every method, drawn once from a fixed seed) and runs no
  test pass; the best subset macro-F1 epoch's weights are kept (CPU copy), restored after the last epoch, and scored
  once on the whole validation and test splits. The record carries the whole-split metrics plus
  `val_subset_macro_f1`, `val_subset_size` and `val_subset_fingerprint`.
* AdamW, no scheduler, no clipping (M9) stay with the adapters.

`convert_batch` (`custom_sampler.py:58-63`, H2GB's transform) casts every batch's `x` to float32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from src.baselines.budget import batch_plan, default_budget
from src.baselines.protocol import NODE_CLASSIFICATION
from src.configs.config import default_eval_period

#: H2GB's config folder per dataset key (`vendor/H2GB/configs/<folder>/<folder>-<Method>.yaml`).
H2GB_CONFIG_FOLDERS: dict[str, str] = {
    "ogbn-mag": "ogbn-mag", "mag-year": "mag-year", "oag-cs": "oag-cs", "oag-eng": "oag-eng",
    "oag-chem": "oag-chem", "ieee-cis-g": "IEEE-CIS", "h-pokec": "Pokec", "pdns": "PDNS",
}
#: Class-A methods (card scheme table) and the file name H2GB gives each one.
H2GB_CONFIG_METHODS: dict[str, str] = {"gcn": "GCN", "gat": "GAT", "han": "HAN", "h2gformer": "H2Gformer"}
#: D1 (author): a class-C method without a node-wise forward uses the dataset's H2GB GCN fan-out.
CLASS_C_REFERENCE_METHOD = "gcn"


@dataclass(frozen=True, slots=True)
class MinibatchPlan:
    """How one baseline run on an H2GB NC dataset is batched (the budget table's row, one B chosen)."""

    dataset: str
    batch_size: int
    epoch_targets: int
    steps_per_epoch: int
    epochs: int
    eval_period: int


def minibatch_plan(dataset: str, batch_size: int) -> MinibatchPlan:
    """The plan of `dataset` at batch size `batch_size` (one of the budget table's range)."""
    plan = batch_plan(NODE_CLASSIFICATION, dataset)
    if plan.mode != "minibatch" or plan.batch_sizes is None or plan.epoch_targets is None:
        raise ValueError(f"{dataset!r} is not a mini-batch node-classification dataset ({plan.dataset_class}).")
    if batch_size not in plan.batch_sizes:
        raise ValueError(f"batch size {batch_size} is not in the shared range {plan.batch_sizes}.")
    if plan.epoch_targets % batch_size:
        raise ValueError(f"{plan.epoch_targets} training targets per epoch is not a multiple of B = {batch_size}.")
    return MinibatchPlan(dataset=dataset, batch_size=batch_size, epoch_targets=plan.epoch_targets,
                         steps_per_epoch=plan.epoch_targets // batch_size,
                         epochs=default_budget(NODE_CLASSIFICATION, dataset).epochs,
                         eval_period=default_eval_period(dataset))


def is_eval_epoch(epoch: int, plan: MinibatchPlan) -> bool:
    """H2GB's `is_eval_epoch` (`H2GB/graphgym/utils/epoch.py:9-12`) with start epoch 0."""
    return (epoch + 1) % plan.eval_period == 0 or epoch == 0 or epoch + 1 == plan.epochs


def h2gb_config_file(method: str, dataset: str) -> Path:
    """H2GB's YAML of `method` (a class-A registry name) on `dataset`."""
    from src.baselines.runtime.vendor_paths import vendor_path

    if method not in H2GB_CONFIG_METHODS:
        raise KeyError(f"{method!r} has no H2GB config (class A: {sorted(H2GB_CONFIG_METHODS)}).")
    folder = H2GB_CONFIG_FOLDERS[dataset]
    path = vendor_path("h2gb") / "configs" / folder / f"{folder}-{H2GB_CONFIG_METHODS[method]}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"H2GB has no config {path}.")
    return path


def h2gb_sample_sizes(method: str, dataset: str, node_types: Iterable[str]) -> dict[str, list[int]]:
    """HGTLoader's `num_samples` as H2GB computes it from its YAML (`custom_sampler.py:323-326`)."""
    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.runtime.vendor_paths import pushd_sys_path

    path = h2gb_config_file(method, dataset)
    ensure_pyg_submodules()
    with pushd_sys_path("h2gb"):
        import H2GB  # noqa: F401  - registers H2GB's config groups
        from H2GB.graphgym.config import cfg, set_cfg  # type: ignore[import-not-found]

        set_cfg(cfg)
        cfg.merge_from_file(str(path))
        if len(cfg.train.neighbor_sizes_dict) > 0:
            sizes = eval(cfg.train.neighbor_sizes_dict)  # noqa: S307 - H2GB's own line (custom_sampler.py:324)
        else:
            sizes = {key: cfg.train.neighbor_sizes for key in node_types}
        set_cfg(cfg)
    return {str(key): [int(v) for v in value] for key, value in sizes.items()}


def repeat_fanout(sample_sizes: dict[str, list[int]], depth: int) -> dict[str, list[int]]:
    """The card's class-A rule: one per-hop fan-out per node type, repeated `depth` times. Refused when H2GB's list
    has no single per-hop value (H-Pokec's user list; C0 stop S1)."""
    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}.")
    out: dict[str, list[int]] = {}
    for key, values in sample_sizes.items():
        if len(set(values)) != 1:
            raise ValueError(f"H2GB's fan-out of {key!r} is {values}: no single per-hop value to repeat "
                             "(C0 stop S1, author decision).")
        out[key] = [values[0]] * depth
    return out


class _H2GBDatasetShim:
    """What `get_HGTloader` reads from H2GB's dataset object: `dataset[0]` and `dataset.data`."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getitem__(self, index: int) -> Any:
        if index != 0:
            raise IndexError(index)
        return self.data


def hgt_loader(data: Any, sample_sizes: dict[str, list[int]], target_type: str, split: str, batch_size: int, *,
               shuffle: bool, steps: int, num_workers: int = 0) -> Any:
    """H2GB's `get_HGTloader` on `data` (seeds: `data[target_type][f'{split}_mask']`), with `steps` batches per
    pass of its `LoaderWrapper` (training: continued across passes; validation/test: restarted each pass)."""
    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.runtime.vendor_paths import pushd_sys_path

    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}.")
    ensure_pyg_submodules()
    with pushd_sys_path("h2gb"):
        import H2GB  # noqa: F401
        from H2GB.graphgym.config import cfg, set_cfg  # type: ignore[import-not-found]
        from H2GB.sampler.custom_sampler import get_HGTloader  # type: ignore[import-not-found]

        set_cfg(cfg)
        cfg.train.neighbor_sizes_dict = repr({key: list(value) for key, value in sample_sizes.items()})
        cfg.dataset.task_entity = target_type
        cfg.num_workers = int(num_workers)
        cfg.train.persistent_workers = num_workers > 0
        cfg.train.pin_memory = False
        cfg.train.iter_per_epoch = int(steps)
        cfg.val.iter_per_epoch = int(steps)
        loader = get_HGTloader(_H2GBDatasetShim(data), batch_size, shuffle=shuffle, split=split)
        set_cfg(cfg)
    return loader


def eval_steps(num_targets: int, batch_size: int) -> int:
    """Batches that cover a split of `num_targets` targets once, the last partial batch included."""
    return math.ceil(num_targets / batch_size)


def eval_loader(data: Any, sample_sizes: dict[str, list[int]], target_type: str, split: str, batch_size: int,
                num_workers: int = 0) -> Any:
    """Every target of `split` once, in the mask's order (no shuffling)."""
    count = int(data[target_type][f"{split}_mask"].sum())
    return hgt_loader(data, sample_sizes, target_type, split, batch_size, shuffle=False,
                      steps=eval_steps(count, batch_size), num_workers=num_workers)


def val_subset_loader(data: Any, sample_sizes: dict[str, list[int]], target_type: str, dataset: str,
                      batch_size: int, num_workers: int = 0, *, run_seed: int | None = None
                      ) -> tuple[Any, torch.Tensor]:
    """(loader, ids) over the B1a validation subset of `dataset`: H2GB's loader, called unchanged, on a shallow copy
    of `data` whose target `val_mask` holds the subset ids only (the caller's graph is not modified). `run_seed`: the
    seed that drew `data`'s split (PDNS, card 1f U5.3; ignored for the other datasets)."""
    from copy import copy

    from src.evaluation.validation_subset import validation_subset

    mask = data[target_type].val_mask
    ids = validation_subset(dataset, mask.nonzero().view(-1), run_seed)
    subset_mask = torch.zeros_like(mask)
    subset_mask[ids.to(mask.device)] = True
    view = copy(data)
    view[target_type].val_mask = subset_mask
    return eval_loader(view, sample_sizes, target_type, "val", batch_size, num_workers=num_workers), ids


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A CPU copy of `model`'s state_dict (the best epoch's weights, B1a)."""
    return {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}


def _subset_fields(ids: torch.Tensor, macro_f1: float) -> dict[str, Any]:
    from src.evaluation.validation_subset import subset_fingerprint

    return {"val_subset_macro_f1": float(macro_f1), "val_subset_size": float(ids.numel()),
            "val_subset_fingerprint": subset_fingerprint(ids)}


def seed_rows(batch: Any, target_type: str, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(global ids, labels) of a batch's seed targets: the first `batch_size` rows of the target type. With `labels`
    (all target nodes' true labels) they are read by global id, not from the batch (an adapter whose input channel
    rewrites `y` in the graph, e.g. H2G-former's training-labels-only channel, part 2 E1)."""
    size = int(batch[target_type].batch_size)
    ids = batch[target_type].n_id[:size]
    return ids, (batch[target_type].y[:size] if labels is None else labels[ids.to(labels.device)])


def evaluate_split(loader: Any, forward_seeds: Callable[[Any], torch.Tensor], target_type: str,
                   expected: torch.Tensor, labels: torch.Tensor | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(ids, logits, labels) of every target of the split, in the loader's order; refuses a pass that does not
    cover `expected` (global ids) exactly once."""
    ids, logits, label_rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_ids, batch_labels = seed_rows(batch, target_type, labels)
            out = forward_seeds(batch)
            if out.size(0) != batch_ids.numel():
                raise RuntimeError(f"forward returned {out.size(0)} rows for {batch_ids.numel()} seed targets.")
            ids.append(batch_ids.cpu())
            logits.append(out)
            label_rows.append(batch_labels)
    all_ids = torch.cat(ids) if ids else torch.empty(0, dtype=torch.long)
    if all_ids.numel() != expected.numel() or not torch.equal(torch.sort(all_ids).values,
                                                              torch.sort(expected.cpu().long()).values):
        raise RuntimeError(f"evaluation covered {all_ids.numel()} targets ({torch.unique(all_ids).numel()} "
                           f"distinct); the split has {expected.numel()}.")
    return all_ids, torch.cat(logits), torch.cat(label_rows)


def train_minibatch_node_classification(
    *,
    model: torch.nn.Module,
    forward_seeds: Callable[[Any], torch.Tensor],
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    val_ids: torch.Tensor,
    test_ids: torch.Tensor,
    val_subset_loader: Any,
    val_subset_ids: torch.Tensor,
    target_type: str,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    plan: MinibatchPlan,
    report: Any = None,
    labels: torch.Tensor | None = None,
) -> dict[str, float]:
    """The shared H2GB NC loop: `plan.steps_per_epoch` steps per epoch (loss = mean over the step's B targets); at
    H2GB's eval epochs the B1a validation subset only (`val_subset_loader` over `val_subset_ids`, no test pass); the
    best subset macro-F1 epoch's weights restored after the last epoch and scored once on the whole validation and
    test splits (`val_loader`, `test_loader`); no early stop."""
    from src.baselines.metrics import classification_metrics, node_classification_record
    from src.baselines.progress import report_epoch
    from src.models.anchored_solve import check_finite_loss

    if len(train_loader) != plan.steps_per_epoch:
        raise ValueError(f"training loader has {len(train_loader)} steps per pass; the plan needs "
                         f"{plan.steps_per_epoch}.")
    best_monitor, best_epoch = float("-inf"), -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(plan.epochs):
        model.train()
        for batch in train_loader:
            _, batch_labels = seed_rows(batch, target_type, labels)
            optimizer.zero_grad()
            logits = forward_seeds(batch)
            loss = criterion(logits, batch_labels.to(logits.device))
            check_finite_loss(loss, epoch)
            loss.backward()
            optimizer.step()
        if not is_eval_epoch(epoch, plan):
            continue
        model.eval()
        _, sub_logits, sub_labels = evaluate_split(val_subset_loader, forward_seeds, target_type, val_subset_ids,
                                                   labels)
        sub_metrics = classification_metrics(sub_logits, sub_labels.to(sub_logits.device), is_multilabel=False)
        if sub_metrics["macro_f1"] > best_monitor:
            best_monitor, best_epoch, best_state = sub_metrics["macro_f1"], epoch, _cpu_state(model)
        report_epoch(report, epoch=epoch, monitor=sub_metrics["macro_f1"], running_best=best_monitor,
                     metrics=sub_metrics)
    if best_state is None:
        raise RuntimeError("mini-batch training produced no evaluated epoch.")
    model.load_state_dict(best_state)
    model.eval()
    _, val_logits, val_labels = evaluate_split(val_loader, forward_seeds, target_type, val_ids, labels)
    _, test_logits, test_labels = evaluate_split(test_loader, forward_seeds, target_type, test_ids, labels)
    val_metrics = classification_metrics(val_logits, val_labels.to(val_logits.device), is_multilabel=False)
    test_metrics = classification_metrics(test_logits, test_labels.to(test_logits.device), is_multilabel=False)
    record: dict[str, Any] = node_classification_record(val_metrics=val_metrics, test_metrics=test_metrics,
                                                        best_epoch=best_epoch)
    record.update(_subset_fields(val_subset_ids, best_monitor))
    return record


def hgb_homogeneous_batch(batch: Any, node_types: tuple[str, ...]) -> tuple[list[torch.Tensor], torch.Tensor,
                                                                          dict[str, tuple[int, int]]]:
    """HGB's homogeneous reduction (`bundles._symmetrize_with_self_loops`: sum of relations, symmetrised, self-loops
    removed then added) of a (sampled) heterogeneous graph, nodes concatenated in `node_types` order: (per-type
    features, edge_index, node slices)."""
    from src.baselines.bundles import _symmetrize_with_self_loops

    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for node_type in node_types:
        count = int(batch[node_type].num_nodes)
        offsets[node_type] = (start, start + count)
        start += count
    edges = []
    for src, rel, dst in batch.edge_types:
        index = batch[src, rel, dst].edge_index
        edges.append(torch.stack([index[0] + offsets[src][0], index[1] + offsets[dst][0]]))
    edge_index = torch.cat(edges, dim=1) if edges else torch.empty(2, 0, dtype=torch.long)
    return [batch[t].x for t in node_types], _symmetrize_with_self_loops(edge_index, start), offsets


def h2gb_graph(bundle: Any) -> Any:
    """H2GB's own graph of a node bundle (`dataset[0]` through the B2/C2 loader, the bundle's data root and run seed):
    the object H2GB's samplers read, for adapters that consume the heterogeneous graph rather than the flattened one."""
    from src.baselines.bundles import bundle_data_root
    from src.data.h2gb_nc import load_h2gb_nc

    return load_h2gb_nc(bundle.dataset, bundle_data_root(bundle), seed=bundle.metadata.get("seed"))


def index_batches(index: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    """Contiguous batches of `index` in its order, the last partial batch included (evaluation of index-batched
    methods: SeHGNN `ogbn/main.py:296-306`, ChebNetII `ogb/large_train.py:75-83`)."""
    return [index[start:start + batch_size] for start in range(0, index.numel(), batch_size)]


def train_index_minibatch(
    *,
    model: torch.nn.Module,
    forward_rows: Callable[[torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    plan: MinibatchPlan,
    base_seed: int,
    report: Any = None,
    run_seed: int | None = None,
) -> dict[str, float]:
    """The shared H2GB NC loop for methods whose source batches target rows of precomputed inputs (class B): each
    epoch takes `plan.steps_per_epoch` batches of B training targets from a seeded order continued across epochs
    (`TrainingOrder`, as H2S-NN's trainer), loss = mean over the batch; evaluation on whole splits in contiguous
    batches by H2GB's eval-epoch rule on the B1a validation subset of `plan.dataset` only (no test pass); the best subset
    macro-F1 epoch's weights restored after the last epoch and scored once on the whole validation and test splits;
    no early stopping. `run_seed`: the seed that drew the split (PDNS, card 1f U5.3)."""
    from src.baselines.metrics import classification_metrics, node_classification_record
    from src.baselines.progress import report_epoch
    from src.models.anchored_solve import check_finite_loss
    from src.trainers.minibatch_trainer import TrainingOrder

    from src.evaluation.validation_subset import validation_subset

    order = TrainingOrder(train_idx.cpu(), base_seed)
    subset_idx = validation_subset(plan.dataset, val_idx, run_seed).to(val_idx.device)

    def evaluate(index: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            logits = torch.cat([forward_rows(batch) for batch in index_batches(index, plan.batch_size)])
        return classification_metrics(logits, labels[index.to(labels.device)], is_multilabel=False)

    best_monitor, best_epoch = float("-inf"), -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(plan.epochs):
        model.train()
        for _ in range(plan.steps_per_epoch):
            batch = order.next(plan.batch_size)
            optimizer.zero_grad()
            logits = forward_rows(batch)
            loss = criterion(logits, labels[batch.to(labels.device)])
            check_finite_loss(loss, epoch)
            loss.backward()
            optimizer.step()
        if not is_eval_epoch(epoch, plan):
            continue
        model.eval()
        sub_metrics = evaluate(subset_idx)
        if sub_metrics["macro_f1"] > best_monitor:
            best_monitor, best_epoch, best_state = sub_metrics["macro_f1"], epoch, _cpu_state(model)
        report_epoch(report, epoch=epoch, monitor=sub_metrics["macro_f1"], running_best=best_monitor,
                     metrics=sub_metrics)
    if best_state is None:
        raise RuntimeError("mini-batch training produced no evaluated epoch.")
    model.load_state_dict(best_state)
    model.eval()
    record: dict[str, Any] = node_classification_record(val_metrics=evaluate(val_idx), test_metrics=evaluate(test_idx),
                                                        best_epoch=best_epoch)
    record.update(_subset_fields(subset_idx, best_monitor))
    return record



def batch_type_inputs(batch: Any, node_types: tuple[str, ...], num_nodes: dict[str, int]) -> list[torch.Tensor]:
    """Per-type input rows of a sampled batch in `node_types` order: the type's `x`, or the one-hot rows of its sampled
    node ids for a featureless type (HGB's identity encoding of attribute-less types, `bundles.build_homogeneous_view`;
    H2GB's rule for such types is a learned per-node vector, the same thing under a linear input layer)."""
    rows = []
    for node_type in node_types:
        store = batch[node_type]
        x = store.get("x")
        if x is None:
            ids = store.n_id
            x = torch.zeros(ids.numel(), num_nodes[node_type])
            x[torch.arange(ids.numel()), ids] = 1.0
        rows.append(x)
    return rows


def h2gb_input_dims(data: Any) -> tuple[tuple[str, ...], dict[str, int], list[int]]:
    """(node types in H2GB's order, node counts, input width per type: `x`'s width or the node count)."""
    node_types = tuple(data.node_types)
    counts = {t: int(data[t].num_nodes) for t in node_types}
    dims = [int(data[t].x.size(1)) if data[t].get("x") is not None else counts[t] for t in node_types]
    return node_types, counts, dims



def fit_homogeneous_on_hgt(bundle: Any, params: Any, *, fanout_method: str, seed: int, device: torch.device,
                           build_model: Callable[[list[int], int], torch.nn.Module],
                           forward: Callable[[torch.nn.Module, list[torch.Tensor], torch.Tensor, int], torch.Tensor],
                           criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
                           ) -> dict[str, float]:
    """A model that consumes HGB's homogeneous reduction (GCN, GAT; NSD, CSNN) trained on H2GB's `hgt` batches with
    the YAML fan-out of `fanout_method` on the dataset, unchanged (author decision S1, part 2): each batch's subgraph is
    reduced as HGB reduces a graph (`hgb_homogeneous_batch`), `forward(model, per-type inputs, edge_index, num_nodes)`
    returns rows for every node, the seed targets' rows are kept."""
    from src.data.h2gb_nc import ALL_H2GB_NC_SPECS
    from src.utils.misc import seed_everything

    seed_everything(seed, device)
    data = h2gb_graph(bundle)
    target = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
    node_types, counts, dims = h2gb_input_dims(data)
    fanout = h2gb_sample_sizes(fanout_method, bundle.dataset, node_types)
    plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
    labels = data[target].y.long()
    num_classes = int(labels.max()) + 1
    model = build_model(dims, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 5e-4)),
                                  weight_decay=float(params.get("weight_decay", 1e-4)))

    def forward_seeds(batch: Any) -> torch.Tensor:
        inputs = [x.to(device) for x in batch_type_inputs(batch, node_types, counts)]
        _, edge_index, offsets = hgb_homogeneous_batch(batch, node_types)
        out = forward(model, inputs, edge_index.to(device), sum(x.size(0) for x in inputs))
        start = offsets[target][0]
        return out[start:start + int(batch[target].batch_size)]

    subset_loader, subset_ids = val_subset_loader(data, fanout, target, bundle.dataset, plan.batch_size,
                                                  run_seed=bundle.metadata.get("seed"))
    record = train_minibatch_node_classification(
        model=model, forward_seeds=forward_seeds,
        train_loader=hgt_loader(data, fanout, target, "train", plan.batch_size, shuffle=True,
                                steps=plan.steps_per_epoch),
        val_subset_loader=subset_loader, val_subset_ids=subset_ids,
        val_loader=eval_loader(data, fanout, target, "val", plan.batch_size),
        test_loader=eval_loader(data, fanout, target, "test", plan.batch_size),
        val_ids=data[target].val_mask.nonzero().view(-1), test_ids=data[target].test_mask.nonzero().view(-1),
        target_type=target, optimizer=optimizer, criterion=criterion or torch.nn.functional.cross_entropy,
        plan=plan, report=getattr(params, "report", None), labels=labels)
    record["batch_size"] = float(plan.batch_size)
    record["sampler_depth"] = float(max(len(v) for v in fanout.values()))
    return record


__all__ = [
    "CLASS_C_REFERENCE_METHOD", "batch_type_inputs", "h2gb_input_dims", "H2GB_CONFIG_FOLDERS", "H2GB_CONFIG_METHODS", "MinibatchPlan", "eval_loader",
    "eval_steps", "evaluate_split", "fit_homogeneous_on_hgt", "h2gb_config_file", "h2gb_graph", "index_batches", "h2gb_sample_sizes", "hgb_homogeneous_batch", "hgt_loader",
    "is_eval_epoch", "minibatch_plan", "repeat_fanout", "seed_rows", "train_index_minibatch",
    "train_minibatch_node_classification", "val_subset_loader",
]
