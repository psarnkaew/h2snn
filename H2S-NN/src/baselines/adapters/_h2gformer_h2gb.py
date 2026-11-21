"""H2G-former on the H2GB node-classification datasets (part 2 E1; D-P3: H2GB's datasets exactly as H2GB's configs;
D-P6; D-P9; author decision S1).

* Configuration: H2GB's own `<dataset>-H2Gformer.yaml` merged into H2GB's cfg (`_h2gb_network.configure_from_yaml`),
  the tuned keys overridden; the encoder stack is the config's (`_h2gformer_encoders.H2GB_STACKS`), MetaPath2Vec width
  `metapath_dim_pe(hidden)` (D-P6).
* MetaPath2Vec channel: H2GB's own `preprocess_Metapath(pe_dir, dataset)` (`H2GB/loader/encoding_generator.py:174-560`;
  its schema per dataset, `SparseAdam`, 5 epochs, D-P9) and the attachment of `master_loader.py:262-281`; the cache
  `pe_dir` is a run directory, never under the data root. Its `test()` fits a probe on every labelled target but only
  prints (`:488-490, 532-534`); the saved embedding is the one after 5 fixed epochs, so no label selects it.
  H2GB has no schema for `mag-year` (no branch for the name `mag-year`, `encoding_generator.py:187-436`;
  `MAGDataset.name = 'mag-year'`, `mag_dataset.py:67`): it takes ogbn-mag's (card 1c F6, decision G-7), stated.
* Label channel (H2GB's `Hetero_Label`): training labels only -- every non-training target gets the placeholder label
  and `val_mask` is all False (`training_label_input`), set **after** the loaders are built (their seeds come from the
  real masks); evaluation runs with `batch.split = 'val'` (H2GB masks with `train_mask` then,
  `hetero_label_encoder.py:41-46`). True labels for the loss and the metric are passed to the loop separately.
* Batches: H2GB's `hgt` sampler with the YAML fan-out (`src/baselines/minibatch.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.baselines.adapters._h2gformer_encoders import (H2GB_STACKS, LABEL, METAPATH, metapath_dim_pe,
                                                        training_label_input)

#: H2GB datasets whose H2G-former config cannot run in H2GB's own code (part 2 E1); none since card 1c F6.
UNRUNNABLE: dict[str, str] = {}
#: Card 1c F6 (decision G-7): mag-year has no MetaPath2Vec schema in H2GB's code (no branch for its name,
#: `encoding_generator.py:187-436`); it takes ogbn-mag's schema (`:263-280`) -- the same graph, only labels differ.
SCHEMA_NAME = {"mag-year": "ogbn-mag"}


def prepare_label_channel(data: Any, target_type: str, labels: torch.Tensor) -> None:
    """The label channel's inputs on `data`, in place: training labels only."""
    y, train, val = training_label_input(labels.cpu(), data[target_type].train_mask.cpu())
    data[target_type].y = y
    data[target_type].train_mask = train
    data[target_type].val_mask = val


class _AutoMultiClassCompat:
    """sklearn >= 1.8 removed `LogisticRegression(multi_class=...)`; H2GB's MetaPath2Vec probe passes
    `multi_class="auto"` with `solver="lbfgs"` (`H2GB/loader/embedding/metapath2vec.py:229-238`), which is the
    current default behaviour. While H2GB's `preprocess_Metapath` runs, `sklearn.linear_model.LogisticRegression` is a
    factory that drops `multi_class` when it is "auto" and returns sklearn's estimator (any other value is refused). The probe's accuracy is only
    printed (`encoding_generator.py:488-490, 532-534`); the embedding does not depend on it (card 1c F6 finding)."""

    def __enter__(self) -> None:
        import sklearn.linear_model as linear_model

        original = linear_model.LogisticRegression

        def LogisticRegression(*args: Any, multi_class: str = "auto", **kwargs: Any) -> Any:  # noqa: N802
            # a factory, not a subclass: sklearn estimators may not take varargs; the probe only instantiates it
            if multi_class != "auto":
                raise ValueError(f"multi_class={multi_class!r} has no equivalent in this sklearn")
            return original(*args, **kwargs)

        self._original, self._module = original, linear_model
        linear_model.LogisticRegression = LogisticRegression

    def __exit__(self, *exc: Any) -> None:
        self._module.LogisticRegression = self._original


def attach_h2gb_metapath(dataset_obj: Any, data: Any, pe_dir: Path, schema_name: str | None = None) -> None:
    """H2GB's MetaPath2Vec embeddings (trained by H2GB's code if `pe_dir` has none) as each type's
    `pestat_Hetero_Metapath` (`master_loader.py:262-281`). Needs H2GB's cfg configured first. `schema_name`: the
    dataset name H2GB's `preprocess_Metapath` dispatches on, when it differs from the dataset's own (decision G-7)."""
    from H2GB.loader.encoding_generator import (check_Metapath, load_Metapath,  # type: ignore[import-not-found]
                                                preprocess_Metapath)

    from src.baselines.adapters._h2gb_network import DatasetShim

    pe_dir = Path(pe_dir)
    if not pe_dir.exists() or not check_Metapath(str(pe_dir)):
        with _AutoMultiClassCompat():
            preprocess_Metapath(str(pe_dir),
                                dataset_obj if schema_name is None else DatasetShim(dataset_obj[0], schema_name))
    model = load_Metapath(str(pe_dir))
    emb = model["model"].weight.data.detach().cpu()
    for node_type in data.node_types:
        data[node_type]["pestat_Hetero_Metapath"] = emb[model["start"][node_type]:model["end"][node_type]]


def overrides(params: Any, num_classes: int, stack: tuple[str, ...]) -> dict[str, Any]:
    hidden = int(params.get("hidden_dim", 64))
    out = {"gnn.dim_inner": hidden, "gt.dim_hidden": hidden, "gt.layers": int(params.get("gt_layers", 3)),
           "gt.attn_heads": int(params.get("attn_heads", 8)), "gt.dropout": float(params.get("dropout", 0.2)),
           "gnn.dropout": float(params.get("dropout", 0.2)), "gt.attn_dropout": float(params.get("attn_dropout", 0.0)),
           "train.sampler": "hgt", "share.dim_out": int(num_classes)}
    if METAPATH in stack:
        out["posenc_Hetero_Metapath.dim_pe"] = metapath_dim_pe(hidden)
    return out


def build_gtmodel(data: Any, dataset: str, num_classes: int) -> torch.nn.Module:
    """H2GB's `GTModel` under the current cfg."""
    from H2GB.graphgym.config import cfg  # type: ignore[import-not-found]
    from H2GB.network.gt_model import GTModel  # type: ignore[import-not-found]

    from src.baselines.adapters._h2gb_network import DatasetShim

    return GTModel(cfg.share["dim_in"], num_classes, DatasetShim(data, dataset))


def fit(bundle: Any, params: Any, *, seed: int, device: torch.device) -> dict[str, float]:
    from src.baselines.adapters._h2gb_network import configure_from_yaml, seed_predictions
    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.bundles import bundle_data_root
    from src.baselines.h2gb_published import (fanout_from_args, h2gb_optimizer_and_schedule, is_h2gb_published,
                                              published_overrides, published_plan, resolve_published, run_settings)
    from src.baselines.protocol import NODE_CLASSIFICATION
    from src.baselines.minibatch import (eval_loader, h2gb_sample_sizes, hgt_loader, minibatch_plan,
                                         train_minibatch_node_classification, val_subset_loader)
    from src.baselines.runtime import pushd_sys_path
    from src.data.h2gb_nc import ALL_H2GB_NC_SPECS, RAW_ONLY, load_h2gb_dataset
    from src.utils.misc import seed_everything

    if bundle.dataset in UNRUNNABLE:
        raise RuntimeError(UNRUNNABLE[bundle.dataset])
    seed_everything(seed, device)
    dataset_obj = load_h2gb_dataset(bundle.dataset, bundle_data_root(bundle),
                                    bundle.metadata.get("seed") if bundle.dataset in RAW_ONLY else None)
    data = dataset_obj[0]
    target = ALL_H2GB_NC_SPECS[bundle.dataset].target_node_type
    stack = H2GB_STACKS[bundle.dataset]
    labels = data[target].y.long().clone()
    num_classes = int(labels.max()) + 1
    # card 8 C1: a published cell runs its yaml exactly (h2gb_published); the tuned path is unchanged
    published = is_h2gb_published("h2gformer", NODE_CLASSIFICATION, bundle.dataset, getattr(params, "source", ""))
    if published:
        values = dict(params.values)
        ensure_pyg_submodules()
        with pushd_sys_path("h2gb"):
            _, args = resolve_published("h2gformer", bundle.dataset, data, target, device, values,
                                        {"share.dim_out": int(num_classes)})
        if args["dataset.node_encoder_name"] != "+".join(stack):
            raise ValueError(f"{bundle.dataset}: yaml encoder {args['dataset.node_encoder_name']!r} differs from the "
                             f"stack {stack}.")
        settings = run_settings(args)
        plan = published_plan(bundle.dataset, settings)
        fanout = fanout_from_args(args, tuple(data.node_types))
        cfg_overrides = {**published_overrides(values), "share.dim_out": int(num_classes)}
        dim_pe = int(args["posenc_Hetero_Metapath.dim_pe"]) if METAPATH in stack else 0
    else:
        fanout = h2gb_sample_sizes("h2gformer", bundle.dataset, tuple(data.node_types))
        plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
        cfg_overrides = overrides(params, num_classes, stack)
        dim_pe = metapath_dim_pe(int(params.get("hidden_dim", 64))) if METAPATH in stack else 0
    loaders = (hgt_loader(data, fanout, target, "train", plan.batch_size, shuffle=True, steps=plan.steps_per_epoch),
               eval_loader(data, fanout, target, "val", plan.batch_size),
               eval_loader(data, fanout, target, "test", plan.batch_size))
    # card 1f B1a: built from the split masks, before the label channel rewrites them (as the loaders above)
    subset_loader, subset_ids = val_subset_loader(data, fanout, target, bundle.dataset, plan.batch_size,
                                                  run_seed=bundle.metadata.get("seed"))
    val_ids = data[target].val_mask.nonzero().view(-1).clone()   # before the label channel rewrites the masks
    test_ids = data[target].test_mask.nonzero().view(-1).clone()
    if LABEL in stack:
        prepare_label_channel(data, target, labels)
    ensure_pyg_submodules()
    with pushd_sys_path("h2gb"):
        cfg = configure_from_yaml("h2gformer", bundle.dataset, data, target, device, cfg_overrides)
        if METAPATH in stack:
            pe_dir = Path(params.get("metapath_cache", "experiments/h2gb_metapath2vec")) / bundle.dataset / \
                f"dim{dim_pe}_seed{seed}"
            attach_h2gb_metapath(dataset_obj, data, pe_dir, SCHEMA_NAME.get(bundle.dataset))
        torch.manual_seed(seed)
        model = build_gtmodel(data, bundle.dataset, num_classes).to(device)
        schedule = None
        if published:
            optimizer, schedule = h2gb_optimizer_and_schedule(model, cfg, args, settings)

    def forward_seeds(batch: Any) -> torch.Tensor:
        batch = batch.to(device)
        batch.split = "train" if model.training else "val"
        return seed_predictions(model, batch, target)

    if not published:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 5e-4)),
                                      weight_decay=float(params.get("weight_decay", 1e-5)))
    record = train_minibatch_node_classification(
        model=model, forward_seeds=forward_seeds, train_loader=loaders[0], val_loader=loaders[1],
        test_loader=loaders[2], val_ids=val_ids, test_ids=test_ids, val_subset_loader=subset_loader,
        val_subset_ids=subset_ids, target_type=target, optimizer=optimizer,
        criterion=torch.nn.functional.cross_entropy, plan=plan, report=getattr(params, "report", None), labels=labels,
        schedule=schedule)
    record["params_source"] = params.source
    record["node_encoder"] = "+".join(stack)
    record["batch_size"] = float(plan.batch_size)
    return record


__all__ = ["UNRUNNABLE", "attach_h2gb_metapath", "build_gtmodel", "fit", "overrides", "prepare_label_channel"]
