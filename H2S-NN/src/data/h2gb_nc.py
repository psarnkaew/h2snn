"""The H2GB node-classification datasets through H2GB's own loaders (Part M3b B2; spec v1.22 §6).

Equal-output wrapper: each dataset is loaded by H2GB's own ``preformat_*`` function of
``H2GB/loader/master_loader.py`` (vendor commit ``d3b49db``), which builds H2GB's dataset class, applies the
transform H2GB applies and sets the split H2GB sets; the graph returned is ``dataset[0]``, the object H2GB's
samplers read (``H2GB/sampler/custom_sampler.py:92, 330``). Nothing of H2GB's preprocessing is re-implemented.

The one stated deviation (decision of this card, Rule "loaders read the processed files offline"): while
``preformat_*`` runs, the dataset class it names is replaced by a subclass whose PyG hooks ``_download`` and
``_process`` only check that the processed file exists and raise otherwise. H2GB's classes would otherwise
download whenever a raw file is missing (``OAGDataset.download``, ``oag_dataset.py:87-88``, with an empty
``NC/OAG/raw``) or re-process when the processed file is missing; PyG would also create ``raw/`` folders. The
class's own ``__init__`` then loads the processed file exactly as H2GB does. Nothing under the data root is
written, downloaded or deleted.

Per dataset (``H2GB_NC_SPECS``): H2GB's class and loader function, the H2GB name, the target type, the label
kind and the split source, with file:line. Roots: ``src/data/data_layout.py`` (``H2GB_NC_DIRS``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from src.data.data_layout import H2GB_NC_DIRS, h2gb_nc_dataset_dir, h2gb_nc_processed_file

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


@dataclass(frozen=True, slots=True)
class H2GBNCSpec:
    """How H2GB builds one dataset (VERIFIED-CODE, vendor H2GB ``d3b49db``)."""

    key: str
    class_name: str          # the name `master_loader.py` imports the class under
    loader: str              # H2GB's preformat function in `H2GB/loader/master_loader.py`
    h2gb_name: str | None    # the `name` argument of the preformat function (None: it takes none)
    target_node_type: str    # H2GB's `dataset.task_entity`
    label: str               # "multiclass" or "binary" (single-label in every case)
    split: str               # where H2GB's split comes from


H2GB_NC_SPECS: dict[str, H2GBNCSpec] = {
    "ogbn-mag": H2GBNCSpec(
        "ogbn-mag", "MAGDataset", "preformat_MAG", "ogbn-mag", "paper", "multiclass",
        "OGB's official split, in the processed file (mag_dataset.py:130-142)"),
    "mag-year": H2GBNCSpec(
        "mag-year", "MAGDataset", "preformat_MAG", "mag-year", "paper", "multiclass",
        "random 50/25/25 drawn at processing time, in the processed file (mag_dataset.py:163-172)"),
    "oag-cs": H2GBNCSpec(
        "oag-cs", "OAGDataset", "preformat_OAG", "cs", "paper", "multiclass",
        "by year: <= 2016 / 2017 / >= 2018, in the processed file (oag_dataset.py:151-153)"),
    "oag-eng": H2GBNCSpec(
        "oag-eng", "OAGDataset", "preformat_OAG", H2GB_NC_DIRS["oag-eng"][1][0], "paper",
        "multiclass", "by year: <= 2016 / 2017 / >= 2018, in the processed file (oag_dataset.py:151-153)"),
    "oag-chem": H2GBNCSpec(
        "oag-chem", "OAGDataset", "preformat_OAG", H2GB_NC_DIRS["oag-chem"][1][0], "paper",
        "multiclass", "by year: <= 2016 / 2017 / >= 2018, in the processed file (oag_dataset.py:151-153)"),
    "ieee-cis-g": H2GBNCSpec(
        "ieee-cis-g", "IeeeCisDataset", "preformat_IEEE_CIS", None, "transaction", "binary",
        "node order: first 80 % train, next 10 % val, rest test, set at load (master_loader.py:766-775)"),
    "h-pokec": H2GBNCSpec(
        "h-pokec", "PokecDataset", "preformat_Pokec", None, "user", "binary",
        "random 50/25/25 of labelled users drawn at processing time, in the processed file "
        "(pokec_dataset.py:291-301)"),
}

#: Datasets H2GB builds from `raw/` on every load (no processed file); their split is drawn from the run seed.
RAW_ONLY_SPECS: dict[str, H2GBNCSpec] = {
    # Card baselines-h2gb C2 (author decision 1, 2026-11-16): H2GB's code exactly -- one graph over windows 0-59,
    # `ToUndirected`, labels from `domains2.csv` (master_loader.py:604-617), a random 70/20/10 split of the labelled
    # domains by an unseeded `torch.randperm` (pdns_dataset.py:302-312) that H2GB fixes with `seed_everything(seed)`
    # just before loading (H2GB/main.py:87-101). No processed file: `raw/` is read on every load.
    "pdns": H2GBNCSpec(
        "pdns", "PDNSDataset", "preformat_PDNS", None, "domain_node", "binary",
        "random 70/20/10 of the labelled domains from the run seed (pdns_dataset.py:302-312, H2GB/main.py:87-101)"),
}
RAW_ONLY: frozenset[str] = frozenset(RAW_ONLY_SPECS)
#: Every H2GB NC dataset with a loader (processed-file datasets and raw-only ones).
ALL_H2GB_NC_SPECS: dict[str, H2GBNCSpec] = {**H2GB_NC_SPECS, **RAW_ONLY_SPECS}

#: What H2GB does with a node type that has no `x`: a learned `nn.Embedding` row per node
#: (`H2GB/encoder/hetero_raw_encoder.py:113-119`); H2S-NN's encoder does the same (learned per-node vectors).
FEATURELESS_TYPES_RULE = ("learned per-node embedding (H2GB/encoder/hetero_raw_encoder.py:113-119); "
                          "the same in H2S-NN's TypeSpecificNodeEncoder")


def _offline(cls: type) -> type:
    """`cls` whose PyG download/process hooks only require the processed files (no download, no write)."""

    class Offline(cls):  # type: ignore[misc, valid-type]
        def _require_processed(self) -> None:
            missing = [path for path in self.processed_paths if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(
                    f"H2GB processed file(s) missing: {missing}. The H2GB loaders of this project read the "
                    "processed files offline and never download or process (Part M3b B2); prepare them with "
                    "scripts/slurm/run_use_dataset.sbatch.")

        def _download(self) -> None:
            self._require_processed()

        def _process(self) -> None:
            self._require_processed()

    Offline.__name__ = cls.__name__
    Offline.__qualname__ = cls.__qualname__
    return Offline


def _offline_raw(cls: type) -> type:
    """`cls` (a raw-only H2GB class: no `process`, no processed file) whose PyG download hook only requires every raw
    file H2GB lists (`raw_file_names`) and never downloads; H2GB's class has no `process`, so nothing is written."""

    class OfflineRaw(cls):  # type: ignore[misc, valid-type]
        def _download(self) -> None:
            missing = [path for path in self.raw_paths if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(
                    f"H2GB raw file(s) missing: {missing[:3]}{' ...' if len(missing) > 3 else ''} "
                    f"({len(missing)} of {len(self.raw_paths)}). This loader never downloads (card baselines-h2gb C2).")

        def _process(self) -> None:
            return None

    OfflineRaw.__name__ = cls.__name__
    OfflineRaw.__qualname__ = cls.__qualname__
    return OfflineRaw


def load_h2gb_dataset(key: str, root: Path | str, seed: int | None = None) -> Any:
    """H2GB's dataset object of `key` as H2GB's loader returns it (transform and split included). A raw-only dataset
    (PDNS) needs the run `seed`: H2GB seeds every generator with it just before loading (`H2GB/main.py:87-101`)."""
    if key not in ALL_H2GB_NC_SPECS:
        raise ValueError(f"unknown H2GB node-classification dataset {key!r}; known: {sorted(ALL_H2GB_NC_SPECS)}")
    spec = ALL_H2GB_NC_SPECS[key]
    if key in RAW_ONLY:
        if seed is None:
            raise ValueError(f"{key}: the split is drawn from the run seed; pass `seed`.")
    else:
        processed = h2gb_nc_processed_file(Path(root), key)
        if not processed.is_file():
            raise FileNotFoundError(
                f"{key}: processed file {processed} is missing; this loader never downloads or processes "
                "(Part M3b B2).")
    from src.baselines.adapters._pyg_compat import ensure_pyg_submodules
    from src.baselines.runtime.vendor_paths import pushd_sys_path

    ensure_pyg_submodules()
    with pushd_sys_path("h2gb"):
        import H2GB  # noqa: F401  - H2GB's package registers its config groups (cfg.dataset.rand_split)
        import H2GB.loader.master_loader as master_loader  # type: ignore[import-not-found]
        from H2GB.graphgym.config import cfg, set_cfg  # type: ignore[import-not-found]

        set_cfg(cfg)  # H2GB's defaults: dataset.rand_split False (H2GB/config/dataset_config.py:24)
        original = getattr(master_loader, spec.class_name)
        setattr(master_loader, spec.class_name, (_offline_raw if key in RAW_ONLY else _offline)(original))
        try:
            preformat = getattr(master_loader, spec.loader)
            dataset_dir = str(h2gb_nc_dataset_dir(Path(root), key))
            if seed is not None:
                from torch_geometric import seed_everything

                seed_everything(int(seed))  # H2GB/main.py:87, the only step before create_loader that matters here
            if spec.h2gb_name is None:
                dataset = preformat(dataset_dir)
            else:
                dataset = preformat(dataset_dir, spec.h2gb_name)
        finally:
            setattr(master_loader, spec.class_name, original)
    return dataset


def load_h2gb_nc(key: str, root: Path | str, seed: int | None = None) -> "HeteroData":
    """The graph H2GB trains and evaluates `key` on (`dataset[0]`), for `build_dataset` (`seed`: raw-only datasets)."""
    return load_h2gb_dataset(key, root, seed if key in RAW_ONLY else None)[0]


def split_record(data: "HeteroData", target_type: str) -> dict[str, Any]:
    """Sizes, class counts and a sha256 of each index set of the split H2GB set on `data` (recorded per seed)."""
    import hashlib

    import numpy as np

    record: dict[str, Any] = {}
    y = data[target_type].y
    for split in ("train", "val", "test"):
        index = data[target_type][f"{split}_mask"].nonzero().view(-1)
        counts = torch.bincount(y[index], minlength=int(y.max()) + 1).tolist() if index.numel() else []
        record[split] = {"size": int(index.numel()), "class_counts": counts,
                         "sha256": hashlib.sha256(np.ascontiguousarray(index.numpy().astype(np.int64))).hexdigest()}
    return record


__all__ = ["FEATURELESS_TYPES_RULE", "H2GBNCSpec", "ALL_H2GB_NC_SPECS", "H2GB_NC_SPECS", "RAW_ONLY", "RAW_ONLY_SPECS", "load_h2gb_dataset", "load_h2gb_nc",
           "split_record"]
