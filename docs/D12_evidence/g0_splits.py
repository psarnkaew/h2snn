"""G0 (a): our train/val/test node ids and labels vs the official HGB split.

Run from H2S-NN/ with the h2snn environment:  python ../docs/D12_evidence/g0_splits.py
Reference parser: HGB's own data_loader (vendored copy in vendor/HINormer/utils/data_loader.py),
which reads the raw files independently of PyG. Writes docs/D12_evidence/g0_splits.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402

RAW = {"acm": "data/acm/acm/raw/ACM", "dblp": "data/dblp/dblp/raw/DBLP", "imdb": "data/imdb/imdb/raw/IMDB"}
OUT = ROOT.parent / "docs" / "D12_evidence" / "g0_splits.json"


def hgb_loader():
    spec = importlib.util.spec_from_file_location(
        "hgb_data_loader", ROOT / "vendor" / "HINormer" / "utils" / "data_loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.data_loader


def hetsheaf_val(train_pool: torch.Tensor, val_ratio: float = 0.1) -> torch.Tensor:
    """HetSheaf's SubsetTrainSplit (seed=None): a fresh torch.Generator(), randperm, first int(r n)."""
    perm = torch.randperm(train_pool.numel(), generator=torch.Generator())
    shuffled = train_pool[perm]
    return shuffled[: int(train_pool.numel() * val_ratio)]


def class_counts(labels: np.ndarray) -> list[int]:
    return [int(c) for c in labels.sum(axis=0)]


def check(dataset: str, data_loader) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml").read_text())
    cfg["device"] = "cpu"
    ours = build_dataset(TrainConfig.from_dict(cfg))
    dl = data_loader(str(ROOT / RAW[dataset]))

    n_target = ours.target_indices.numel()
    shift0, count0 = int(dl.nodes["shift"][0]), int(dl.nodes["count"][0])
    hgb_train_pool = np.nonzero(dl.labels_train["mask"])[0]
    hgb_test = np.nonzero(dl.labels_test["mask"])[0]
    # Our target-local index i is HGB global id shift0 + i iff the target type is HGB type 0 in file order.
    local_to_hgb = shift0 + np.arange(n_target)
    our_train = local_to_hgb[ours.target_train_mask.numpy()]
    our_val = local_to_hgb[ours.target_val_mask.numpy()]
    our_test = local_to_hgb[ours.target_test_mask.numpy()]

    # Labels as multi-hot rows over HGB's class count.
    num_classes = dl.labels_train["num_classes"]
    if ours.is_multilabel:
        our_rows = ours.target_y.numpy().astype(int)
    else:
        our_rows = np.zeros((n_target, num_classes), dtype=int)
        valid = ours.target_y.numpy() >= 0
        our_rows[np.nonzero(valid)[0], ours.target_y.numpy()[valid]] = 1
    labelled_local = np.concatenate([our_train, our_val, our_test]) - shift0
    hgb_rows = np.where(dl.labels_train["mask"][:, None], dl.labels_train["data"], dl.labels_test["data"])
    label_mismatch = int((our_rows[labelled_local] != hgb_rows[labelled_local + shift0]).any(axis=1).sum())

    pool_t = torch.as_tensor(hgb_train_pool - shift0)
    hs_val = hetsheaf_val(pool_t).numpy() + shift0
    result = {
        "target_type_is_hgb_type0": bool(shift0 == 0 and count0 == n_target),
        "n_target": n_target,
        "sizes_ours": {"train": len(our_train), "val": len(our_val), "test": len(our_test)},
        "sizes_hgb": {"train_pool(label.dat)": len(hgb_train_pool), "test(label.dat.test)": len(hgb_test),
                      "val=int(0.2 n)": int(len(hgb_train_pool) * 0.2),
                      "train": len(hgb_train_pool) - int(len(hgb_train_pool) * 0.2)},
        "test_equal": bool(np.array_equal(np.sort(our_test), np.sort(hgb_test))),
        "train_union_val_equal_label_dat": bool(np.array_equal(
            np.sort(np.concatenate([our_train, our_val])), np.sort(hgb_train_pool))),
        "train_val_disjoint": bool(len(np.intersect1d(our_train, our_val)) == 0),
        "test_disjoint_from_train_val": bool(len(np.intersect1d(np.concatenate([our_train, our_val]), our_test)) == 0),
        "label_mismatch_count": label_mismatch,
        "labelled_checked": int(len(labelled_local)),
        "class_counts": {
            "train": class_counts(our_rows[our_train - shift0]),
            "val": class_counts(our_rows[our_val - shift0]),
            "test": class_counts(our_rows[our_test - shift0]),
            "hgb_test": class_counts(dl.labels_test["data"][hgb_test]),
        },
        "hetsheaf_split": {"val": len(hs_val), "train": len(hgb_train_pool) - len(hs_val),
                           "val_overlap_with_our_val": int(len(np.intersect1d(hs_val, our_val)))},
        "imdb_unlabelled_rows_in_pool": int((hgb_rows[labelled_local + shift0].sum(axis=1) == 0).sum()),
    }
    return result


def main() -> None:
    data_loader = hgb_loader()
    results = {}
    for dataset in sys.argv[1:] or ["acm", "dblp", "imdb"]:
        results[dataset] = check(dataset, data_loader)
        print(dataset, json.dumps(results[dataset]), flush=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
