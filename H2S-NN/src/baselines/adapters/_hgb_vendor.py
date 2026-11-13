"""Shared plumbing for the HGB-format polynomial baselines (PSHGCN, H2SGNN).

Both repositories descend from the same template: a ``processing.load_dataset(args)`` that
hardcodes ``root = "./data/"``, builds a dict of named relation adjacencies plus the
polynomial term lists, and returns the HGB split alongside them. Both are run here the
same way:

* their ``load_dataset`` executes verbatim, so the per-dataset special cases (ACM's
  paper-paper merge, DBLP's adjacency reordering) stay exactly as published;
* the ``data_loader`` symbol their module resolves at call time is patched to read our
  absolute raw-HGB path, which avoids both a chdir and a copy of a 127 MB ``node.dat``;
* only the train/validation split is replaced, with the project's canonical one.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

import torch

from src.baselines.bundles import NodeBundle, hgb_raw_dir
from src.baselines.runtime import pushd_sys_path

#: Our dataset keys -> the directory/config keys these repositories use.
VENDOR_DATASET = {"acm": "ACM", "dblp": "DBLP", "imdb": "IMDB"}


def load_vendor_dataset(vendor: str, dataset: str, args: Any, root: Path) -> tuple[Any, ...]:
    """Run ``<vendor>/processing.load_dataset(args)`` against our raw HGB directory under the
    bundle's data root `root` (`bundle_data_root`)."""

    raw_dir = hgb_raw_dir(dataset, root)
    with pushd_sys_path(vendor):
        import processing  # type: ignore[import-not-found]
        from dataset_loader import data_loader  # type: ignore[import-not-found]

        original = processing.data_loader
        # load_dataset computes `root + args.dataset` internally; ignore that and read the
        # absolute path instead. Nothing downstream of the loader is touched.
        processing.data_loader = lambda _path: data_loader(str(raw_dir))
        try:
            return processing.load_dataset(args)
        finally:
            processing.data_loader = original


def canonical_split(
    bundle: NodeBundle, vendor_test_idx: Any, *, vendor: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the project's canonical masks onto the vendor's target-local index space.

    Both index the HGB target node type as ``0..n_target-1`` in file order: our
    ``stack_node_features`` iterates ``data.node_types`` in the order HGB assigns type ids,
    and the target type is id 0 for ACM, DBLP and IMDB alike.

    The vendor's own test split is HGB's published test mask, so it must already agree
    with ours. If it does not, the two pipelines are indexing different graphs and every
    metric downstream would be meaningless -- so this raises rather than warns.
    """

    data = bundle.data
    train_idx = data.target_train_mask.nonzero(as_tuple=False).view(-1)
    val_idx = data.target_val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.target_test_mask.nonzero(as_tuple=False).view(-1)

    theirs = torch.as_tensor(vendor_test_idx, dtype=torch.long).sort().values
    if not torch.equal(theirs, test_idx.sort().values):
        raise RuntimeError(
            f"{vendor} and the project bundle disagree on the {bundle.dataset} test "
            f"split ({theirs.numel()} vs {test_idx.numel()} nodes). The two pipelines are "
            "not indexing the same graph."
        )
    return train_idx, val_idx, test_idx


def verify_labels(bundle: NodeBundle, vendor_labels: torch.Tensor, *, vendor: str) -> None:
    """Cross-check the vendor's label tensor against the project's for the target type.

    These repositories build ``labels`` over *every* node -- HGB's
    ``labels_train['data'] + labels_test['data']`` is an all-node array that is zero off
    the target type -- and then rely on target nodes occupying the leading block, so only
    the first ``n_target`` rows are ever indexed (PSHGCN/hgb/main.py:46-47). Compare the
    same leading block rather than the whole tensor.
    """

    ours = bundle.data.target_y
    num_target = int(ours.shape[0])
    theirs = vendor_labels.detach().cpu()
    if theirs.shape[0] < num_target:
        raise RuntimeError(
            f"{vendor} produced only {theirs.shape[0]} labels for {bundle.dataset}, "
            f"fewer than the {num_target} target nodes in the project bundle."
        )
    theirs = theirs[:num_target]
    if bundle.is_multilabel:
        matches = torch.equal(theirs.float(), ours.float())
    else:
        matches = torch.equal(theirs.long(), ours.long())
    if not matches:
        raise RuntimeError(
            f"{vendor}'s labels for {bundle.dataset} differ from the project bundle's; "
            "the comparison would not be like for like."
        )


__all__ = [
    "VENDOR_DATASET",
    "canonical_split",
    "load_vendor_dataset",
    "verify_labels",
]
