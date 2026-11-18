"""B1a: the fixed validation subset used for model selection during training on the H2GB NC datasets.

Author decision B1a (card 1f; decisions log row "Card 2 review", 2026-11-16). On the H2GB NC datasets, validation
*during training* (checkpoint selection and the tuning objective) uses a fixed subset of the validation split of
H2GB's own size, drawn once from a fixed seed. The subset is the same for every method, baselines and H2S-NN alike.
The selected checkpoint is then evaluated on the whole validation and test splits.

The subset depends only on the dataset's validation ids: never on the method, the trial or the training seed.
This module is shared: the baseline loops (`src/baselines/minibatch.py`) and H2S-NN (card 5) import the same
function, and `docs/b1a_evidence/subset_fingerprints.json` records the fingerprints from the real splits.
"""

from __future__ import annotations

import hashlib

import torch

#: The one seed every subset is drawn with (never the training seed).
SEED = 0

#: Subset size per dataset: H2GB's own validation size (card 1f table); the subset is min(split, size).
SUBSET_SIZES: dict[str, int] = {
    "ogbn-mag": 16_384,
    "mag-year": 8_192,
    "oag-cs": 4_096,
    "oag-eng": 4_096,
    "oag-chem": 4_096,
    "ieee-cis-g": 8_192,
    "pdns": 8_192,
    "h-pokec": 16_384,
}

#: PDNS draws its 70/20/10 split from the run seed (`src/data/h2gb_nc.py:74-81`, H2GB `pdns_dataset.py:302-312`),
#: so its validation split is not one id set across seeds: card 1f stop item, refused until the design session
#: decides.
PDNS_STOP_ITEM = ("pdns: the validation split is drawn from the run seed (src/data/h2gb_nc.py:74-81), so it is not a "
                  "single id set across seeds (card 1f stop item; design-session decision pending)")
REFUSED: dict[str, str] = {"pdns": PDNS_STOP_ITEM}


def validation_subset(dataset: str, val_ids: torch.Tensor) -> torch.Tensor:
    """Sorted int64 ids: the first `min(len(val_ids), size)` of a `randperm` drawn with `SEED` over `val_ids`
    sorted ascending, then sorted. Refuses datasets outside the table (no silent whole-split fallback)."""
    if dataset in REFUSED:
        raise ValueError(REFUSED[dataset])
    if dataset not in SUBSET_SIZES:
        raise ValueError(f"{dataset!r} has no B1a validation subset (datasets: {sorted(SUBSET_SIZES)}); "
                         "there is no whole-split fallback.")
    ids = torch.sort(val_ids.detach().to("cpu", torch.int64).view(-1)).values
    if ids.numel() and bool((ids[1:] == ids[:-1]).any()):
        raise ValueError(f"{dataset}: validation ids are not distinct.")
    size = min(int(ids.numel()), SUBSET_SIZES[dataset])
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(ids.numel(), generator=generator)
    return torch.sort(ids[order[:size]]).values


def subset_fingerprint(ids: torch.Tensor) -> str:
    """sha256 of the ids as little-endian int64 bytes."""
    array = ids.detach().to("cpu", torch.int64).contiguous().numpy()
    return hashlib.sha256(array.astype("<i8").tobytes()).hexdigest()


__all__ = ["PDNS_STOP_ITEM", "SEED", "SUBSET_SIZES", "subset_fingerprint", "validation_subset"]
