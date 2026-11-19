"""B1a: the fixed validation subset used for model selection during training on the H2GB NC datasets.

Author decision B1a (card 1f; decisions log row "Card 2 review", 2026-11-16). On the H2GB NC datasets, validation
*during training* (checkpoint selection and the tuning objective) uses a fixed subset of the validation split of
H2GB's own size, drawn once from a fixed seed. The subset is the same for every method, baselines and H2S-NN alike.
The selected checkpoint is then evaluated on the whole validation and test splits.

The subset depends only on the dataset's validation ids: never on the method or the trial. PDNS is the one exception
to "one subset per dataset" (card 1f U5.3, decision (a)): its split is drawn from the run seed, so its subset is drawn
from that seed's validation split with an RNG seeded by (`SEED`, run seed); the same run seed gives the same nodes for
every method.
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

#: Datasets whose split is drawn from the run seed (`src/data/h2gb_nc.py:74-81`, H2GB `pdns_dataset.py:302-312`):
#: the subset is drawn per run seed (card 1f U5.3, decision (a)).
PER_RUN_SEED: frozenset[str] = frozenset({"pdns"})


def subset_generator_seed(run_seed: int) -> int:
    """The RNG seed of a per-run-seed subset: a function of (`SEED`, run seed) only."""
    digest = hashlib.sha256(f"b1a:{SEED}:{int(run_seed)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def validation_subset(dataset: str, val_ids: torch.Tensor, run_seed: int | None = None) -> torch.Tensor:
    """Sorted int64 ids: the first `min(len(val_ids), size)` of a `randperm` over `val_ids` sorted ascending, then
    sorted. The RNG is seeded with `SEED`; for a per-run-seed dataset (PDNS) with `subset_generator_seed(run_seed)`,
    and `val_ids` must be that run seed's validation split. `run_seed` is ignored for the other datasets. Refuses
    datasets outside the table (no silent whole-split fallback)."""
    if dataset not in SUBSET_SIZES:
        raise ValueError(f"{dataset!r} has no B1a validation subset (datasets: {sorted(SUBSET_SIZES)}); "
                         "there is no whole-split fallback.")
    ids = torch.sort(val_ids.detach().to("cpu", torch.int64).view(-1)).values
    if ids.numel() and bool((ids[1:] == ids[:-1]).any()):
        raise ValueError(f"{dataset}: validation ids are not distinct.")
    size = min(int(ids.numel()), SUBSET_SIZES[dataset])
    if dataset in PER_RUN_SEED:
        if run_seed is None:
            raise ValueError(f"{dataset}: the split is drawn from the run seed; pass `run_seed` (card 1f U5.3).")
        generator = torch.Generator().manual_seed(subset_generator_seed(run_seed))
    else:
        generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(ids.numel(), generator=generator)
    return torch.sort(ids[order[:size]]).values


def subset_fingerprint(ids: torch.Tensor) -> str:
    """sha256 of the ids as little-endian int64 bytes."""
    array = ids.detach().to("cpu", torch.int64).contiguous().numpy()
    return hashlib.sha256(array.astype("<i8").tobytes()).hexdigest()


__all__ = ["PER_RUN_SEED", "SEED", "SUBSET_SIZES", "subset_fingerprint", "subset_generator_seed", "validation_subset"]
