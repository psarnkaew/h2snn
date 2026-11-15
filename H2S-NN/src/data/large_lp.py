"""The two large link-prediction tasks (spec §6 LP, D7/F21; card Part M3b B6): ogbn-mag paper–paper citation
and H-Pokec user–user friendship, built from the B2 node-classification graphs (H2GB's own loaders).

* **Split** (spec v1.8 F10, as `scripts/lp/build_derived_lp.py:95-105, 171-212`): the non-loop **unordered**
  pairs of the target relation, sorted, permuted with numpy's `default_rng(seed)`; the first `round(0.10 n)`
  are test, the next `round(0.09 n)` validation, the rest training (81/9/10). A pair and its reverse are one
  pair, so they always fall in the same part. Self-loops are in no part (they stay in the message graph).
* **Message graph**: H2GB's loaded graph (`dataset[0]`) without every fact whose node pair is a held-out
  (validation or test) pair, in both orientations and whatever its relation (`held_out_fact_mask`); training
  positives stay in it.
* **Files** (`data/LP/large/<task>/`, never overwritten): `split.npz` (train/valid/test pairs, `[P, 2]`
  int64, type-local ids, `u < v`), `negatives.npz` (HGB test 2-hop and random sets, validation negatives),
  `meta.json` (seeds, counts, source checksum). Written by `scripts/lp/build_large_lp.py` in a SLURM job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.data_layout import large_lp_dir

#: numpy default_rng seed of the pair permutation (the derived tasks' recorded default, build_derived_lp.py:256)
LARGE_LP_SPLIT_SEED = 42
LARGE_LP_VAL_FRAC = 0.09
LARGE_LP_TEST_FRAC = 0.10
#: seed of HGB's `get_valid_neg` draw written into the validation file (fixed once per task)
LARGE_LP_VALID_NEG_SEED = 1


@dataclass(frozen=True, slots=True)
class LargeLPTask:
    key: str
    nc_dataset: str                       # the B2 loader key
    relation: tuple[str, str, str]        # the target relation in H2GB's loaded graph
    node_type: str                        # both endpoints (same-type tasks)


LARGE_LP_TASKS: dict[str, LargeLPTask] = {
    "ogbn-mag_pp": LargeLPTask("ogbn-mag_pp", "ogbn-mag", ("paper", "cites", "paper"), "paper"),
    "h-pokec_uu": LargeLPTask("h-pokec_uu", "h-pokec", ("user", "has_friend", "user"), "user"),
}


def unordered_pairs(edge_index: torch.Tensor) -> tuple[np.ndarray, dict[str, int]]:
    """Sorted unique non-loop pairs `(min, max)` of `edge_index` ([2, E]) and the fact statistics:
    facts, self-loop facts, distinct ordered facts, pairs, pairs stored in both orientations, pairs with
    a repeated identical fact (multi-edge), pairs carrying more than one fact."""
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    loop = src == dst
    lo, hi = np.minimum(src, dst)[~loop], np.maximum(src, dst)[~loop]
    width = int(max(hi.max(initial=0), lo.max(initial=0))) + 1
    ordered_key = src[~loop] * width + dst[~loop]
    pair_key = lo * width + hi
    pair_unique, facts_per_pair = np.unique(pair_key, return_counts=True)
    ordered_unique, ordered_count = np.unique(ordered_key, return_counts=True)
    multi_pairs = np.unique((np.minimum(ordered_unique // width, ordered_unique % width) * width
                             + np.maximum(ordered_unique // width, ordered_unique % width))
                            [ordered_count > 1]).size
    orientations = np.unique(np.minimum(ordered_unique // width, ordered_unique % width) * width
                             + np.maximum(ordered_unique // width, ordered_unique % width),
                             return_counts=True)[1]
    stats = {"facts": int(src.size), "self_loop_facts": int(loop.sum()),
             "distinct_ordered_facts": int(ordered_unique.size), "pairs": int(pair_unique.size),
             "pairs_in_both_orientations": int((orientations == 2).sum()),
             "pairs_with_a_repeated_fact": int(multi_pairs),
             "pairs_with_more_than_one_fact": int((facts_per_pair > 1).sum())}
    pairs = np.stack([pair_unique // width, pair_unique % width], axis=1)
    return pairs, stats


def split_pairs(pairs: np.ndarray, seed: int = LARGE_LP_SPLIT_SEED, val_frac: float = LARGE_LP_VAL_FRAC,
                test_frac: float = LARGE_LP_TEST_FRAC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, validation, test) of sorted unordered pairs, as `build_derived_lp.split_facts`."""
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    ordered = pairs[order]
    n = ordered.shape[0]
    n_test, n_val = round(test_frac * n), round(val_frac * n)
    perm = np.random.default_rng(seed).permutation(n)
    return (ordered[perm[n_test + n_val:]], ordered[perm[n_test:n_test + n_val]], ordered[perm[:n_test]])


def held_out_fact_mask(edge_index: torch.Tensor, held_out: torch.Tensor) -> torch.Tensor:
    """True for every fact of `edge_index` ([2, E], global ids) whose node pair is a pair of `held_out`
    ([2, H], global ids), in either orientation, whatever its relation."""
    if held_out.numel() == 0:
        return torch.zeros(edge_index.size(1), dtype=torch.bool)
    width = int(max(int(edge_index.max()), int(held_out.max()))) + 1
    lo = torch.minimum(held_out[0], held_out[1]).long()
    hi = torch.maximum(held_out[0], held_out[1]).long()
    keys = torch.unique(lo * width + hi)
    e_lo = torch.minimum(edge_index[0], edge_index[1]).long()
    e_hi = torch.maximum(edge_index[0], edge_index[1]).long()
    return torch.isin(e_lo * width + e_hi, keys)


__all__ = ["LARGE_LP_SPLIT_SEED", "LARGE_LP_TASKS", "LARGE_LP_TEST_FRAC", "LARGE_LP_VALID_NEG_SEED",
           "LARGE_LP_VAL_FRAC", "LargeLPTask", "held_out_fact_mask", "large_lp_dir", "split_pairs",
           "unordered_pairs"]
