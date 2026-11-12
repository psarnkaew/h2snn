"""Part N-fix F1c: why mutation M2 (validation both ways in the 2-hop adjacency) is not caught by
the oracle datasets. For every dataset x seed, compares, for the test heads, the 2-hop candidates
and the known-neighbour sets of the adjacency with validation one way (HGB, ours) and both ways
(M2). Read-only; run from `H2S-NN/`."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.data.hgb_lp_negatives import hgb_adjacency, two_hop_candidates  # noqa: E402
from tests.test_hgb_oracle import (DATASETS, SEEDS, _edge_index, _facts_both_ways,  # noqa: E402
                                   _hgb_module, _load_hgb, _tail_range, _valid_one_way, _write)

hgb = _hgb_module()
tmp = Path(tempfile.mkdtemp())
for name, build in DATASETS.items():
    folder = tmp / name
    _write(folder, build())
    for seed in SEEDS:
        dl = _load_hgb(hgb, folder, seed)
        both, valid = _facts_both_ways(dl), _valid_one_way(dl)
        n = dl.nodes["total"]
        one_way = hgb_adjacency(both, valid, n)                                 # HGB, ours
        m2 = hgb_adjacency(torch.cat([both, valid], dim=1), valid[:, :0], n)    # M2: both ways
        for relation in dl.test_types:
            heads = np.unique(_edge_index(dl.links_test["data"][relation].nonzero())[0].numpy())
            rng = _tail_range(dl, relation)
            c1 = two_hop_candidates(one_way, heads, rng)
            c2 = two_hop_candidates(m2, heads, rng)
            cand_changed = sum(not np.array_equal(a, b) for a, b in zip(c1, c2))
            known_changed = sum(
                set(one_way[h].indices.tolist()) != set(m2[h].indices.tolist()) for h in heads)
            print(f"{name:26s} seed {seed}  validation facts {valid.size(1):2d}  test heads "
                  f"{heads.size:2d}  heads whose 2-hop candidates change {cand_changed}  "
                  f"heads whose known set changes {known_changed}")
