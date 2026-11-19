"""Card 7 amendment 3 (author 2026-11-19): the edge-list fingerprint of a perturbation graph.

One canonical form, computed on both sides of the check:
- the **generator's** fingerprint, from the HGB files it wrote (`info.dat`, `node.dat`, `link.dat`), parsed here
  without PyG (`from_raw`);
- the **run's** fingerprint, from the graph its loader actually built (`from_prepared`: H2S-NN's `build_dataset`
  output, which is also every baseline bundle's `bundle.data`, `src/baselines/bundles.py: build_bundle`).

Canonical form: the set of facts as unordered pairs of endpoints, an endpoint being (node type name, local index),
with PyG `HGBDataset`'s naming and numbering (type names from `info.dat`'s `node.dat` / `node type` block; local
index = position among the nodes of that type in `node.dat` order, `torch_geometric/datasets/hgb_dataset.py`).
Link types are not part of it: a fact and its mirrored reverse line are one pair, and H2S-NN's fact layer (which
keeps one of two mirrored relations) and a baseline's symmetrised view both keep the same pairs. Each endpoint is
encoded as `rank(type name) * 2**32 + local`, a pair as (min, max), the unique pairs are sorted and the SHA-256 of
their int64 little-endian bytes is the fingerprint; the number of pairs is reported beside it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SHIFT = np.int64(1) << np.int64(32)


def _digest(keys_a: np.ndarray, keys_b: np.ndarray) -> dict[str, Any]:
    pairs = np.stack([np.minimum(keys_a, keys_b), np.maximum(keys_a, keys_b)], axis=1).astype("<i8")
    pairs = np.unique(pairs, axis=0)                                   # sorted lexicographically, unique rows
    return {"sha256": hashlib.sha256(np.ascontiguousarray(pairs).tobytes()).hexdigest(),
            "pairs": int(pairs.shape[0])}


def _ranks(type_names) -> dict[str, int]:
    return {name: i for i, name in enumerate(sorted(set(type_names)))}


def from_raw(raw_dir: Path) -> dict[str, Any]:
    """The fingerprint of the HGB release in `raw_dir` (`.../raw/<NAME>/`), from its files alone."""
    raw_dir = Path(raw_dir)
    info = json.loads((raw_dir / "info.dat").read_text(encoding="utf-8"))
    n_types = {int(k): v for k, v in info["node.dat"]["node type"].items()}
    ranks = _ranks(n_types.values())
    key_of: dict[int, int] = {}
    counts: dict[str, int] = {}
    with (raw_dir / "node.dat").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split("\t", 3)
            name = n_types[int(parts[2])]
            local = counts.get(name, 0)
            counts[name] = local + 1
            key_of[int(parts[0])] = ranks[name] * (1 << 32) + local
    src, dst = [], []
    with (raw_dir / "link.dat").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.split("\t", 3)
            src.append(key_of[int(parts[0])])
            dst.append(key_of[int(parts[1])])
    return _digest(np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64))


def from_prepared(data: Any) -> dict[str, Any]:
    """The fingerprint of a loaded graph (`PreparedH2SData`: `edge_index` over the stacked node index,
    `node_type` / `node_type_names` / `node_slices` giving each node's type and local index)."""
    names = list(data.node_type_names)
    ranks = _ranks(names)
    node_type = data.node_type.cpu().numpy().astype(np.int64)
    starts = np.asarray([data.node_slices[n][0] for n in names], dtype=np.int64)
    type_rank = np.asarray([ranks[n] for n in names], dtype=np.int64)
    keys = type_rank[node_type] * SHIFT + (np.arange(node_type.shape[0], dtype=np.int64) - starts[node_type])
    edge_index = data.edge_index.cpu().numpy().astype(np.int64)
    return _digest(keys[edge_index[0]], keys[edge_index[1]])
