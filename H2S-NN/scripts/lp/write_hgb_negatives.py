"""Write each link-prediction task's HGB test negatives once: both test sets of HGB's runner
(spec v1.16 §6, K1-K2; cards Part M M7, Part N N1).

HGB's `get_test_neigh`, ported in `src/data/hgb_lp_negatives.py`: 2-hop candidates of the
symmetric train + validation + test adjacency, direct links and self-loops removed, filtered to the
relation's head and tail types, `random.choices` per head (ascending) with k = the head's test
positives, seed 1; a head without candidates gets no negatives (no fallback); validation positives
enter that adjacency as stored only. Random set (`get_test_neigh_w_random`): one tail per test
positive, redrawn while a known neighbour, seed 1. Output, in raw HGB ids, under
`<task folder>/hgb_negatives/`: `test_neg_2hop_rel<r>.npy`, `test_neg_random_rel<r>.npy`
([M, 2] int64, head, tail) and
`meta.json` (the test positives' md5 the file was drawn for, counts, the heads without
candidates and the share of test positives they carry). The loader reads the file when present
(and refuses one written for other positives), so every method (H2S-NN and the baseline bundles)
sees the same negatives. Validation negatives (`get_valid_neg`) are drawn by the loader itself
with a recorded seed (cheap, deterministic).

    python scripts/lp/write_hgb_negatives.py --tasks acm_pa dblp_ap imdb_ma freebase_bb pubmed lastfm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.config import TrainConfig  # noqa: E402
from src.data.data_layout import REPO_DATA_ROOT  # noqa: E402
from src.data.hgb_lp_negatives import HGB_TEST_NEGATIVE_SEED  # noqa: E402
from src.data.link_prediction_dataset import (HGB_NEGATIVES_DIR, _load_local_hgb_link_data,  # noqa: E402
                                              build_link_prediction_dataset, hgb_raw_pairs,
                                              local_hgb_link_dir)

TARGETS: dict[str, tuple[str, str, str]] = {
    "acm_pa": ("paper", "rel_2", "author"),
    "dblp_ap": ("author", "rel_0", "paper"),
    "imdb_ma": ("movie", "rel_2", "actor"),
    "freebase_bb": ("book", "rel_0", "book"),
    "pubmed": ("disease", "rel_2", "disease"),
    "lastfm": ("user", "rel_0", "artist"),
}


def task_config(root: Path, task: str) -> TrainConfig:
    # path cells are not used here: the smallest WISP draw keeps the load cheap
    return TrainConfig.from_dict({"device": "cpu", "data": {
        "dataset_name": task, "task_type": "link_prediction", "root": str(root),
        "target_relation": list(TARGETS[task]), "wisp_walks_per_anchor": 1,
        "wisp_max_length": 2}})


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_task(root: Path, task: str) -> dict:
    start = time.perf_counter()
    folder = local_hgb_link_dir(root, task)
    out_dir = folder / HGB_NEGATIVES_DIR
    if (out_dir / "meta.json").exists():  # recompute, never read the old file
        (out_dir / "meta.json").unlink()
    config = task_config(root, task)
    data = build_link_prediction_dataset(config)
    split = _load_local_hgb_link_data(config)
    local_to_raw = {(split.raw_to_type_name[raw], local): raw
                    for raw, local in split.raw_to_local.items()}
    src_type, rel_name, dst_type = TARGETS[task]
    rel_id = int(rel_name.removeprefix("rel_"))
    src_offset, dst_offset = data.src_node_offset, data.dst_node_offset
    types = [src_type, dst_type]
    raw_pos = hgb_raw_pairs(data.test_pos_edge_index, local_to_raw, src_offset, dst_offset, types)
    raw_neg = hgb_raw_pairs(data.test_neg_edge_index, local_to_raw, src_offset, dst_offset, types)
    raw_random = hgb_raw_pairs(data.test_neg_random_edge_index, local_to_raw, src_offset,
                               dst_offset, types)
    heads_without = sorted(local_to_raw[(src_type, int(h) - src_offset)]
                           for h in data.test_heads_without_negatives.tolist())
    out_dir.mkdir(parents=True, exist_ok=True)
    neg_file = f"test_neg_2hop_rel{rel_id}.npy"
    random_file = f"test_neg_random_rel{rel_id}.npy"
    np.save(out_dir / neg_file, raw_neg)
    np.save(out_dir / random_file, raw_random)
    for stale in out_dir.glob("test_neg_rel*.npy"):  # Part M's single-set file
        stale.unlink()
    n_pos = int(raw_pos.shape[0])
    heads_total = int(np.unique(raw_pos[:, 0]).size)
    entry = {
        "types": types,
        "test_neg_2hop_file": neg_file,
        "test_neg_2hop_md5": md5(out_dir / neg_file),
        "test_neg_random_file": random_file,
        "test_neg_random_md5": md5(out_dir / random_file),
        "test_pos_md5": hashlib.md5(np.ascontiguousarray(raw_pos).tobytes()).hexdigest(),
        "test_positives": n_pos,
        "test_negatives_2hop": int(raw_neg.shape[0]),
        "test_negatives_random": int(raw_random.shape[0]),
        "two_hop_empty_reason": data.test_2hop_empty_reason,
        "test_heads": heads_total,
        "heads_without_candidates": heads_without,
        "heads_without_candidates_count": len(heads_without),
        "test_positives_without_negatives": int(data.test_positives_without_negatives),
        "share_of_test_positives_without_negatives": (
            data.test_positives_without_negatives / n_pos if n_pos else 0.0),
    }
    meta = {
        "rule": "HGB run_new.py test sets: get_test_neigh (2-hop; train and test both ways, "
                "validation as stored; no fallback) and get_test_neigh_w_random (random, "
                "redraw while known), LP/benchmark/scripts/data_loader.py @2ccb1d8, ported in "
                "src/data/hgb_lp_negatives.py",
        "seed": HGB_TEST_NEGATIVE_SEED,
        "task": task,
        "target_relation": list(TARGETS[task]),
        "inputs_md5": {name: md5(folder / name)
                       for name in ("node.dat", "link.dat", "link.dat.test", "link.dat.valid")
                       if (folder / name).exists()},
        "relations": {str(rel_id): entry},
        "seconds": round(time.perf_counter() - start, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=REPO_DATA_ROOT)
    parser.add_argument("--tasks", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    args = parser.parse_args()
    for task in args.tasks:
        meta = write_task(args.data_root, task)
        entry = next(iter(meta["relations"].values()))
        print(json.dumps({"task": task, **{k: v for k, v in entry.items()
                                           if k != "heads_without_candidates"},
                          "seconds": meta["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
