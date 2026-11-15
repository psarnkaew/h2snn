"""Part M3b B6 data job: build one large link-prediction task (ogbn-mag P–P or H-Pokec U–U) and its negatives.

    python scripts/lp/build_large_lp.py --root data --task ogbn-mag_pp --report <file.json> [--head-chunk 512]

From the B2 graph (H2GB's own loader, `load_h2gb_dataset`): (1) duplicate facts per node pair of the target
relation (stored and loaded graph) and the other relations between the same node type; (2) the 81/9/10 split by
unordered pair (`src/data/large_lp.py`); (3) the message graph without every fact of a held-out pair (both
orientations, any relation), checked to hold none of them; (4) HGB's test negatives (2-hop `get_test_neigh` and
random `get_test_neigh_w_random`, seed 1) on the message graph + test facts both ways + validation facts as
stored, and the validation negatives (`get_valid_neg`, seed 1), via `src/data/hgb_lp_negatives.py`.
Writes `data/LP/large/<task>/{split.npz, negatives.npz, meta.json}` (refuses if the folder exists: nothing is
overwritten) and a small report (counts, timings, peak memory) to `--report`. Run in a SLURM job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.data_layout import h2gb_nc_processed_file, large_lp_dir  # noqa: E402
from src.data.h2gb_nc import load_h2gb_dataset  # noqa: E402
from src.data.hetero_flatten import flatten_edges  # noqa: E402
from src.data.hgb_lp_negatives import (HGB_TEST_NEGATIVE_SEED, hgb_random_test_negatives,  # noqa: E402
                                       hgb_test_negatives, hgb_valid_negatives)
from src.data.large_lp import (LARGE_LP_SPLIT_SEED, LARGE_LP_TASKS, LARGE_LP_TEST_FRAC,  # noqa: E402
                               LARGE_LP_VAL_FRAC, LARGE_LP_VALID_NEG_SEED, held_out_fact_mask,
                               split_pairs, unordered_pairs)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO / "data"))
    parser.add_argument("--task", required=True, choices=sorted(LARGE_LP_TASKS))
    parser.add_argument("--report", required=True)
    parser.add_argument("--head-chunk", type=int, default=512)
    args = parser.parse_args()
    root = Path(args.root)
    task = LARGE_LP_TASKS[args.task]
    out_dir = large_lp_dir(root, task.key)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; this script never overwrites task files.")
    report: dict = {"task": task.key, "nc_dataset": task.nc_dataset, "relation": list(task.relation)}
    t0 = time.perf_counter()
    dataset = load_h2gb_dataset(task.nc_dataset, root)
    stored, loaded = dataset.get(0), dataset[0]
    report["load_seconds"] = round(time.perf_counter() - t0, 1)
    _log("loaded")

    # (1) facts per pair
    same_type = [r for r in loaded.edge_types if r[0] == task.node_type and r[2] == task.node_type]
    report["relations_between_endpoint_type"] = ["|".join(r) for r in same_type]
    report["stored_target_facts"] = unordered_pairs(stored[task.relation].edge_index)[1]
    pairs, loaded_stats = unordered_pairs(loaded[task.relation].edge_index)
    report["loaded_target_facts"] = loaded_stats
    if len(same_type) > 1:  # several relation types between the same node pair
        keys = [set(map(tuple, unordered_pairs(loaded[r].edge_index)[0].tolist())) for r in same_type]
        report["pairs_in_more_than_one_relation"] = len(set.intersection(*keys))
    else:
        report["pairs_in_more_than_one_relation"] = 0
    _log(f"pairs {pairs.shape[0]}")

    # (2) split by unordered pair
    train, valid, test = split_pairs(pairs, LARGE_LP_SPLIT_SEED, LARGE_LP_VAL_FRAC, LARGE_LP_TEST_FRAC)
    report["split"] = {"train": int(train.shape[0]), "valid": int(valid.shape[0]), "test": int(test.shape[0]),
                       "seed": LARGE_LP_SPLIT_SEED, "val_frac": LARGE_LP_VAL_FRAC, "test_frac": LARGE_LP_TEST_FRAC}

    # (3) message graph without held-out pairs (global ids as H2S-NN flattens the loaded graph)
    node_slices, offset = {}, 0
    for node_type in loaded.node_types:
        node_slices[str(node_type)] = (offset, offset + int(loaded[node_type].num_nodes))
        offset += int(loaded[node_type].num_nodes)
    num_nodes = offset
    edge_index, edge_type, relation_names, _ = flatten_edges(loaded, node_slices)
    base = node_slices[task.node_type][0]
    dst_range = node_slices[task.node_type]
    to_global = lambda part: torch.from_numpy(part.T.copy()) + base  # noqa: E731
    valid_g, test_g = to_global(valid), to_global(test)
    held = torch.cat([valid_g, test_g], dim=1)
    removed = held_out_fact_mask(edge_index, held)
    message = edge_index[:, ~removed]
    target_id = relation_names.index("__".join(task.relation))
    report["message_graph"] = {
        "facts_before": int(edge_index.size(1)), "facts_removed": int(removed.sum()),
        "target_facts_removed": int((removed & (edge_type == target_id)).sum()),
        "other_relation_facts_removed": int((removed & (edge_type != target_id)).sum()),
        "held_out_pairs": int(held.size(1)),
        "held_out_facts_left_either_orientation": int(held_out_fact_mask(message, held).sum())}
    _log(f"message graph {report['message_graph']}")
    del edge_index, edge_type, removed

    # (4) negatives
    t1 = time.perf_counter()
    both_ways = torch.cat([message, test_g], dim=1)

    def progress(done: int, total: int) -> None:
        if done % (args.head_chunk * 200) == 0:
            _log(f"2-hop heads {done}/{total}")

    two_hop = hgb_test_negatives(test_g, both_ways, valid_g, num_nodes, dst_range,
                                 head_chunk=args.head_chunk, progress=progress)
    report["two_hop_seconds"] = round(time.perf_counter() - t1, 1)
    _log("2-hop done")
    t2 = time.perf_counter()
    random_set = hgb_random_test_negatives(test_g, both_ways, valid_g, num_nodes, dst_range)
    report["random_seconds"] = round(time.perf_counter() - t2, 1)
    valid_neg = hgb_valid_negatives(valid_g, dst_range, LARGE_LP_VALID_NEG_SEED)
    heads = int(np.unique(test[:, 0]).size)
    report["negatives"] = {
        "test_2hop": int(two_hop.edge_index.size(1)), "test_random": int(random_set.size(1)),
        "valid": int(valid_neg.size(1)), "test_heads": heads,
        "heads_without_2hop_candidates": int(two_hop.heads_without_candidates.numel()),
        "share_of_heads_without": two_hop.heads_without_candidates.numel() / heads if heads else 0.0,
        "test_positives_without_2hop_negatives": int(two_hop.positives_of_heads_without),
        "seed_test": HGB_TEST_NEGATIVE_SEED, "seed_valid": LARGE_LP_VALID_NEG_SEED}

    # files (type-local ids)
    local = lambda t: (t - base).t().contiguous().numpy().astype(np.int64)  # noqa: E731
    out_dir.mkdir(parents=True)
    np.savez(out_dir / "split.npz", train=train, valid=valid, test=test)
    np.savez(out_dir / "negatives.npz", test_2hop=local(two_hop.edge_index), test_random=local(random_set),
             valid=local(valid_neg), heads_without_2hop=(two_hop.heads_without_candidates - base).numpy())
    processed = h2gb_nc_processed_file(root, task.nc_dataset)
    meta = {**{k: report[k] for k in ("task", "nc_dataset", "relation", "split", "negatives")},
            "source_processed_file": str(processed), "source_sha256": _sha256(processed),
            "ids": f"type-local ids of node type {task.node_type!r} in H2GB's loaded graph",
            "files_sha256": {name: _sha256(out_dir / name) for name in ("split.npz", "negatives.npz")}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    report["files"] = meta["files_sha256"]
    report["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
    _log(json.dumps(report)[:2000])


if __name__ == "__main__":
    main()
