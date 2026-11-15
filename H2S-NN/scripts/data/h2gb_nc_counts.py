"""Part M3b B2/B3 data check: count each H2GB node-classification dataset as H2GB loads it.

    python scripts/data/h2gb_nc_counts.py --root data --out <file.json> KEY [KEY ...]
    python scripts/data/h2gb_nc_counts.py --root data --out <file.json> --compare-mag

Per dataset (through `src/data/h2gb_nc.py`, i.e. H2GB's own loader): nodes per type, edges per relation of the
stored graph (the processed file, H2GB's stored orientation) and of the loaded graph (`dataset[0]`, after
H2GB's transform), feature dimension per type, target type, classes, split sizes, class counts per split,
sha256 and size of the processed file, load time. `--compare-mag` loads ogbn-mag and mag-year and states
what differs (graph, features, labels, masks). Reads only; writes only `--out`. Run in a SLURM job.
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

import torch  # noqa: E402

from src.data.data_layout import h2gb_nc_processed_file  # noqa: E402
from src.data.h2gb_nc import H2GB_NC_SPECS, load_h2gb_dataset  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _graph_counts(data) -> dict:
    nodes = {str(t): int(data[t].num_nodes) for t in data.node_types}
    feats = {str(t): (list(data[t].x.shape[1:]) if "x" in data[t] and data[t].x is not None else None)
             for t in data.node_types}
    edges = {"|".join(map(str, et)): int(data[et].edge_index.size(1)) for et in data.edge_types}
    return {"nodes": nodes, "total_nodes": sum(nodes.values()), "feature_dims": feats,
            "edges": edges, "total_edges": sum(edges.values()),
            "num_node_types": len(nodes), "num_edge_types": len(edges)}


def _label_counts(store) -> dict:
    y = store.y
    out: dict = {"y_shape": list(y.shape), "y_dtype": str(y.dtype)}
    labelled = y >= 0
    out["labelled"] = int(labelled.sum())
    out["unlabelled_(y=-1)"] = int((~labelled).sum())
    out["num_classes_(max_y+1)"] = int(y[labelled].max()) + 1 if out["labelled"] else 0
    out["classes_present"] = int(torch.unique(y[labelled]).numel())
    masks = {}
    for split in ("train", "val", "test"):
        mask = getattr(store, f"{split}_mask", None)
        if mask is None:
            masks[split] = None
            continue
        mask = mask.bool()
        counts = torch.bincount(y[mask & labelled], minlength=out["num_classes_(max_y+1)"])
        masks[split] = {"size": int(mask.sum()), "unlabelled_in_mask": int((mask & ~labelled).sum()),
                        "classes_present": int((counts > 0).sum()),
                        "class_counts": counts.tolist() if counts.numel() <= 16 else
                        {"min": int(counts.min()), "max": int(counts.max()),
                         "top5": torch.topk(counts, 5).values.tolist()}}
    names = ("train", "val", "test")
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma, mb = getattr(store, f"{a}_mask", None), getattr(store, f"{b}_mask", None)
            if ma is not None and mb is not None:
                overlaps[f"{a}&{b}"] = int((ma.bool() & mb.bool()).sum())
    out["splits"] = masks
    out["split_overlaps"] = overlaps
    return out


def count(key: str, root: Path) -> dict:
    spec = H2GB_NC_SPECS[key]
    processed = h2gb_nc_processed_file(root, key)
    record: dict = {"key": key, "h2gb_class": spec.class_name, "h2gb_loader": spec.loader,
                    "h2gb_name": spec.h2gb_name, "target_node_type": spec.target_node_type,
                    "label": spec.label, "split_source": spec.split,
                    "processed_file": str(processed.relative_to(root.resolve().parent)
                                          if processed.is_absolute() else processed),
                    "processed_bytes": processed.stat().st_size}
    start = time.perf_counter()
    dataset = load_h2gb_dataset(key, root)
    stored = dataset.get(0)
    loaded = dataset[0]
    record["load_seconds"] = round(time.perf_counter() - start, 1)
    record["transform"] = repr(dataset.transform)
    record["stored"] = _graph_counts(stored)
    record["loaded"] = _graph_counts(loaded)
    record["labels"] = _label_counts(loaded[spec.target_node_type])
    record["processed_sha256"] = _sha256(processed)
    for sibling in sorted(processed.parent.iterdir()):
        record.setdefault("processed_dir_files", {})[sibling.name] = sibling.stat().st_size
    return record


def compare_mag(root: Path) -> dict:
    a = load_h2gb_dataset("ogbn-mag", root).get(0)
    b = load_h2gb_dataset("mag-year", root).get(0)
    out: dict = {"node_types_equal": a.node_types == b.node_types,
                 "edge_types_equal": a.edge_types == b.edge_types}
    out["edge_index_equal"] = {"|".join(et): bool(torch.equal(a[et].edge_index, b[et].edge_index))
                               for et in a.edge_types if et in b.edge_types}
    out["x_equal"] = {t: bool(torch.equal(a[t].x, b[t].x)) for t in a.node_types
                      if "x" in a[t] and "x" in b[t]}
    pa, pb = a["paper"], b["paper"]
    out["y_equal"] = bool(torch.equal(pa.y, pb.y))
    out["masks_equal"] = {m: bool(torch.equal(getattr(pa, m), getattr(pb, m)))
                          for m in ("train_mask", "val_mask", "test_mask")}
    out["paper_attributes"] = {"ogbn-mag": sorted(pa.keys()), "mag-year": sorted(pb.keys())}
    out["y_classes"] = {"ogbn-mag": int(pa.y.max()) + 1, "mag-year": int(pb.y.max()) + 1}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*")
    parser.add_argument("--root", default=str(REPO / "data"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare-mag", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result: dict = {}
    for key in args.keys:
        result[key] = count(key, root)
        print(json.dumps({key: {k: result[key][k] for k in ("load_seconds", "processed_bytes")}}), flush=True)
    if args.compare_mag:
        result["mag_vs_mag_year"] = compare_mag(root)
    # peak resident memory of this process (Linux ru_maxrss is in KiB)
    result["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")
    print(json.dumps(result, indent=1)[:4000])


if __name__ == "__main__":
    main()
