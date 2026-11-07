"""U1: dump everything a baseline sees from build_bundle (node: ACM/DBLP/IMDB, graph: MUTAG),
so the dump before and after the default flip can be compared bit-for-bit.
  python u1_bundle_dump.py dump <out.pt>        (cwd = H2S-NN)
  python u1_bundle_dump.py compare <a.pt> <b.pt>"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))


def dump(out: Path) -> None:
    from src.baselines.bundles import build_bundle
    rec = {}
    for name in ("acm", "dblp", "imdb"):
        d = build_bundle("node_classification", name).data
        for f in ("x", "y", "node_type", "edge_index", "edge_type", "target_y", "train_mask",
                  "val_mask", "test_mask", "target_indices", "metrics"):
            rec[f"{name}/{f}"] = getattr(d, f)
        rec[f"{name}/relation_names"] = list(d.relation_names)
        for g, grp in enumerate(d.face_groups):
            for f in ("face_node_index", "face_edge_index", "relation_index", "face_edge_sign"):
                rec[f"{name}/group{g}/{f}"] = getattr(grp, f)
            rec[f"{name}/group{g}/name"] = grp.template_name
    fold = build_bundle("graph_classification", "mutag").fold
    for split in ("train", "val", "test"):
        ds = getattr(fold, f"{split}_loader").dataset
        rec[f"mutag/{split}/edge_index"] = [x.edge_index for x in ds]
        rec[f"mutag/{split}/edge_type"] = [x.edge_type for x in ds]
    torch.save(rec, out)
    print(f"dumped {len(rec)} entries -> {out}")


def compare(a: Path, b: Path) -> None:
    x, y = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
    bad = sorted(set(x) ^ set(y))
    for k in sorted(set(x) & set(y)):
        u, v = x[k], y[k]
        if isinstance(u, list) and u and isinstance(u[0], torch.Tensor):
            same = len(u) == len(v) and all(torch.equal(p, q) for p, q in zip(u, v))
        elif isinstance(u, torch.Tensor):
            same = u.dtype == v.dtype and u.shape == v.shape and torch.equal(u, v)
        else:
            same = u == v
        if not same:
            bad.append(k)
    print(f"compared {len(set(x) & set(y))} entries; mismatched or missing: {len(bad)}")
    for k in bad[:20]:
        print("  MISMATCH", k)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    dump(Path(sys.argv[2])) if sys.argv[1] == "dump" else compare(Path(sys.argv[2]), Path(sys.argv[3]))
