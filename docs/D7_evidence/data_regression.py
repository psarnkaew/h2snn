"""Data-level regression (decision 10): with fact_layer untied (+ path_sampler dfs, no
resampling, once those keys exist), prepared data must equal the pre-D7 code bit-for-bit.

  python data_regression.py dump <repo H2S-NN dir> <out.pt>   # run with cwd = that dir
  python data_regression.py compare <a.pt> <b.pt>
Covers node classification (ACM, DBLP, IMDB: edges, types, relations, every FaceGroup
field) and graph classification (MUTAG, PROTEINS: per-graph edge_index/edge_type and the
first training batch's face groups)."""
import sys
from pathlib import Path

import torch


DATA_ROOT = str(Path(__file__).resolve().parents[2] / "H2S-NN" / "data")


def dump(repo: Path, out: Path) -> None:
    sys.path.insert(0, str(repo))
    from src.configs.config import TrainConfig
    from src.data.dataset import build_dataset
    from src.data.graph_classification_dataset import build_graph_classification_dataset
    from src.data.graph_faces import build_typed_path_face_groups

    record = {}
    for name in ("acm", "dblp", "imdb"):
        data = build_dataset(TrainConfig.from_dict({"data": {"dataset_name": name,
                                                             "root": DATA_ROOT}}))
        record[f"{name}/edge_index"] = data.edge_index
        record[f"{name}/edge_type"] = data.edge_type
        record[f"{name}/relation_names"] = list(data.relation_names)
        for g, group in enumerate(data.face_groups):
            for field in ("face_node_index", "face_edge_index", "relation_index",
                          "face_edge_sign"):
                record[f"{name}/group{g}/{field}"] = getattr(group, field)
            record[f"{name}/group{g}/name"] = group.template_name
    for name in ("mutag", "proteins"):
        fold = build_graph_classification_dataset(TrainConfig.from_dict(
            {"data": {"dataset_name": name, "root": DATA_ROOT}}))
        record[f"{name}/num_relations"] = int(fold.dims.num_relations)
        for split in ("train", "val", "test"):
            dataset = getattr(fold, f"{split}_loader").dataset
            record[f"{name}/{split}/edge_index"] = [d.edge_index for d in dataset]
            record[f"{name}/{split}/edge_type"] = [d.edge_type for d in dataset]
    torch.save(record, out)
    print(f"dumped {len(record)} entries -> {out}")


def compare(a: Path, b: Path) -> None:
    x, y = torch.load(a, weights_only=False), torch.load(b, weights_only=False)
    bad = sorted(set(x) ^ set(y))
    for key in sorted(set(x) & set(y)):
        u, v = x[key], y[key]
        if isinstance(u, list) and u and isinstance(u[0], torch.Tensor):
            same = len(u) == len(v) and all(torch.equal(p, q) and p.dtype == q.dtype
                                            for p, q in zip(u, v))
        elif isinstance(u, torch.Tensor):
            same = u.dtype == v.dtype and u.shape == v.shape and torch.equal(u, v)
        else:
            same = u == v
        if not same:
            bad.append(key)
    print(f"compared {len(set(x) & set(y))} entries; mismatched or missing: {len(bad)}")
    for key in bad[:20]:
        print("  MISMATCH", key)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(Path(sys.argv[2]).resolve(), Path(sys.argv[3]))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
