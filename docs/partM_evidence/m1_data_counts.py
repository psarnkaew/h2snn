"""Part M, M1 gate: every local dataset loads from the new layout (data/NC, data/LP, data/GC)
through the project's loaders, and its counts equal the counts recorded before the move.

Recorded sources (read, not recomputed):
- NC files and counts: docs/lp_evidence/inventory.json (Part J J1.1: md5 of every HGB file, node
  counts per type, distinct stored edges per relation, facts before/after tying);
  docs/WISP_evidence/h0_data_facts.json (num_nodes, facts); labels:
  docs/heterophily_evidence/results.md (labelled targets and class counts).
- LP: docs/H2S-NN_partJ_report.md §5 and §10 (train/valid/test of the derived tasks; freebase_bb
  pairs), docs/lp_evidence/j42_record.md (target rows of the PubMed/LastFM validation files),
  docs/lp_evidence/j54_write_valid_files_output.txt (the target-only files of spec v1.9 F13, J5.4,
  which superseded the J4.2 carve files of j42_record.md), spec v1.15 §6 (link.dat.test rows 8,528
  and 18,567).
- GC: tests/test_paths_and_facts.py T4 (stored edges before/after tying, recorded in Part G) and
  spec v1.15 §6 (test sets 20 / 112 / 60 / 411 graphs under the 80/10/10 split).

Run from H2S-NN/:  python ../docs/partM_evidence/m1_data_counts.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "H2S-NN"
DOCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.data.data_layout import (REPO_DATA_ROOT, gc_root, hgb_nc_raw_dir,  # noqa: E402
                                  lp_derived_root, lp_root, nc_root)

failures: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got} recorded {want}")
    if not ok:
        failures.append(name)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- NC (HGB 4)
inventory = json.loads((DOCS / "lp_evidence" / "inventory.json").read_text(encoding="utf-8"))
h0 = json.loads((DOCS / "WISP_evidence" / "h0_data_facts.json").read_text(encoding="utf-8"))
LABELS = {  # heterophily_evidence/results.md: labelled targets / class counts
    "acm": (3025, [994, 970, 1061]),
    "dblp": (4057, [1197, 745, 1109, 1006]),
    "imdb": (4573, [1090, 1384, 1837, 1135, 2517]),
    "freebase": (7954, [429, 3886, 1748, 185, 962, 359, 385]),
}
TARGET = {"acm": "paper", "dblp": "author", "imdb": "movie", "freebase": "book"}

from torch_geometric.datasets import HGBDataset  # noqa: E402

from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import _load_hetero_data  # noqa: E402
from src.data.fact_layer import build_fact_layer  # noqa: E402
from src.data.hetero_flatten import flatten_edges, stack_node_features  # noqa: E402

for key, record in inventory.items():
    raw_dir = hgb_nc_raw_dir(REPO_DATA_ROOT, key)
    for name, digest in record["files_md5"].items():
        check(f"NC {key} {name} md5", md5(raw_dir / name), digest)
    if key in ("acm", "dblp", "imdb"):  # the H2S-NN node-classification loader
        data = _load_hetero_data(TrainConfig.from_dict(
            {"data": {"dataset_name": key, "root": str(REPO_DATA_ROOT)}}))
    else:  # Freebase: no H2S-NN NC loader; PyG HGBDataset as the heterophily script loads it
        data = HGBDataset(root=str(nc_root(REPO_DATA_ROOT)), name="Freebase")[0]
    check(f"NC {key} nodes per type",
          {t: int(data[t].num_nodes) for t in data.node_types}, record["node_types"])
    distinct = sorted(int(torch.unique(data[et].edge_index, dim=1).size(1))
                      for et in data.edge_types)
    check(f"NC {key} distinct stored edges per relation (sorted)", distinct,
          sorted(r["stored_distinct_edges"] for r in record["relations"]))
    x, _, _, slices, _ = stack_node_features(data)
    edge_index, edge_type, names, _ = flatten_edges(data, slices)
    layer = build_fact_layer(edge_index, edge_type, names, x.size(0), "tied")
    check(f"NC {key} facts before/after tying", (layer.facts_before, layer.facts_after),
          (record["facts_before"], record["facts_after"]))
    if key in h0:
        check(f"NC {key} num_nodes/facts (h0)", (int(x.size(0)), layer.facts_after),
              (h0[key]["num_nodes"], h0[key]["facts"]))
    store = data[TARGET[key]]
    y = store.y
    labelled = store.train_mask | store.test_mask
    if y.dim() == 2:  # IMDB multi-hot
        labelled = labelled & (y.sum(dim=1) > 0)
        counts = [int(v) for v in y[labelled].sum(dim=0).tolist()]
    else:
        counter = Counter(int(v) for v in y[labelled].tolist())
        counts = [counter[c] for c in range(len(LABELS[key][1]))]
    check(f"NC {key} labelled targets / class counts", (int(labelled.sum()), counts), LABELS[key])

# ----------------------------------------------------------------------------- LP (6 local)
from src.data.link_prediction_dataset import _load_local_hgb_link_data  # noqa: E402

LP_TASKS = {  # target, recorded (train, valid, test) of the target relation
    "acm_pa": (("paper", "rel_2", "author"), (8059, 895, 995)),
    "dblp_ap": (("author", "rel_0", "paper"), (15913, 1768, 1964)),
    "imdb_ma": (("movie", "rel_2", "actor"), (11971, 1330, 1478)),
}
for name, (target, want) in LP_TASKS.items():
    folder = lp_derived_root(REPO_DATA_ROOT) / name
    meta = json.loads((folder / "split_meta.json").read_text(encoding="utf-8"))
    for file in ("link.dat", "link.dat.test", "link.dat.valid"):
        check(f"LP {name} {file} md5", md5(folder / file), meta["output_md5"][file])
    split = _load_local_hgb_link_data(TrainConfig.from_dict({"data": {
        "dataset_name": name, "task_type": "link_prediction", "root": str(REPO_DATA_ROOT),
        "target_relation": list(target)}}))
    got = tuple(int(p[target].size(1)) for p in (split.train_edge_index_by_relation,
                                                 split.valid_edge_index_by_relation,
                                                 split.test_edge_index_by_relation))
    check(f"LP {name} train/valid/test", got, want)

# freebase_bb: unordered pairs (Part J §10 J4.3: 103,303 / 11,478 / 12,754)
folder = lp_derived_root(REPO_DATA_ROOT) / "freebase_bb"
meta = json.loads((folder / "split_meta.json").read_text(encoding="utf-8"))
for file in ("link.dat", "link.dat.test", "link.dat.valid"):
    check(f"LP freebase_bb {file} md5", md5(folder / file), meta["output_md5"][file])
split = _load_local_hgb_link_data(TrainConfig.from_dict({"data": {
    "dataset_name": "freebase_bb", "task_type": "link_prediction", "root": str(REPO_DATA_ROOT),
    "target_relation": ["book", "rel_0", "book"]}}))
target = ("book", "rel_0", "book")


def pairs(index: torch.Tensor) -> set[tuple[int, int]]:
    return {(min(u, v), max(u, v)) for u, v in index.t().tolist() if u != v}


got = (len(pairs(split.train_edge_index_by_relation[target])),
       len(pairs(split.valid_edge_index_by_relation[target])),
       len(pairs(split.test_edge_index_by_relation[target])))
check("LP freebase_bb unordered pairs train/valid/test", got, (103303, 11478, 12754))

for name, folder_name, target, valid_md5, valid_rows, test_rows in (
        ("pubmed", "PubMed", ("disease", "rel_2", "disease"),
         "3d2b5114128e4f1208978b822ae8e22e", 1942, 8528),
        ("lastfm", "LastFM", ("user", "rel_0", "artist"),
         "8ffd3a6821fe7b2fad4f6afbb60bf19b", 7301, 18567)):
    folder = lp_root(REPO_DATA_ROOT) / folder_name
    check(f"LP {name} link.dat.valid md5", md5(folder / "link.dat.valid"), valid_md5)
    rows = sum(1 for _ in (folder / "link.dat.test").open("rb"))
    check(f"LP {name} link.dat.test rows", rows, test_rows)
    valid_target = sum(1 for line in (folder / "link.dat.valid").open("rb")
                       if line.split(b"\t")[2] == target[1].split("_")[1].encode())
    check(f"LP {name} link.dat.valid target rows", valid_target, valid_rows)
    split = _load_local_hgb_link_data(TrainConfig.from_dict(
        {"data": {"dataset_name": name, "root": str(REPO_DATA_ROOT)}}))
    print(f"INFO LP {name} loader train/valid/test of {target}: "
          f"{tuple(int(p[target].size(1)) for p in (split.train_edge_index_by_relation, split.valid_edge_index_by_relation, split.test_edge_index_by_relation))}")

# ----------------------------------------------------------------------------- GC (TU 4)
from torch_geometric.datasets import TUDataset  # noqa: E402

from src.data.graph_classification_dataset import _load_tu_type_tensors, _tie_tu_facts  # noqa: E402

TU = {"MUTAG": (7442, 3721, 20), "PROTEINS": (162088, 81044, 112),
      "ENZYMES": (74564, 37282, 60), "NCI1": (265506, 132753, 411)}
for tud, (before, after, test_size) in TU.items():
    dataset = TUDataset(root=str(gc_root(REPO_DATA_ROOT)), name=tud)
    data_list = [dataset[i].clone() for i in range(len(dataset))]
    _, edge_types, _, num_relations = _load_tu_type_tensors(REPO_DATA_ROOT, tud, data_list)
    layer, _, _ = _tie_tu_facts(data_list, edge_types, num_relations, "tied")
    check(f"GC {tud} stored edges before/after tying", (layer.facts_before, layer.facts_after),
          (before, after))
    n = len(dataset)
    # HetSheaf's split: train = int(0.8 n), val = int(0.1 n), test = the rest (spec §6 sizes).
    check(f"GC {tud} test-set size of the 80/10/10 split (n = {n})",
          n - int(0.8 * n) - int(0.1 * n), test_size)

print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
