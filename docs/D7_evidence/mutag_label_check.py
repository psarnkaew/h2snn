"""Is MUTAG's raw edge-label file symmetric, and does the pipeline align it with PyG's edge order?"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2] / "H2S-NN"
sys.path.insert(0, str(REPO))
from torch_geometric.datasets import TUDataset  # noqa: E402

raw = REPO / "data" / "tud" / "MUTAG" / "raw"
pairs = [tuple(int(v) for v in line.replace(",", " ").split()[:2])
         for line in (raw / "MUTAG_A.txt").read_text().splitlines() if line.strip()]
labels = [int(line) for line in (raw / "MUTAG_edge_labels.txt").read_text().splitlines() if line.strip()]
label_of = dict(zip(pairs, labels))
asym = sum(1 for (u, v), lab in label_of.items() if label_of.get((v, u)) != lab)
print("raw edges", len(pairs), "raw (u,v) whose reverse has a different/missing label:", asym)

ds = TUDataset(root=str(REPO / "data" / "tud"), name="MUTAG")
offset, in_raw_order = 0, 0
total = 0
for g in range(len(ds)):
    data = ds[g]
    ei = (data.edge_index + offset + 1).t().tolist()  # raw ids are 1-based, global
    raw_slice = pairs[total:total + len(ei)]
    in_raw_order += sum(1 for a, b in zip(ei, raw_slice) if tuple(a) == b)
    pyg_lab = data.edge_attr.argmax(-1).tolist()
    total += len(ei)
    offset += data.num_nodes
print("PyG edges in the same position as raw A.txt:", in_raw_order, "of", total)
pyg_vs_raw = 0
offset = 0
for g in range(len(ds)):
    data = ds[g]
    for (a, b), lab in zip((data.edge_index + offset + 1).t().tolist(), data.edge_attr.argmax(-1).tolist()):
        pyg_vs_raw += int(label_of[(a, b)] != lab)
    offset += data.num_nodes
print("PyG edge_attr label vs raw label of the same (u,v):", pyg_vs_raw, "mismatches")
