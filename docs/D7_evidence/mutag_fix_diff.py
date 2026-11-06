"""MUTAG loader fix: graphs / edges whose edge type changes, old (raw position, git HEAD
before the fix) vs new (PyG edge_attr), and rho of the new types.
Run from H2S-NN/: python ../docs/D7_evidence/mutag_fix_diff.py"""
import subprocess
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_geometric.datasets import TUDataset

from src.data.graph_classification_dataset import _TUD_NAMES, _load_tu_type_tensors as new_fn

old_src = subprocess.run(["git", "show", "6bb779d:H2S-NN/src/data/graph_classification_dataset.py"],
                         capture_output=True, text=True, check=True).stdout
old_mod = types.ModuleType("old_gc")
sys.modules["old_gc"] = old_mod
exec(compile(old_src, "old_gc", "exec"), old_mod.__dict__)

import inventory_rho

root = Path("data")
for name in ("mutag", "proteins", "enzymes", "nci1"):
    tud = _TUD_NAMES[name]
    ds = TUDataset(root=str(root / "tud"), name=tud)
    dl = [ds[i] for i in range(len(ds))]
    _, old_e, _, old_r = old_mod._load_tu_type_tensors(root, tud, dl)
    _, new_e, _, new_r = new_fn(root, tud, dl)
    graphs = sum(int(not torch.equal(a, b)) for a, b in zip(old_e, new_e))
    edges = sum(int((a != b).sum()) for a, b in zip(old_e, new_e))
    total = sum(int(a.numel()) for a in new_e)
    print(f"{tud}: relations old={old_r} new={new_r}; graphs changed {graphs}/{len(dl)}; "
          f"edges changed {edges}/{total}")
    if name == "mutag":
        offset, ei = 0, []
        for d in dl:
            ei.append(d.edge_index + offset)
            offset += int(d.num_nodes)
        res = inventory_rho.analyse(tud, torch.cat(ei, 1), torch.cat(new_e),
                                    [f"edge_label_{r}" for r in range(new_r)],
                                    [("node", "node")] * new_r, offset)
        print(f"  rho with fixed types: self_inverse={res['self_inverse']} untied={res['untied']} "
              f"facts {res['facts_before']} -> {res['facts_after']}")
