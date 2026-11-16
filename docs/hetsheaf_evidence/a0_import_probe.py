"""A0 (H-1) probe: can HetSheaf's model code be imported and run in h2snn-b?

Three checks, each printed with its outcome:
  1. `import hetsheaf.models.sheaf_gnn` the normal way (runs hetsheaf/__init__.py).
  2. The same with the package __init__ chain bypassed: `hetsheaf`, `hetsheaf.models`, `hetsheaf.graph_classification`
     registered as empty package modules whose __path__ is the vendor folder (no vendor file changed, nothing installed);
     then one forward + backward per map family on a synthetic heterogeneous graph (to_homogeneous type ids), and the
     non-late_token GC poolings (PyG globals, `graph_classification/pooling.py`).
  3. SheafPool (`LateTokenSheafReadoutTransport`, `graph_classification/sheaf_graph_classifier.py:33`): import as is
     (needs `lightning`), then with `lightning.pytorch` aliased to the installed `pytorch_lightning` (probe only).
Run in a job: python docs/hetsheaf_evidence/a0_import_probe.py
"""
import importlib
import sys
import traceback
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "H2S-NN" / "vendor" / "Heterogeneous-Sheaf-Neural-Networks"
PKG = ROOT / "hetsheaf"


def try_(label, fn):
    try:
        out = fn()
        print(f"[OK]   {label}" + (f": {out}" if out is not None else ""), flush=True)
        return True
    except Exception as exc:  # probe: report and continue
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc(limit=3)
        return False


def clear_hetsheaf():
    for name in [m for m in sys.modules if m == "hetsheaf" or m.startswith("hetsheaf.")]:
        del sys.modules[name]


def stub_package(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def synthetic_hetero():
    from torch_geometric.data import HeteroData
    g = torch.Generator().manual_seed(0)
    data = HeteroData()
    data["paper"].x = torch.rand(6, 5, generator=g)
    data["author"].x = torch.rand(4, 3, generator=g)
    data["paper", "to", "author"].edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [0, 1, 2, 3, 0, 1, 3]])
    data["author", "to", "paper"].edge_index = data["paper", "to", "author"].edge_index.flip(0)
    data["paper", "cites", "paper"].edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    data["paper", "cites", "paper"].edge_index = torch.cat(
        [data["paper", "cites", "paper"].edge_index, data["paper", "cites", "paper"].edge_index.flip(0)], 1)
    return data.to_homogeneous()


def main():
    sys.path.insert(0, str(ROOT))
    print("torch", torch.__version__)

    print("== 1. normal import")
    try_("import hetsheaf.models.sheaf_gnn (package __init__ chain)",
         lambda: importlib.import_module("hetsheaf.models.sheaf_gnn") and None)
    clear_hetsheaf()

    print("== 2. package __init__ chain bypassed")
    stub_package("hetsheaf", PKG)
    stub_package("hetsheaf.models", PKG / "models")
    stub_package("hetsheaf.graph_classification", PKG / "graph_classification")
    if not try_("import hetsheaf.models.sheaf_gnn", lambda: importlib.import_module("hetsheaf.models.sheaf_gnn") and None):
        return
    sg = sys.modules["hetsheaf.models.sheaf_gnn"]
    from hetsheaf.models.sheaf_gnn.config import SheafModelArguments
    data = synthetic_hetero()
    print("synthetic: N", data.num_nodes, "E", data.edge_index.size(1), "node types", int(data.node_type.max()) + 1,
          "edge types", int(data.edge_type.max()) + 1, "x", tuple(data.x.shape))
    fams = {"diag": sg.DiscreteDiagSheafDiffusion, "bundle": sg.DiscreteBundleSheafDiffusion,
            "general": sg.DiscreteGeneralSheafDiffusion}
    learners = {"diag": ["local_concat", "type_concat", "type_ensemble", "types_only"],
                "bundle": ["type_concat", "attention_type_concat"],
                "general": ["type_concat", "attention_type_concat"]}
    for fam, cls in fams.items():
        for learner in learners[fam]:
            def run(cls=cls, learner=learner):
                torch.manual_seed(0)
                args = SheafModelArguments(d=3, layers=2, hidden_channels=4, device="cpu", graph_size=data.num_nodes,
                                           input_dim=data.x.size(1), output_dim=8,
                                           num_edge_types=int(data.edge_type.max()) + 1,
                                           num_node_types=int(data.node_type.max()) + 1)
                model = cls(data.edge_index, args, learner)
                out, maps = model(data)
                out.sum().backward()
                ok = torch.isfinite(out).all().item()
                ngrad = sum(p.grad is not None for p in model.parameters())
                return f"out {tuple(out.shape)} finite={ok} params_with_grad={ngrad}/{len(list(model.parameters()))}"
            try_(f"forward+backward {fam} / {learner}", run)

    def pools():
        from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
        from hetsheaf.graph_classification.pooling import AttentionPooling
        x = torch.rand(10, 6)
        b = torch.tensor([0] * 5 + [1] * 5)
        return [tuple(f(x, b).shape) for f in (global_mean_pool, global_max_pool, global_add_pool,
                                                AttentionPooling(6, heads=2))]
    try_("GC poolings mean/max/add/attention (pooling.py)", pools)

    print("== 3. SheafPool (late_token)")
    try_("import hetsheaf.graph_classification.sheaf_graph_classifier as is",
         lambda: importlib.import_module("hetsheaf.graph_classification.sheaf_graph_classifier") and None)
    sys.modules.pop("hetsheaf.graph_classification.sheaf_graph_classifier", None)
    sys.modules.pop("hetsheaf.graph_classification.graph_classifier", None)

    def alias_and_import():
        import pytorch_lightning
        import pytorch_lightning.utilities.types as plt_types
        light = types.ModuleType("lightning")
        light.pytorch = pytorch_lightning
        sys.modules["lightning"] = light
        sys.modules["lightning.pytorch"] = pytorch_lightning
        sys.modules["lightning.pytorch.utilities"] = pytorch_lightning.utilities
        sys.modules["lightning.pytorch.utilities.types"] = plt_types
        mod = importlib.import_module("hetsheaf.graph_classification.sheaf_graph_classifier")
        pool = mod.LateTokenSheafReadoutTransport(stalk_dim=5, channel_dim=4)
        h = torch.rand(10, 20, requires_grad=True)
        out = pool(h, torch.tensor([0] * 5 + [1] * 5))
        out.sum().backward()
        return f"pytorch_lightning {pytorch_lightning.__version__}; out {tuple(out.shape)} finite={torch.isfinite(out).all().item()}"
    try_("SheafPool with lightning.pytorch aliased to pytorch_lightning (probe only)", alias_and_import)


if __name__ == "__main__":
    main()
