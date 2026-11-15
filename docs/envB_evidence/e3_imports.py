"""E3: import matrix, one fresh interpreter per library, warnings visible (-W default); then one tiny functional call per extension.
usage (in the env under test): python docs/envB_evidence/e3_imports.py
"""
import subprocess
import sys

LIBS = [("torch", "torch"), ("torch_geometric", "torch_geometric"), ("torch_scatter", "torch_scatter"),
        ("torch_sparse", "torch_sparse"), ("torch_cluster", "torch_cluster"), ("dgl", "dgl"),
        ("torchdata", "torchdata"), ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
        ("sklearn", "sklearn"), ("ogb", "ogb"), ("pytorch_lightning", "pytorch_lightning"),
        ("wandb", "wandb"), ("torchmetrics", "torchmetrics"), ("umap", "umap"),
        ("performer_pytorch", "performer_pytorch")]

CALLS = {
 "torch cuda": "import torch; print('cuda_available', torch.cuda.is_available()); "
               "x=torch.arange(4.,device='cuda'); print('cuda sum', (x*x).sum().item(), torch.cuda.get_device_name(0))",
 "torch_scatter": "import torch, torch_scatter as s; print(s.scatter_add(torch.tensor([1.,2.,3.]), torch.tensor([0,1,0])))",
 "torch_sparse": "import torch; from torch_sparse import SparseTensor; a=SparseTensor(row=torch.tensor([0,1]),col=torch.tensor([1,0]),value=torch.tensor([1.,2.]),sparse_sizes=(2,2)); print(a.matmul(torch.eye(2)))",
 "torch_cluster": "import torch, torch_cluster as c; x=torch.tensor([[0.,0.],[1.,0.],[5.,5.]]); print(c.radius_graph(x, r=2.0)); print(c.fps(x, ratio=0.67))",
 "torch_geometric": "import torch; from torch_geometric.nn import GCNConv; from torch_geometric.data import Data; d=Data(x=torch.randn(3,4),edge_index=torch.tensor([[0,1,2],[1,2,0]])); print(GCNConv(4,2)(d.x,d.edge_index).shape)",
 "dgl message passing": "import torch, dgl, dgl.function as fn; g=dgl.graph(([0,1,2],[1,2,0])); g.ndata['h']=torch.ones(3,2); g.update_all(fn.copy_u('h','m'), fn.sum('m','h2')); print(g.ndata['h2'])",
 "dgl graphbolt": "import dgl.graphbolt as gb; print('graphbolt', gb)",
 "torchdata datapipes": "from torchdata.datapipes.iter import IterableWrapper; print(list(IterableWrapper(range(3))))",
 "pytorch_lightning+torchmetrics": "import torch, torchmetrics as tm; m=tm.Accuracy(task='multiclass',num_classes=3); print(m(torch.tensor([0,1,2]),torch.tensor([0,1,1])))",
 "performer_pytorch": "import torch; from performer_pytorch import Performer; print(Performer(dim=8,depth=1,heads=2,dim_head=4)(torch.randn(1,5,8)).shape)",
 "ogb": "from ogb.nodeproppred import Evaluator; print(Evaluator('ogbn-arxiv'))",
}

def run(code):
    p = subprocess.run([sys.executable, "-W", "default", "-c", code], capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

print("python", sys.version.split()[0], "executable", sys.executable)
print("\n=== import matrix (fresh interpreter each, python -W default) ===")
for label, mod in LIBS:
    rc, out, err = run(f"import {mod}; print(getattr({mod}, '__version__', 'no __version__'))")
    print(f"--- {label}: rc={rc} version={out!r}")
    if err:
        print("    stderr:\n" + "\n".join("      " + l for l in err.splitlines()))
print("\n=== functional calls ===")
for label, code in CALLS.items():
    rc, out, err = run(code)
    print(f"--- {label}: rc={rc}")
    if out:
        print("    stdout:\n" + "\n".join("      " + l for l in out.splitlines()))
    if err:
        print("    stderr:\n" + "\n".join("      " + l for l in err.splitlines()[-12:]))
