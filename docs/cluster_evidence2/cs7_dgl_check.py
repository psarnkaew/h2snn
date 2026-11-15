"""CS7-1c: DGL 2.4.0 known-answer checks on CUDA (cluster-setup2 card).

Run from H2S-NN/ with `python -W default ../docs/cluster_evidence2/cs7_dgl_check.py`.
Tolerance rtol=atol=1e-6 is fixed by the card (float32 sums of <= 6 terms). Exit 1 if any check fails.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path.cwd()))

import dgl  # noqa: E402
import dgl.function as fn  # noqa: E402

TOL = dict(rtol=1e-6, atol=1e-6)
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cuda available", torch.cuda.is_available())
print("numpy", np.__version__, "scipy", scipy.__version__)
print("dgl.__version__", dgl.__version__, "| importlib.metadata.version('dgl')", importlib.metadata.version("dgl"))
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))

# (v) the two optional subpackages
for mod in ("dgl.graphbolt", "dgl.sparse"):
    try:
        __import__(mod)
        record(f"(v) import {mod}", True, "ok")
    except Exception as err:  # reported, the run continues so every check is listed
        record(f"(v) import {mod}", False, f"{type(err).__name__}: {err}")

sos = sorted({ln.split()[-1] for ln in Path("/proc/self/maps").read_text().splitlines()
              if ".so" in ln and ("dgl" in ln or "graphbolt" in ln or "tensoradapter" in ln)})
print("loaded DGL shared objects:")
for s in sos:
    print("   ", s)
for lib in ("libgraphbolt_pytorch_2.4.1.so", "libdgl_sparse_pytorch_2.4.1.so", "libtensoradapter_pytorch_2.4.1.so"):
    record(f"loaded {lib}", any(s.endswith(lib) for s in sos), "in /proc/self/maps" if any(s.endswith(lib) for s in sos) else "NOT loaded")

dev = torch.device("cuda")
gen = torch.Generator().manual_seed(0)

# (i) fixed 6-node directed graph
src = torch.tensor([0, 0, 1, 2, 3, 3, 4, 5])
dst = torch.tensor([1, 2, 2, 3, 4, 5, 5, 0])
x = torch.randn(6, 4, generator=gen)
g = dgl.graph((src, dst), num_nodes=6).to(dev)
g.ndata["h"] = x.to(dev)
g.update_all(fn.copy_u("h", "m"), fn.sum("m", "s"))
ref_sum = torch.zeros(6, 4).index_add_(0, dst, x[src])
got = g.ndata["s"].cpu()
record("(i) update_all(copy_u, sum)", torch.allclose(got, ref_sum, **TOL), f"max|diff|={maxdiff(got, ref_sum):.3e}")
g.update_all(fn.copy_u("h", "m"), fn.mean("m", "a"))
deg = torch.zeros(6).index_add_(0, dst, torch.ones(len(dst)))
ref_mean = ref_sum / deg.clamp(min=1).unsqueeze(1)
got = g.ndata["a"].cpu()
record("(i) update_all(copy_u, mean)", torch.allclose(got, ref_mean, **TOL), f"max|diff|={maxdiff(got, ref_mean):.3e}")

# (ii) 3-type heterograph, multi_update_all with sum
rels = {
    ("author", "writes", "paper"): (torch.tensor([0, 1, 1, 2]), torch.tensor([0, 0, 1, 2])),
    ("paper", "cites", "paper"): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])),
    ("venue", "publishes", "paper"): (torch.tensor([0, 1, 0]), torch.tensor([0, 1, 2])),
}
n = {"author": 3, "paper": 3, "venue": 2}
feats = {t: torch.randn(k, 4, generator=gen) for t, k in n.items()}
hg = dgl.heterograph(rels, num_nodes_dict=n).to(dev)
for t in n:
    hg.nodes[t].data["h"] = feats[t].to(dev)
hg.multi_update_all({et: (fn.copy_u("h", "m"), fn.sum("m", "s")) for et in hg.canonical_etypes}, "sum")
ref = torch.zeros(3, 4)
for (st, _, _), (s, d) in rels.items():
    ref.index_add_(0, d, feats[st][s])
got = hg.nodes["paper"].data["s"].cpu()
record("(ii) multi_update_all sum (3 node types)", torch.allclose(got, ref, **TOL), f"max|diff|={maxdiff(got, ref):.3e}")

# (iii) GraphConv (identity weight, no bias, norm='none') and GATConv, forward and backward
conv = dgl.nn.GraphConv(4, 4, norm="none", weight=True, bias=False).to(dev)
with torch.no_grad():
    conv.weight.copy_(torch.eye(4))
xin = x.clone().to(dev).requires_grad_(True)
out = conv(g, xin)
out.sum().backward()
finite = bool(torch.isfinite(out).all() and torch.isfinite(xin.grad).all() and torch.isfinite(conv.weight.grad).all())
record("(iii) GraphConv == plain sum", torch.allclose(out.detach().cpu(), ref_sum, **TOL),
       f"max|diff|={maxdiff(out.detach().cpu(), ref_sum):.3e}")
record("(iii) GraphConv finite outputs and gradients", finite, f"finite={finite}")
gat = dgl.nn.GATConv(4, 4, num_heads=2).to(dev)
xin2 = x.clone().to(dev).requires_grad_(True)
out2 = gat(g, xin2)
out2.sum().backward()
finite2 = bool(torch.isfinite(out2).all() and torch.isfinite(xin2.grad).all()
               and all(torch.isfinite(p.grad).all() for p in gat.parameters() if p.grad is not None))
record("(iii) GATConv finite outputs and gradients", finite2, f"out shape {tuple(out2.shape)}, finite={finite2}")

# (iv) graph from numpy and from scipy.sparse equals the graph from torch tensors
g_t = dgl.graph((src, dst), num_nodes=6)
g_np = dgl.graph((src.numpy(), dst.numpy()), num_nodes=6)
coo = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(6, 6))
g_sp = dgl.from_scipy(coo)
for name, other in (("numpy", g_np), ("scipy.sparse", g_sp)):
    eu, ev = other.edges()
    tu, tv = g_t.edges()
    ok = torch.equal(eu, tu) and torch.equal(ev, tv) and other.num_nodes() == 6
    record(f"(iv) graph from {name} == from torch", ok, f"edges equal={ok}")

# the project's own DGL CUDA report
from src.baselines.runtime.dependencies import dgl_cuda_report  # noqa: E402

uses_gpu, why = dgl_cuda_report()
print(f"dependencies.dgl_cuda_report() -> ({uses_gpu}, {why!r})")
record("dependencies.dgl_cuda_report uses GPU", uses_gpu, why)

n_fail = sum(not ok for _, ok, _ in results)
print(f"SUMMARY checks={len(results)} failed={n_fail}")
sys.exit(1 if n_fail else 0)
