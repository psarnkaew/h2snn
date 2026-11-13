"""E3b: DGL through the project's own route (src/baselines/runtime/_dgl_compat.py stubs GraphBolt), run from H2S-NN/.
usage: cd H2S-NN; python -W default ../docs/envB_evidence/e3b_dgl_via_shim.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch  # noqa: E402
from src.baselines.runtime import _dgl_compat  # noqa: E402,F401
import dgl  # noqa: E402
import dgl.function as fn  # noqa: E402
import numpy  # noqa: E402

print("torch", torch.__version__, "| numpy", numpy.__version__, "| dgl", dgl.__version__)
print("dgl.graphbolt is stub:", type(sys.modules.get("dgl.graphbolt")).__name__, getattr(sys.modules.get("dgl.graphbolt"), "__file__", None))
print("dgl_supports_cuda():", _dgl_compat.dgl_supports_cuda())
g = dgl.graph(([0, 1, 2], [1, 2, 0]))
g.ndata["h"] = torch.ones(3, 2)
g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h2"))
print("homogeneous message passing:", g.ndata["h2"].tolist())
hg = dgl.heterograph({("a", "w", "p"): ([0, 1], [0, 0]), ("p", "c", "p"): ([0], [1])})
hg.nodes["a"].data["x"] = torch.ones(2, 3)
hg.update_all(fn.copy_u("x", "m"), fn.sum("m", "y"), etype="w")
print("heterograph message passing:", hg.nodes["p"].data["y"].tolist())
from dgl.nn import GraphConv, GATConv  # noqa: E402
print("GraphConv:", GraphConv(2, 4)(g, g.ndata["h"]).shape, "GATConv:", GATConv(2, 4, 2)(g, g.ndata["h"]).shape)
