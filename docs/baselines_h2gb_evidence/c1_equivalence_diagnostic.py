"""C1 diagnostic (card baselines-h2gb): max |seed output - full-batch output| of the two equivalence-test models on the
same synthetic graph, by sampler depth, full fan-out (no tolerance is changed; this only locates the cause of the C1
test failures). GCN: HGB's GCN with 2 GraphConvs; HAN: two PyG HANConv layers (H2GB's HAN).

Expected if the harness is right: GCN differs at depth 2 (boundary nodes' degrees in the induced subgraph enter the
`norm="both"` coefficients) and is exact from depth 3; HAN differs at every depth below the graph's diameter
(semantic attention = mean over every destination node of the subgraph, `han_conv.py` group()), and is exact only when
the subgraph holds every node of every type that the full graph has.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.test_baselines_minibatch import full_fanout, synthetic_hetero  # noqa: E402
# part 2 E1: the reproducer moved to tests/test_baselines_minibatch_equivalence.py (GCN only); the HAN
# model of this diagnostic was defined in its part 1 version (git show 6ba6dec:docs/baselines_h2gb_evidence/c1_equivalence_tests.py)
from tests.test_baselines_minibatch_equivalence import build_hgb_gcn, gcn_full, gcn_seeds  # noqa: E402
from src.baselines.minibatch import eval_loader  # noqa: E402


def worst(data, full_out, forward, depth):
    loader = eval_loader(data, full_fanout(data, depth), "a", "val", 4)
    out, sizes = 0.0, []
    with torch.no_grad():
        for batch in loader:
            ids = batch["a"].n_id[: int(batch["a"].batch_size)]
            out = max(out, (forward(batch) - full_out[ids]).abs().max().item())
            sizes.append(sum(int(batch[t].num_nodes) for t in batch.node_types))
    return out, max(sizes)


data = synthetic_hetero()
total = sum(int(data[t].num_nodes) for t in data.node_types)
gcn, types = build_hgb_gcn(data, num_layers=1)
gcn_full_out = gcn_full(gcn, data, types)
han = _H2GBHAN(data).double().eval()
with torch.no_grad():
    han_full_out = han(data.x_dict, data.edge_index_dict)
print(f"graph: {total} nodes; target type a {int(data['a'].num_nodes)}")
for depth in (1, 2, 3, 4, 6):
    g, gs = worst(data, gcn_full_out, gcn_seeds(gcn, types), depth)
    h, hs = worst(data, han_full_out, lambda b: han(b.x_dict, b.edge_index_dict)[: int(b["a"].batch_size)], depth)
    print(f"depth={depth}: GCN max|diff|={g:.3e}  HAN max|diff|={h:.3e}  (largest subgraph {max(gs, hs)} of {total} nodes)")
