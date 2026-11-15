"""Card baselines-h2gb C1: full-fan-out equivalence of a class-A model on H2GB's `hgt` sampler.

On a synthetic heterogeneous graph, with fan-out >= every node and sampler depth = the model's propagation steps (the
card's rule), each seed target's output on its sampled subgraph equals its output of the full-batch forward (float64,
atol 1e-10, fixed in advance); negative control: fan-out 1 differs.

* GCN: HGB's GCN (`vendor/HGB/NC/benchmark/methods/GNN/GNN.py:58-90`, the adapter's model; `num_layers + 1`
  GraphConvs) on HGB's homogeneous reduction of the graph / of the sampled subgraph (`model.g` set per batch).
* HAN: H2GB's HAN, PyG `HANConv` on the raw relations (`H2GB/network/hetero_gnn_model.py:127-129`), two layers.

The real data root is unreachable; nothing is downloaded.

C1 stopped on these two tests (VERIFIED-RUN, job 16741 step 60; `c1_tests.txt`); they are kept here, outside
`tests/`, as the reproducer until the author decides C0 S1 (depth) and S5 (HAN). Run from `H2S-NN/`:
`python ../docs/baselines_h2gb_evidence/c1_equivalence_tests.py -v`.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "H2S-NN"))

from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.test_baselines_minibatch import full_fanout, synthetic_hetero

ATOL = 1e-10


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _dgl_graph(edge_index: torch.Tensor, num_nodes: int):
    import dgl

    return dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)


def build_hgb_gcn(data, num_layers: int):
    """HGB's GCN in float64, eval mode, constructed inside the vendor block (vendor_paths docstring)."""
    import torch.nn.functional as F

    from src.baselines.minibatch import hgb_homogeneous_batch
    from src.baselines.runtime import pushd_sys_path
    from src.baselines.runtime._dgl_compat import resolve_dgl_device  # noqa: F401  - must precede dgl use

    node_types = tuple(data.node_types)
    features, edge_index, _ = hgb_homogeneous_batch(data, node_types)
    torch.manual_seed(0)
    with pushd_sys_path("hgb_nc_gnn"):
        from GNN import GCN  # type: ignore[import-not-found]

        model = GCN(_dgl_graph(edge_index, sum(f.size(0) for f in features)), [f.size(1) for f in features],
                    8, 3, num_layers, F.elu, 0.0)
    return model.double().eval(), node_types


def gcn_full(model, data, node_types):
    from src.baselines.minibatch import hgb_homogeneous_batch

    features, edge_index, offsets = hgb_homogeneous_batch(data, node_types)
    model.g = _dgl_graph(edge_index, sum(f.size(0) for f in features))
    with torch.no_grad():
        out = model([f.double() for f in features])
    start, end = offsets["a"]
    return out[start:end]


def gcn_seeds(model, node_types):
    from src.baselines.minibatch import hgb_homogeneous_batch

    def forward(batch):
        features, edge_index, offsets = hgb_homogeneous_batch(batch, node_types)
        model.g = _dgl_graph(edge_index, sum(f.size(0) for f in features))
        out = model([f.double() for f in features])
        start = offsets["a"][0]
        return out[start:start + int(batch["a"].batch_size)]
    return forward


class _H2GBHAN(torch.nn.Module):
    """Per-type input projection, two PyG HANConv layers on the raw relations (H2GB's HAN), linear head."""

    def __init__(self, data) -> None:
        super().__init__()
        from torch_geometric.nn import HANConv

        torch.manual_seed(0)
        self.inputs = torch.nn.ModuleDict({t: torch.nn.Linear(data[t].x.size(1), 8) for t in data.node_types})
        self.convs = torch.nn.ModuleList([HANConv(8, 8, data.metadata(), heads=1, dropout=0.0) for _ in range(2)])
        self.head = torch.nn.Linear(8, 3)

    def forward(self, x_dict, edge_index_dict):
        h = {t: self.inputs[t](x.double()) for t, x in x_dict.items()}
        for conv in self.convs:
            out = conv(h, edge_index_dict)
            h = {t: torch.relu(v) for t, v in out.items() if v is not None}
        return self.head(h["a"])


class FullFanoutEquivalenceTests(unittest.TestCase):
    def _check(self, data, full_out, forward, depth: int) -> None:
        from src.baselines.minibatch import eval_loader

        for split in ("val", "test"):
            loader = eval_loader(data, full_fanout(data, depth), "a", split, 4)
            with torch.no_grad():
                for batch in loader:
                    ids = batch["a"].n_id[: int(batch["a"].batch_size)]
                    got = forward(batch)
                    diff = (got - full_out[ids]).abs().max().item()
                    self.assertLessEqual(diff, ATOL, f"{split}: first differing batch (seeds {ids.tolist()}), "
                                                     f"max |diff| = {diff:.3e}")
        # negative control: fan-out 1 at the same depth differs somewhere
        loader = eval_loader(data, {t: [1] * depth for t in data.node_types}, "a", "val", 4)
        worst = 0.0
        with torch.no_grad():
            for batch in loader:
                ids = batch["a"].n_id[: int(batch["a"].batch_size)]
                worst = max(worst, (forward(batch) - full_out[ids]).abs().max().item())
        self.assertGreater(worst, 1e-6, "negative control: fan-out 1 gave the full-batch output")

    def test_gcn_full_fanout_equals_full_batch(self) -> None:
        data = synthetic_hetero()
        model, node_types = build_hgb_gcn(data, num_layers=1)  # 2 GraphConvs -> depth 2
        self._check(data, gcn_full(model, data, node_types), gcn_seeds(model, node_types), depth=2)

    def test_han_full_fanout_equals_full_batch(self) -> None:
        data = synthetic_hetero()
        model = _H2GBHAN(data).double().eval()
        with torch.no_grad():
            full_out = model(data.x_dict, data.edge_index_dict)

        def forward(batch):
            return model(batch.x_dict, batch.edge_index_dict)[: int(batch["a"].batch_size)]

        self._check(data, full_out, forward, depth=2)


if __name__ == "__main__":
    unittest.main()
