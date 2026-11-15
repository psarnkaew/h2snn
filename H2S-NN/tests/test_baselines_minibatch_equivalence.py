"""Full-fan-out equivalence of a class-A model on H2GB's `hgt` sampler (card baselines-h2gb C1; part 2 E1, author
decision S1).

On a synthetic heterogeneous graph, with fan-out >= every node and sampler depth = the model's propagation steps + 1,
each seed target's output on its sampled subgraph equals its output of the full-batch forward (float64, atol 1e-10,
fixed in advance); negative control: fan-out 1 differs. Depth + 1 because HGB's GCN normalises by the degrees of both
endpoints (`norm="both"`), and a node at the last hop has its full degree only if its neighbours are sampled too (part 1
C1 diagnostic: exact from depth = steps + 1, not before).

* GCN: HGB's GCN (`vendor/HGB/NC/benchmark/methods/GNN/GNN.py:58-90`, the adapter's model; `num_layers + 1`
  GraphConvs) on HGB's homogeneous reduction of the graph / of the sampled subgraph (`model.g` set per batch).
HAN is not tested (decision S5: H2GB's HAN averages its semantic attention over the sampled subgraph, stated).
The real data root is unreachable; nothing is downloaded.
"""

from __future__ import annotations

import unittest

import torch

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
        model, node_types = build_hgb_gcn(data, num_layers=1)  # 2 GraphConvs -> depth 2 + 1
        self._check(data, gcn_full(model, data, node_types), gcn_seeds(model, node_types), depth=3)


if __name__ == "__main__":
    unittest.main()
