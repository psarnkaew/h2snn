"""D12 (task card Part G): relational context (G1, T-G7) and the filter-bank encoder (G2, T-G1 ...
T-G6, T-G8). Float64 fixtures with path cells and signed hops."""
from __future__ import annotations

import unittest

import torch

from src.data.fact_layer import apply_fact_layer, build_fact_layer
from src.models.relational_context import RelationalContext
from src.models.sheaf_maps import _build_mlp, mlp_on_blocks
from tests.test_v2_model import _float64_pap, _v2_model


def _reference_context(h0: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                       num_relations: int, shared: tuple[int, ...] = ()) -> torch.Tensor:
    """Loop reference of `c`: slot (r, role) of node v = mean over the other endpoints."""
    n, d = h0.shape
    sums = torch.zeros(n, 2 * num_relations, d, dtype=h0.dtype)
    counts = torch.zeros(n, 2 * num_relations, dtype=h0.dtype)
    for (u, v), r in zip(edge_index.t().tolist(), edge_type.tolist()):
        role_v = 0 if r in shared else 1
        sums[u, 2 * r] += h0[v]
        counts[u, 2 * r] += 1
        sums[v, 2 * r + role_v] += h0[u]
        counts[v, 2 * r + role_v] += 1
    return (sums / counts.clamp_min(1).unsqueeze(-1)).reshape(n, -1)


def _random_facts(seed: int, num_nodes: int = 12, num_edges: int = 30, num_relations: int = 3):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(num_nodes, (2, num_edges), generator=g),
            torch.randint(num_relations, (num_edges,), generator=g),
            torch.randn(num_nodes, 5, generator=g, dtype=torch.float64))


# ----------------------------------------------------------------------------- G1: context
class RelationalContextTests(unittest.TestCase):
    def test_context_equals_the_loop_reference(self) -> None:
        for seed in range(5):
            edge_index, edge_type, h0 = _random_facts(seed)
            c = RelationalContext(3)(h0, edge_index, edge_type)
            self.assertEqual(tuple(c.shape), (12, 2 * 3 * 5))
            self.assertLessEqual(float((c - _reference_context(h0, edge_index, edge_type, 3)).abs().max()),
                                 1e-12)
            # a node with no neighbour in a slot gets zeros there
        isolated = torch.zeros(1, 2 * 3 * 5, dtype=torch.float64)
        edge_index = torch.tensor([[0], [1]])
        c = RelationalContext(3)(torch.randn(3, 5, dtype=torch.float64), edge_index, torch.tensor([0]))
        self.assertTrue(torch.equal(c[2:3], isolated))

    def test_shared_role_relation_is_orientation_independent(self) -> None:
        edge_index, edge_type, h0 = _random_facts(7)
        flipped = edge_index.clone()
        on_zero = edge_type == 0
        flipped[:, on_zero] = edge_index.flip(0)[:, on_zero]
        shared = RelationalContext(3, (0,))
        self.assertLessEqual(float((shared(h0, edge_index, edge_type)
                                    - shared(h0, flipped, edge_type)).abs().max()), 1e-12)
        reference = _reference_context(h0, edge_index, edge_type, 3, shared=(0,))
        self.assertLessEqual(float((shared(h0, edge_index, edge_type) - reference).abs().max()), 1e-12)
        directed = RelationalContext(3)  # control: roles distinct, flipping moves c
        self.assertGreater(float((directed(h0, edge_index, edge_type)
                                  - directed(h0, flipped, edge_type)).abs().max()), 1e-6)

    def test_facts_only_no_double_counting_through_the_reverse_relation(self) -> None:
        """Stored edges in both directions (r and rho(r)); the tied fact layer keeps r. c over the
        facts equals the loop reference over the stored edges of r alone."""
        g = torch.Generator().manual_seed(3)
        writes = torch.stack([torch.randint(6, 12, (20,), generator=g), torch.randint(0, 6, (20,), generator=g)])
        writes = torch.unique(writes, dim=1)
        stored_index = torch.cat([writes, writes.flip(0)], dim=1)
        stored_type = torch.cat([torch.zeros(writes.size(1), dtype=torch.long),
                                 torch.ones(writes.size(1), dtype=torch.long)])
        layer = build_fact_layer(stored_index, stored_type, ("writes", "written_by"), 12, "tied")
        facts, fact_type = apply_fact_layer(layer, stored_index, stored_type)
        self.assertEqual(len(layer.kept), 1)
        h0 = torch.randn(12, 4, generator=g, dtype=torch.float64)
        c = RelationalContext(len(layer.kept))(h0, facts, fact_type)
        reference = _reference_context(h0, writes, torch.zeros(writes.size(1), dtype=torch.long), 1)
        self.assertEqual(tuple(c.shape), (12, 2 * 4))
        self.assertLessEqual(float((c - reference).abs().max()), 1e-12)


class FactoredFirstLayerTests(unittest.TestCase):
    def test_gathered_first_layer_equals_the_concatenation(self) -> None:
        g = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        mlp = _build_mlp(2 * 7 + 3 + 2, 11, 5).double()
        h = torch.randn(9, 7, generator=g, dtype=torch.float64)
        rel = torch.randn(4, 3, generator=g, dtype=torch.float64)
        src, tgt = torch.randint(9, (2, 25), generator=g)
        types = torch.randint(4, (25,), generator=g)
        side = torch.randn(25, 2, generator=g, dtype=torch.float64)
        blocks = ((h, src), (h, tgt), (rel, types), (side, None))
        factored = mlp_on_blocks(mlp, blocks, factored=True)
        concat = mlp_on_blocks(mlp, blocks, factored=False)
        self.assertLessEqual(float((factored - concat).abs().max()), 1e-12)
        self.assertTrue(torch.equal(concat, mlp(torch.cat([h[src], h[tgt], rel[types], side], -1))))


class ContextModelTests(unittest.TestCase):
    """T-G7: one neighbour's h0 reaches a node's maps and gates only under geometry_context."""

    def setUp(self) -> None:
        self.data = _float64_pap()
        ei = self.data.edge_index
        # a paper v with two distinct author neighbours u (perturbed) and w (edge kept)
        for v in range(int((self.data.node_type == 0).sum())):
            out = ei[1][ei[0] == v].unique()
            if out.numel() >= 2:
                self.v, self.u, self.w = v, int(out[0]), int(out[1])
                break
        self.edge = int(((ei[0] == self.v) & (ei[1] == self.w)).nonzero()[0])

    def _maps_and_gates(self, model, h0):
        geometry = model.geometry_inputs(h0, self.data.edge_index, self.data.edge_type)
        maps = model.node_to_edge(geometry, self.data.edge_index, self.data.edge_type,
                                  model.relation_embedding.weight).maps
        listen, broadcast = model.coupling_gates(geometry)
        return (maps.src.materialize()[self.edge], maps.tgt.materialize()[self.edge],
                listen[self.v], broadcast[self.v])

    def test_TG7_neighbour_reaches_maps_and_gates_only_with_context(self) -> None:
        for context in (True, False):
            with self.subTest(geometry_context=context):
                model = _v2_model(self.data, geometry_context=context).double().eval()
                with torch.no_grad():
                    h0 = model.encode_nodes(self.data.x, self.data.node_type)
                    moved = h0.clone()
                    moved[self.u] += 1.0
                    before = self._maps_and_gates(model, h0)
                    after = self._maps_and_gates(model, moved)
                for name, a, b in zip(("F_src", "F_tgt", "g", "b"), before, after):
                    if context:
                        self.assertGreater(float((a - b).abs().max()), 1e-12, name)
                    else:
                        self.assertTrue(torch.equal(a, b), name)

    def test_context_off_builds_the_d11_parameter_shapes(self) -> None:
        off = _v2_model(self.data, geometry_context=False)
        on = _v2_model(self.data)
        self.assertIsNone(off.relational_context)
        self.assertEqual(off.geometry_dim, off.d)
        num_relations = len(self.data.relation_names)
        self.assertEqual(on.geometry_dim, on.d * (1 + 2 * num_relations))
        self.assertEqual(off.listen_gate.in_features, off.d)
        self.assertEqual(on.listen_gate.in_features, on.geometry_dim)
        self.assertEqual(sorted(off.state_dict()), sorted(on.state_dict()))  # parameter-free context


if __name__ == "__main__":
    unittest.main()
