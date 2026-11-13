"""Part MB, MB1 (`docs/H2S-NN_partMB_card.md`): the scope builder and global node ids.

- the scope of P1 on the hand graph equals the hand lists (nodes, facts, cells);
- the scope is a lower set (every cell's facts and nodes, every fact's endpoints are in it);
- no fact joins two scope nodes unless it is a fact of the target or on one of its cells;
- the forced cell maps are well defined on scope cells (`validate_face_orientation`);
- a featureless node copied into two scopes gets the same h0 as in the whole graph;
- a scope built on a second graph uses that graph's facts only (N6).
Card mutations: local position instead of the global id -> the h0 test fails; one induced fact
added -> the lower-set / no-induced test fails.
"""

from __future__ import annotations

import unittest

import torch

from src.data.scopes import build_scope_batch, scope_members
from tests.partMB_fixtures import (
    EDGES,
    cells_of_paper_1,
    example_cells,
    example_graph,
    fact,
    restore_real_data,
    unreachable_real_data,
)


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


HAND_NODES = [0, 1, 2, 3, 6, 7, 8, 10]
HAND_FACTS = sorted([fact(0, 6, 0), fact(0, 7, 0), fact(0, 8, 0), fact(0, 10, 1), fact(0, 1, 2),
                     fact(2, 0, 2), fact(1, 6, 0), fact(3, 10, 1), fact(3, 7, 0)])


class ScopeTests(unittest.TestCase):
    def test_scope_of_p1_equals_the_hand_lists(self) -> None:
        graph = example_graph()
        members = scope_members(graph, 0, example_cells())
        self.assertEqual(members.nodes.tolist(), HAND_NODES)
        self.assertEqual(members.facts.tolist(), HAND_FACTS)
        batch = build_scope_batch(graph, torch.tensor([0]), [example_cells()])
        self.assertEqual(batch.global_node_id.tolist(), HAND_NODES)
        self.assertEqual(batch.global_fact_id.tolist(), HAND_FACTS)
        cells = [batch.global_node_id[g.face_node_index].tolist() for g in batch.data.face_groups]
        self.assertEqual(cells, [[[0, 6, 1]], [[0, 10, 3, 7]]])
        self.assertEqual(int(batch.global_node_id[batch.target_rows[0]]), 0)

    def test_lower_set_and_no_induced_facts(self) -> None:
        graph = example_graph()
        batch = build_scope_batch(graph, torch.tensor([0, 1, 3]),
                                  [example_cells(), cells_of_paper_1(), ()])
        data = batch.data
        # lower set: every cell's facts and nodes, every fact's endpoints, in the cell's scope
        for group, scope in zip(data.face_groups, batch.cell_scope):
            for c in range(group.num_faces):
                s = int(scope[c])
                self.assertTrue(bool((batch.node_scope[group.face_node_index[c]] == s).all()))
                self.assertTrue(bool((batch.fact_scope[group.face_edge_index[c]] == s).all()))
        self.assertTrue(bool((batch.node_scope[data.edge_index[0]] == batch.fact_scope).all()))
        self.assertTrue(bool((batch.node_scope[data.edge_index[1]] == batch.fact_scope).all()))
        # every batch fact is the copy of its graph fact (same endpoints and relation)
        ends = batch.global_node_id[data.edge_index]
        self.assertTrue(torch.equal(ends, graph.edge_index[:, batch.global_fact_id]))
        self.assertTrue(torch.equal(data.edge_type, graph.edge_type[batch.global_fact_id]))
        # no induced fact: each batch fact is a fact of its target or on one of its cells
        cells = {0: example_cells(), 1: cells_of_paper_1(), 3: ()}
        for e in range(data.edge_index.size(1)):
            s = int(batch.fact_scope[e])
            target = int(batch.targets[s])
            g = int(batch.global_fact_id[e])
            own = set(sum((grp.face_edge_index.reshape(-1).tolist() for grp in cells[target]), []))
            u, v = graph.edge_index[:, g].tolist()
            self.assertTrue(target in (u, v) or g in own, f"induced fact {g} in scope of {target}")
        self.assertNotIn(fact(1, 3, 2), batch.global_fact_id[batch.fact_scope == 0].tolist())

    def test_forced_cell_maps_are_defined_on_scope_cells(self) -> None:
        from src.data.face_geometry import validate_face_orientation

        graph = example_graph()
        batch = build_scope_batch(graph, torch.tensor([0, 1]), [example_cells(), cells_of_paper_1()])
        for group in batch.data.face_groups:
            validate_face_orientation(group.face_node_index, group.face_edge_index,
                                      batch.data.edge_index, "scope",
                                      face_edge_sign=group.face_edge_sign)

    def test_featureless_node_in_two_scopes_has_its_global_h0(self) -> None:
        from src.configs.config import TrainConfig
        from src.models.h2snn_model import H2SNN, ModelDims

        graph = example_graph()
        cfg = TrainConfig.from_dict({
            "device": "cpu", "data": {"dataset_name": "acm", "wisp_walks_per_anchor": 2},
            "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                      "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
            "ablation": {"coupling": "symmetric"}})
        torch.manual_seed(0)
        model = H2SNN(cfg, ModelDims.from_prepared_data(graph)).eval()
        batch = build_scope_batch(graph, torch.tensor([0, 1]), [example_cells(), cells_of_paper_1()])
        with torch.no_grad():
            whole = model.encode_nodes(graph.x, graph.node_type)
            scoped = model.encode_nodes(batch.data.x, batch.data.node_type,
                                        batch.data.global_node_id)
        copies = (batch.global_node_id == 6).nonzero(as_tuple=True)[0]
        self.assertEqual(copies.numel(), 2)  # author A6 is in both scopes
        self.assertEqual(batch.node_scope[copies].tolist(), [0, 1])
        self.assertTrue(torch.equal(scoped, whole[batch.global_node_id]))

    def test_scope_on_a_second_graph_uses_its_facts(self) -> None:
        second = [(u, v, r) for u, v, r in EDGES if (u, v) != (0, 8)] + [(0, 9, 0)]
        graph = example_graph(edges=second)
        batch = build_scope_batch(graph, torch.tensor([0]), [()])
        members = sorted(batch.global_fact_id.tolist())
        expected = sorted(i for i, (u, v, _) in enumerate(second) if 0 in (u, v))
        self.assertEqual(members, expected)
        self.assertIn(9, batch.global_node_id.tolist())
        self.assertNotIn(8, batch.global_node_id.tolist())


if __name__ == "__main__":
    unittest.main()
