"""Part MB-fix, MF3 (`docs/H2S-NN_partMBfix_card.md`; decision "Part MB review (tests)"): T1
functoriality of the forced fact->cell maps on a batch of scopes.

The T1 computation of `tests/test_sheaf_energy.py:302-358` (`FunctorialityOnRealDrawTests`, which
needs IMDB data) on a batch of 4 scopes of the synthetic hand graph (`tests/partMB_fixtures.py`):
mini-batch draw (chunked, merged per anchor), factorized maps on the batch graph, `build_cell_group`
per group; `||R F_near - T_near||` and `||R F_far - T_far||` below `T1_TOL` (the same constant,
fp32 as there), with at least one reversed hop and one featureless node copied into two scopes;
`validate_face_orientation` holds on every batch group.
Card mutation: one batch cell's `face_edge_sign` flipped -> the test fails.
"""

from __future__ import annotations

import unittest

import torch

from src.data.face_geometry import validate_face_orientation
from src.data.scopes import build_scope_batch
from src.data.wisp_sampler import build_wisp_anchor_cells
from src.models.sheaf_energy import build_cell_group
from tests.partMB_fixtures import example_graph, restore_real_data, unreachable_real_data
from tests.test_sheaf_energy import T1_TOL


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class ScopeFunctorialityTests(unittest.TestCase):
    def test_T1_on_a_batch_of_scopes(self) -> None:
        from src.models.sheaf_maps import FactorizedNodeToEdgeSheaf

        graph = example_graph()
        targets = torch.tensor([0, 1, 3, 4])
        cells = build_wisp_anchor_cells(graph.edge_index, graph.edge_type, graph.node_type, (),
                                        targets, 6, 3, seed=5, chunk_size=2)
        batch = build_scope_batch(graph, targets, [cells.of(int(t)) for t in targets.tolist()])
        data = batch.data
        groups = tuple(g for g in data.face_groups if g.num_faces > 0)
        self.assertGreaterEqual(batch.num_scopes, 3)
        # the batch has a reversed hop and a featureless node (an author) in two scopes
        self.assertTrue(any(bool((g.face_edge_sign < 0).any()) for g in groups))
        authors = (data.node_type == 1).nonzero(as_tuple=True)[0]
        scopes_of = {}
        for row in authors.tolist():
            scopes_of.setdefault(int(batch.global_node_id[row]), set()).add(
                int(batch.node_scope[row]))
        self.assertTrue(any(len(s) >= 2 for s in scopes_of.values()))
        for group in groups:
            validate_face_orientation(group.face_node_index, group.face_edge_index,
                                      data.edge_index, "scope", face_edge_sign=group.face_edge_sign)

        torch.manual_seed(0)
        d = 32
        num_relations = len(data.relation_names)
        module = FactorizedNodeToEdgeSheaf(
            d=d, d_r=16, num_relations=num_relations, hidden_dim=64, householder_k=2,
            scale_eps=0.1, orthogonal_map="fasthpp", map_scale="per_edge",
            shared_role_relations=())
        h = torch.randn(data.num_nodes, d)
        relation_embeddings = torch.randn(num_relations, 16)
        num_edges = int(data.edge_index.size(1))
        with torch.no_grad():
            maps = module(h, data.edge_index, data.edge_type, relation_embeddings).maps
            scale = maps.src.scale
            worst, where, incidences = 0.0, None, 0
            for g_index, group in enumerate(groups):
                cell = build_cell_group(maps, group, data.edge_index, torch.ones(group.num_faces))
                P, L = group.face_edge_index.shape
                basis = torch.eye(d)
                T = torch.stack([cell.transport.apply(basis[k].expand(data.num_nodes, d))
                                 for k in range(d)], dim=-1)                     # [P, L+1, d, d]
                R = torch.stack([cell.fact_maps.apply(basis[k].expand(num_edges, d))
                                 for k in range(d)], dim=-1)                     # [P, L, d, d]
                facts = group.face_edge_index.reshape(-1)
                F_src = (maps.src.select(facts).materialize() / scale[facts].view(-1, 1, 1)
                         ).view(P, L, d, d)
                F_tgt = (maps.tgt.select(facts).materialize() / scale[facts].view(-1, 1, 1)
                         ).view(P, L, d, d)
                reversed_hop = (group.face_edge_sign < 0).view(P, L, 1, 1)
                F_near = torch.where(reversed_hop, F_tgt, F_src)
                F_far = torch.where(reversed_hop, F_src, F_tgt)
                residual = torch.maximum((R @ F_near - T[:, :-1]).norm(dim=(-2, -1)),
                                         (R @ F_far - T[:, 1:]).norm(dim=(-2, -1)))  # [P, L]
                value = float(residual.max())
                if value > worst:
                    flat = int(residual.argmax())
                    cell_index, hop = divmod(flat, L)
                    where = (int(batch.cell_scope[g_index][cell_index]), cell_index, hop)
                    worst = value
                incidences += P * L
        self.assertGreater(incidences, 0)
        self.assertLess(worst, T1_TOL, f"functoriality residual {worst} at (scope, cell, hop) "
                                       f"{where} on {incidences} incidences")


if __name__ == "__main__":
    unittest.main()
