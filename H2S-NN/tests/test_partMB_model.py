"""Part MB, MB3 (`docs/H2S-NN_partMB_card.md`): the unchanged model on a batch of scopes.

On a synthetic ACM-shaped graph (`tests.partO_fixtures`), with mini-batch draws merged per anchor:
(a) batch = scopes alone: float64, solver tolerance 1e-10, every target's x* from a batch of 5
    scopes of different sizes equals the scope solved alone (max relative difference <= 1e-8);
(b) production setting (float32, tol 1e-4): every scope's own relative residual <= 1e-4, also
    when the other scopes' encodings are 1e6 times the hardest scope's;
(c) evaluation logits do not depend on the batching (batches of 3 and 7, shuffled), float64;
(d) whole-graph harness: the whole graph as one scope reproduces the full-batch logits;
(e) context and the per-row degree scalars of a scope's nodes equal those of the scope alone;
(f) no warm start in mini-batch evaluation; in full batch only for the same evaluation draw.
Card mutations: one global inner product (`scope_rows` None) -> (b) fails; warm start re-enabled
in mini-batch -> (f) fails.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest import mock

import torch

import src.models.anchored_solve as solver_module
from src.data.scopes import build_scope_batch
from src.data.structures import ScopeIndex
from src.data.wisp_sampler import build_wisp_anchor_cells
from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.partO_fixtures import acm_like_hetero, patched_acm_loader


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _config(tolerance: float, iterations: int = 100):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({
        "device": "cpu", "seed": 3,
        "data": {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 4},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full",
                  "solver_tolerance": tolerance, "solver_max_iterations": iterations},
        "ablation": {"coupling": "symmetric"}})


def _setup(tolerance: float, double: bool, iterations: int = 100):
    from dataclasses import replace

    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model

    cfg = _config(tolerance, iterations)
    with patched_acm_loader(acm_like_hetero(num_papers=30, num_authors=16, seed=12)):
        graph = build_dataset(cfg)
    torch.manual_seed(0)
    model = build_model(cfg, graph).eval()
    if double:
        model = model.double()
        graph = replace(graph, x=graph.x.double())
    targets = graph.target_indices[:10]
    cells = build_wisp_anchor_cells(graph.edge_index, graph.edge_type, graph.node_type,
                                    graph.fact_layer.self_inverse, targets, 4, 3, seed=21,
                                    chunk_size=4)
    return model, graph, targets, cells


def _batch(graph, cells, targets):
    batch = build_scope_batch(graph, targets, [cells.of(int(t)) for t in targets])
    if graph.x.dtype == torch.float64:
        from dataclasses import replace

        batch.data = replace(batch.data, x=batch.data.x.double())
    return batch


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


@contextmanager
def spy_solve():
    calls = []
    real = solver_module.solve_anchored

    def spy(system, h0, y0, tolerance, max_iterations, scope_rows=None, num_scopes=1,
            strict=True):
        out = real(system, h0, y0, tolerance, max_iterations, scope_rows, num_scopes, strict)
        calls.append({"system": system, "h0": h0, "y0": y0, "y": out[0]})
        return out

    with mock.patch.object(solver_module, "solve_anchored", spy):
        yield calls


class BatchTests(unittest.TestCase):
    def test_a_batch_equals_scopes_alone(self) -> None:
        model, graph, targets, cells = _setup(1e-10, True, iterations=2000)
        chosen = targets[:5]
        batch = _batch(graph, cells, chosen)
        sizes = torch.bincount(batch.node_scope).tolist()
        self.assertGreater(len(set(sizes)), 1, f"scopes of different sizes: {sizes}")
        with torch.no_grad():
            batched = model(batch.data).x_star[batch.target_rows]
            for i, t in enumerate(chosen.tolist()):
                alone = _batch(graph, cells, torch.tensor([t]))
                single = model(alone.data).x_star[alone.target_rows[0]]
                self.assertLessEqual(_rel(batched[i], single), 1e-8, f"scope {i} size {sizes[i]}")

    def test_b_every_scope_meets_its_own_tolerance(self) -> None:
        from src.models.h2snn_model import solved_system_scope_rows
        from src.models.sheaf_energy import IncidenceComplex

        model, graph, targets, cells = _setup(1e-4, False)
        batch = _batch(graph, cells, targets[:6])
        data = batch.data
        groups = tuple(g for g in data.face_groups if g.num_faces > 0)
        complex_ = IncidenceComplex.from_face_groups(data.edge_index, data.num_nodes, groups)
        rows = solved_system_scope_rows(data.scope_index, complex_, data.face_groups,
                                        complex_.num_shared_facts > 0, True)
        real = solver_module.solve_anchored
        found = {}

        def per_scope(system, h0, y0, tolerance, max_iterations, scope_rows=None,
                      num_scopes=1, strict=True):  # the test's own per-scope rows
            out = real(system, h0, y0, tolerance, max_iterations, rows, batch.num_scopes, strict)
            found["iterations"] = out[1].tolist()
            found["y"] = out[0]
            return out

        with mock.patch.object(solver_module, "solve_anchored", per_scope), torch.no_grad():
            model(data)
        iterations = found["iterations"]
        hard = max(range(len(iterations)), key=iterations.__getitem__)
        self.assertGreater(max(iterations), min(iterations), f"iterations {iterations}")
        # scopes of very different scale and progress: every other scope's right-hand side is
        # 1e6 times the hardest scope's and starts at its (scaled) per-scope solution, the
        # hardest scope at the D-B7 guess (operator and g unchanged). Per-scope stopping still
        # iterates the hardest scope to its own tolerance; one global inner product would be
        # satisfied by the large, already solved scopes at once.
        scale = torch.full((data.num_nodes, 1), 1e6)
        scale[batch.node_scope == hard] = 1.0
        easy_rows = rows != hard
        captured = {}

        def scaled(system, h0, y0, tolerance, max_iterations, scope_rows=None, num_scopes=1,
                   strict=True):  # the model's own scope_rows, passed through
            h0 = h0 * scale
            start = system.operator.initial_guess(h0).clone()
            start[easy_rows] = found["y"][easy_rows] * 1e6
            out = real(system, h0, start, tolerance, max_iterations, scope_rows, num_scopes,
                       strict)
            captured.update(system=system, h0=h0, y=out[0])
            return out

        with mock.patch.object(solver_module, "solve_anchored", scaled), torch.no_grad():
            model(data)
        system = captured["system"]
        b = system.right_hand_side(captured["h0"])
        r = b - system.apply(captured["y"])
        num = torch.zeros(batch.num_scopes).index_add(0, rows, (r * r).sum(-1)).sqrt()
        den = torch.zeros(batch.num_scopes).index_add(0, rows, (b * b).sum(-1)).sqrt()
        for scope, value in enumerate((num / den).tolist()):
            self.assertLessEqual(value, 1e-4, f"scope {scope}: relative residual {value:.2e}")

    def test_c_logits_do_not_depend_on_the_batching(self) -> None:
        model, graph, targets, cells = _setup(1e-10, True, iterations=2000)

        def logits(order, size):
            out = {}
            with torch.no_grad():
                for start in range(0, len(order), size):
                    part = torch.tensor(order[start:start + size])
                    batch = _batch(graph, cells, part)
                    for t, row in zip(part.tolist(), model(batch.data).logits):
                        out[t] = row
            return out

        order = targets.tolist()
        shuffled = [order[i] for i in torch.randperm(len(order),
                                                     generator=torch.Generator().manual_seed(5))]
        a, b, c = logits(order, 3), logits(order, 7), logits(shuffled, 3)
        for t in order:
            self.assertLessEqual(_rel(a[t], b[t]), 1e-8)
            self.assertLessEqual(_rel(a[t], c[t]), 1e-8)

    def test_d_whole_graph_harness(self) -> None:
        from dataclasses import replace

        model, graph, _, _ = _setup(1e-10, True, iterations=2000)
        groups = graph.face_groups
        whole = replace(graph, global_node_id=torch.arange(graph.num_nodes),
                        scope_index=ScopeIndex(
                            torch.zeros(graph.num_nodes, dtype=torch.long),
                            torch.zeros(graph.num_edges, dtype=torch.long),
                            tuple(torch.zeros(g.num_faces, dtype=torch.long) for g in groups), 1))
        with torch.no_grad():
            full = model(graph).logits
            harness = model(whole).logits
        self.assertLessEqual(_rel(harness, full), 1e-8)

    def test_e_context_and_degrees_are_the_scopes(self) -> None:
        model, graph, targets, cells = _setup(1e-10, True, iterations=2000)
        batch = _batch(graph, cells, targets[:4])
        alone = _batch(graph, cells, targets[1:2])  # scope 1 of the batch
        with torch.no_grad(), spy_solve() as calls:
            model(batch.data)
            model(alone.data)
            h0_b = model.encode_nodes(batch.data.x, batch.data.node_type, batch.global_node_id)
            h0_a = model.encode_nodes(alone.data.x, alone.data.node_type, alone.global_node_id)
            geo_b = model.geometry_inputs(h0_b, batch.data.edge_index, batch.data.edge_type)
            geo_a = model.geometry_inputs(h0_a, alone.data.edge_index, alone.data.edge_type)
        rows = (batch.node_scope == 1).nonzero(as_tuple=True)[0]
        self.assertTrue(torch.equal(batch.global_node_id[rows], alone.global_node_id))
        self.assertLessEqual(_rel(geo_b[rows], geo_a), 1e-12)  # relational context c
        jac_b = calls[0]["system"].jacobi_inverse()[:batch.data.num_nodes]
        jac_a = calls[1]["system"].jacobi_inverse()[:alone.data.num_nodes]
        self.assertLessEqual(_rel(jac_b[rows], jac_a), 1e-12)  # 1 / (1 + tau d_v)

    def test_f_no_warm_start_in_mini_batch(self) -> None:
        from dataclasses import replace

        model, graph, targets, cells = _setup(1e-4, False)
        batch = _batch(graph, cells, targets[:4])
        with torch.no_grad(), spy_solve() as calls:
            model(batch.data)
            model(batch.data)  # same batch, same rows: still the D-B7 guess
            model(graph)
            model(graph)  # full batch, same evaluation draw: warm start
            other = tuple(replace(g, path_weight=g.path_weight * 0.5) for g in graph.face_groups)
            model(replace(graph, face_groups=other))  # same shapes, another draw: no warm start
        y0 = [c["y0"] for c in calls]
        self.assertIsNone(y0[0])
        self.assertIsNone(y0[1])
        self.assertIsNone(y0[2])
        self.assertIsNotNone(y0[3])
        self.assertIsNone(y0[4])


if __name__ == "__main__":
    unittest.main()
