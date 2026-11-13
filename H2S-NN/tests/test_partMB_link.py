"""Part MB, MB5 (`docs/H2S-NN_partMB_card.md`): mini-batch link prediction for the large tasks.

On a synthetic same-type graph (users; some facts stored in both orientations):
- link scores from a batch of endpoint scopes equal those from each endpoint's scope solved alone
  (float64, tol 1e-10, <= 1e-8 relative);
- a repeated endpoint is solved once (spy on the scope builder) and gets one representation;
- no held-out fact (either orientation) is in any scope;
- evaluation scores every validation/test link.
Card mutation: a held-out fact kept in the message graph -> the leak test fails.
"""

from __future__ import annotations

import random
import unittest
from unittest import mock

import torch

import src.trainers.minibatch_lp as mlp
from src.data.structures import PreparedH2SData
from src.trainers.minibatch_lp import LinkScopes, remove_held_out_facts
from tests.partMB_fixtures import restore_real_data, unreachable_real_data


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


N = 24


def _pairs(seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < 60:
        u, v = rng.sample(range(N), 2)
        pairs.add((u, v))
    pairs |= {(v, u) for u, v in list(pairs)[:10]}  # some facts stored in both orientations
    return sorted(pairs)


def _graph(pairs, double: bool = False) -> PreparedH2SData:
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(N, 4, generator=generator)
    edge_index = torch.tensor(pairs, dtype=torch.long).t()
    mask = torch.zeros(N, dtype=torch.bool)
    return PreparedH2SData(
        x=x.double() if double else x, y=torch.zeros(N, dtype=torch.long),
        node_type=torch.zeros(N, dtype=torch.long), node_type_names=("user",),
        node_slices={"user": (0, N)}, feature_dims={"user": 4}, edge_index=edge_index,
        edge_type=torch.zeros(edge_index.size(1), dtype=torch.long),
        relation_names=("user__rel_0__user",), relation_tuples=(("user", "rel_0", "user"),),
        train_mask=mask, val_mask=mask, test_mask=mask, target_node_type="user",
        target_indices=torch.arange(N), target_y=torch.zeros(N, dtype=torch.long),
        target_train_mask=mask, target_val_mask=mask, target_test_mask=mask, num_classes=2,
        is_multilabel=False, face_groups=(), metrics=torch.zeros(0))


def _model(graph, tolerance: float = 1e-10, double: bool = True):
    from src.configs.config import TrainConfig
    from src.models.h2snn_model import H2SNN, ModelDims

    cfg = TrainConfig.from_dict({
        "device": "cpu",
        "data": {"dataset_name": "lastfm", "wisp_walks_per_anchor": 3, "wisp_max_length": 3},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full",
                  "solver_tolerance": tolerance, "solver_max_iterations": 2000},
        "ablation": {"coupling": "symmetric"}})
    dims = ModelDims.from_prepared_data(graph)._replace(face_lengths=(2, 3), path_actions=True)
    torch.manual_seed(0)
    model = H2SNN(cfg, dims).eval()
    return model.double() if double else model


class LinkScopeTests(unittest.TestCase):
    def test_scores_equal_each_endpoint_alone(self) -> None:
        graph = _graph(_pairs(), double=True)
        scopes = LinkScopes(_model(graph), graph, 3, 3)
        links = torch.tensor([[0, 3, 5, 7], [1, 4, 6, 0]])
        cells = scopes.cells(links.reshape(-1), seed=8)
        with torch.no_grad():
            batched = scopes.score(links, cells)
            for i in range(links.size(1)):
                u, v = links[:, i].tolist()
                _, rep_u = scopes.representations(torch.tensor([u]), cells)
                _, rep_v = scopes.representations(torch.tensor([v]), cells)
                rep = torch.cat([rep_u, rep_v])
                alone = scopes.model.link_readout(rep, torch.tensor([[0], [1]]), None)
                rel = float((batched[i] - alone[0]).abs() / alone[0].abs().clamp_min(1e-300))
                self.assertLessEqual(rel, 1e-8, f"link {u}-{v}")

    def test_repeated_endpoint_is_solved_once(self) -> None:
        graph = _graph(_pairs(), double=True)
        scopes = LinkScopes(_model(graph), graph, 3, 3)
        links = torch.tensor([[0, 0, 3], [1, 2, 0]])  # node 0 in three links
        cells = scopes.cells(links.reshape(-1), seed=8)
        seen = []
        real = mlp.build_scope_batch

        def spy(g, targets, own):
            seen.append(targets.clone())
            return real(g, targets, own)

        with mock.patch.object(mlp, "build_scope_batch", spy), torch.no_grad():
            unique, representation = scopes.representations(links.reshape(-1), cells)
            scopes.score(links, cells)
        self.assertEqual(unique.tolist(), [0, 1, 2, 3])
        self.assertEqual(representation.size(0), 4)
        for targets in seen:
            self.assertEqual(sorted(targets.tolist()), [0, 1, 2, 3])  # 0 once per solve

    def test_no_held_out_fact_in_any_scope(self) -> None:
        pairs = _pairs()
        both = [(u, v) for u, v in pairs if (v, u) in pairs and u < v]
        held_out = torch.tensor(both[:4] + [p for p in pairs if (p[1], p[0]) not in pairs][:4]).t()
        whole = _graph(pairs)
        edge_index, edge_type = remove_held_out_facts(whole.edge_index, whole.edge_type, held_out)
        message = _graph([tuple(p) for p in edge_index.t().tolist()])
        scopes = LinkScopes(_model(message, 1e-4, double=False), message, 3, 3)
        endpoints = torch.arange(N)
        cells = scopes.cells(endpoints, seed=2)
        from src.data.scopes import build_scope_batch

        batch = build_scope_batch(scopes.graph, endpoints, [cells.of(t) for t in range(N)])
        facts = {tuple(p) for p in scopes.graph.edge_index[:, batch.global_fact_id].t().tolist()}
        held = {tuple(p) for p in held_out.t().tolist()}
        held |= {(v, u) for u, v in held}
        self.assertEqual(facts & held, set())
        self.assertGreater(len(facts), 0)

    def test_evaluation_scores_every_link(self) -> None:
        graph = _graph(_pairs())
        scopes = LinkScopes(_model(graph, 1e-4, double=False), graph, 3, 3)
        links = torch.tensor([[i, (i + 5) % N] for i in range(N)]).t()
        scores = scopes.evaluate(links, seed=3)
        self.assertEqual(scores.numel(), links.size(1))
        self.assertTrue(bool(torch.isfinite(scores).all()))


if __name__ == "__main__":
    unittest.main()
