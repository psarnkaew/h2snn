"""Part M3b B6: the two large link-prediction tasks and their epoch loop, on synthetic graphs (never real data).

(i)   a node in several links of a step is computed once and its score equals the score from its own scope
      (float64, solver tolerance 1e-12, |difference| <= 1e-10 relative, fixed in advance); negative control: a
      link's score does not change when the other links of the batch change;
(ii)  `assert_no_held_out_facts` holds for every scope of a step, in both orientations, with a pair carrying
      several facts (both orientations, a repeated fact and a second relation);
(iii) epoch and patience counting: a scripted validation sequence stops at the expected epoch; steps per
      epoch = ceil(P / 8192) and a step never crosses a pass;
(iv)  the split is by unordered pair (a pair and its reverse are always in the same part) and equals the
      derived tasks' split rule;
(v)   the decoder `concat_mlp` stays refused for the large tasks.
Also: the chunked 2-hop negatives equal the unchunked ones, and the list-free `choices` equals
`random.choices`. Card mutations: scopes that include the other endpoint's cells -> (i) fails; held-out removal
restricted to one orientation -> (ii) fails.
"""

from __future__ import annotations

import random
import unittest
from unittest import mock

import numpy as np
import torch

import src.trainers.minibatch_lp as mlp
from src.data.large_lp import held_out_fact_mask, split_pairs, unordered_pairs
from src.data.structures import PreparedH2SData
from src.trainers.minibatch_lp import LinkEpochLoop, LinkScopes, remove_held_out_facts
from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.test_partMB_link import N, _graph, _model, _pairs


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


TOL = 1e-10  # relative, float64, fixed in advance (card B6 (i))


class StepScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = _graph(_pairs(), double=True)
        self.scopes = LinkScopes(_model(self.graph, tolerance=1e-12), self.graph, 3, 3)

    def _alone(self, u: int, v: int, cells) -> float:
        _, rep_u = self.scopes.representations(torch.tensor([u]), cells)
        _, rep_v = self.scopes.representations(torch.tensor([v]), cells)
        return float(self.scopes.model.link_readout(torch.cat([rep_u, rep_v]), torch.tensor([[0], [1]]),
                                                    None)[0])

    def test_i_shared_endpoint_computed_once_and_equal_to_its_own_scope(self) -> None:
        links = torch.tensor([[0, 0, 3, 5, 0], [1, 2, 0, 6, 7]])  # node 0 in four links
        cells = self.scopes.cells(links.reshape(-1), seed=8)
        solved = []
        real = mlp.build_scope_batch

        def spy(g, targets, own):
            solved.append(targets.clone())
            return real(g, targets, own)

        with mock.patch.object(mlp, "build_scope_batch", spy), torch.no_grad():
            batched = self.scopes.score(links, cells)
        self.assertEqual(len(solved), 1)
        self.assertEqual(solved[0].tolist(), sorted(set(links.reshape(-1).tolist())))  # each node once
        with torch.no_grad():
            for i in range(links.size(1)):
                u, v = links[:, i].tolist()
                alone = self._alone(u, v, cells)
                self.assertLessEqual(abs(float(batched[i]) - alone) / max(abs(alone), 1e-300), TOL,
                                     f"link {u}-{v}")

    def test_i_negative_control_other_links_do_not_change_a_score(self) -> None:
        cells = self.scopes.cells(torch.arange(N), seed=8)
        with torch.no_grad():
            a = self.scopes.score(torch.tensor([[0, 3, 5], [1, 4, 6]]), cells)[0]
            b = self.scopes.score(torch.tensor([[9, 0, 11, 12], [10, 1, 13, 2]]), cells)[1]
        self.assertLessEqual(abs(float(a - b)) / max(abs(float(a)), 1e-300), TOL)


class HeldOutTests(unittest.TestCase):
    def test_ii_no_held_out_fact_in_any_scope_of_a_step_with_duplicate_facts(self) -> None:
        from src.data.fact_layer import build_fact_layer
        from src.data.link_prediction_dataset import assert_no_held_out_facts
        from src.data.scopes import build_scope_batch

        pairs = _pairs()
        held = [(0, 1), (2, 3), (4, 5)]
        # every held-out pair carries several facts: both orientations, a repeated fact, a second relation
        facts = [(u, v, 0) for u, v in pairs if (u, v) not in held and (v, u) not in held]
        for u, v in held:
            facts += [(u, v, 0), (v, u, 0), (u, v, 0), (u, v, 1)]
        edge_index = torch.tensor([[u for u, _, _ in facts], [v for _, v, _ in facts]])
        edge_type = torch.tensor([r for _, _, r in facts])
        held_out = torch.tensor(held).t()
        message_index, message_type = remove_held_out_facts(edge_index, edge_type, held_out)
        graph = _graph([tuple(p) for p in message_index.t().tolist()])
        scopes = LinkScopes(_model(graph, 1e-4, double=False), graph, 3, 3)
        positives = torch.tensor([[6, 7, 8], [9, 10, 11]])
        negatives = torch.tensor([[6, 7, 8], [0, 2, 4]])  # endpoints of held-out pairs are step targets
        endpoints = torch.unique(torch.cat([positives, negatives], dim=1))
        cells = scopes.cells(endpoints, seed=3)
        layer = build_fact_layer(scopes.graph.edge_index, scopes.graph.edge_type,
                                 scopes.graph.relation_names, N, "untied")
        both = torch.cat([held_out, held_out.flip(0)], dim=1)
        for target in endpoints.tolist():
            batch = build_scope_batch(scopes.graph, torch.tensor([target]), [cells.of(target)])
            used = batch.global_fact_id
            assert_no_held_out_facts(layer, scopes.graph.edge_index[:, used], scopes.graph.edge_type[used],
                                     (), both, torch.zeros(both.size(1), dtype=torch.long))
            pairs_used = {tuple(p) for p in scopes.graph.edge_index[:, used].t().tolist()}
            self.assertFalse(pairs_used & {tuple(p) for p in both.t().tolist()}, target)
        self.assertEqual(int(held_out_fact_mask(message_index, held_out).sum()), 0)
        self.assertEqual(int(held_out_fact_mask(edge_index, held_out).sum()), 4 * len(held))


class EpochLoopTests(unittest.TestCase):
    def test_iii_patience_counted_in_epochs_stops_at_the_expected_epoch(self) -> None:
        loop = LinkEpochLoop(scopes=mock.Mock(), optimizer=mock.Mock(), base_seed=0, epochs=40, patience=3,
                             dst_range=(0, N))
        sequence = iter([0.5, 0.6, 0.6, 0.55, 0.59, 0.7, 0.1])  # best at epoch 1; 3 epochs without gain
        with mock.patch.object(LinkEpochLoop, "train_epoch", return_value=0.0), \
                mock.patch.object(LinkEpochLoop, "validate", side_effect=lambda *a: next(sequence)):
            best = loop.fit(torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 1), torch.zeros(2, 1))
        self.assertEqual(best, 1)
        self.assertEqual([h["epoch"] for h in loop.history], [0, 1, 2, 3, 4])

    def test_iii_one_epoch_is_one_pass_and_a_step_never_crosses_it(self) -> None:
        from src.trainers.minibatch_trainer import TrainingOrder

        loop = LinkEpochLoop(scopes=mock.Mock(), optimizer=mock.Mock(), base_seed=0, epochs=1, patience=1,
                             dst_range=(0, N))
        self.assertEqual(loop.epoch_steps(8192), 1)
        self.assertEqual(loop.epoch_steps(8193), 2)
        positives = torch.zeros(2, 8192 + 100, dtype=torch.long)
        taken = []
        loop.scopes.train_step.side_effect = lambda p, n, *a: taken.append(p.size(1)) or 0.0
        loop.train_epoch(TrainingOrder(torch.arange(positives.size(1)), 0), positives, 0)
        self.assertEqual(taken, [8192, 100])

    def test_iii_training_run_on_a_synthetic_task(self) -> None:
        graph = _graph(_pairs())
        scopes = LinkScopes(_model(graph, 1e-4, double=False), graph, 2, 2)
        optimizer = torch.optim.AdamW(scopes.model.parameters(), lr=1e-3)
        loop = LinkEpochLoop(scopes, optimizer, base_seed=0, epochs=2, patience=40, dst_range=(0, N))
        train_pos = graph.edge_index[:, :20]
        best = loop.fit(train_pos, graph.edge_index[:, 20:24], torch.tensor([[0, 1, 2, 3], [9, 8, 7, 6]]))
        self.assertIn(best, (0, 1))
        self.assertEqual(len(loop.history), 2)
        self.assertTrue(all(np.isfinite(h["loss"]) for h in loop.history))


class SplitTests(unittest.TestCase):
    def test_iv_split_is_by_unordered_pair(self) -> None:
        rng = random.Random(0)
        facts = {(rng.randrange(50), rng.randrange(50)) for _ in range(600)}
        facts |= {(v, u) for u, v in list(facts)[:200]}  # reverse copies
        edge_index = torch.tensor(sorted(facts)).t()
        pairs, stats = unordered_pairs(torch.cat([edge_index, edge_index[:, :5]], dim=1))  # + repeated facts
        self.assertEqual(stats["pairs_with_a_repeated_fact"], len({(min(u, v), max(u, v))
                                                                   for u, v in sorted(facts)[:5] if u != v}))
        train, valid, test = split_pairs(pairs)
        part = {}
        for name, block in (("train", train), ("valid", valid), ("test", test)):
            for u, v in block.tolist():
                self.assertLess(u, v)
                part[(u, v)] = name
        for u, v in facts:
            if u != v:
                self.assertEqual(part[(min(u, v), max(u, v))], part[(min(v, u), max(v, u))])
        self.assertEqual(len(part), pairs.shape[0])
        self.assertEqual((test.shape[0], valid.shape[0]), (round(0.10 * len(part)), round(0.09 * len(part))))

    def test_iv_split_equals_the_derived_tasks_rule(self) -> None:
        from scripts.lp.build_derived_lp import split_facts

        pairs = np.array(sorted({(min(a, b), max(a, b)) for a, b in
                                 zip(range(0, 300, 3), range(7, 400, 4)) if a != b}))
        train, valid, test = split_pairs(pairs, 42)
        ref = split_facts([tuple(p) for p in pairs.tolist()], 42, 0.09, 0.10)
        self.assertEqual([list(map(tuple, b.tolist())) for b in (train, valid, test)], [list(r) for r in ref])


class DecoderTests(unittest.TestCase):
    def test_v_concat_mlp_is_refused_for_the_large_tasks(self) -> None:
        from src.configs.config import TrainConfig

        for task in ("ogbn-mag_pp", "h-pokec_uu"):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "concat_mlp"):
                TrainConfig.from_dict({"data": {"dataset_name": task, "task_type": "link_prediction",
                                                "decoder_type": "concat_mlp"}})


class NegativeTests(unittest.TestCase):
    def test_chunked_two_hop_equals_unchunked(self) -> None:
        from src.data.hgb_lp_negatives import hgb_test_negatives

        rng = torch.Generator().manual_seed(5)
        edges = torch.randint(0, 40, (2, 300), generator=rng)
        test = torch.randint(0, 40, (2, 60), generator=rng)
        valid = torch.randint(0, 40, (2, 10), generator=rng)
        whole = hgb_test_negatives(test, torch.cat([edges, test], 1), valid, 40, (0, 40))
        for chunk in (1, 3, 7):
            part = hgb_test_negatives(test, torch.cat([edges, test], 1), valid, 40, (0, 40), head_chunk=chunk)
            self.assertTrue(torch.equal(whole.edge_index, part.edge_index))
            self.assertTrue(torch.equal(whole.heads_without_candidates, part.heads_without_candidates))

    def test_list_free_choices_equal_random_choices(self) -> None:
        from src.data.hgb_lp_negatives import _choices

        population = np.array([3, 9, 27, 81, 5, 11], dtype=np.int64)
        for seed in range(5):
            a, b = random.Random(seed), random.Random(seed)
            for k in (1, 4, 13):
                self.assertEqual(_choices(a, population, k), b.choices(population.tolist(), k=k))


if __name__ == "__main__":
    unittest.main()
