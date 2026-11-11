"""Part M, M7: HGB's get_test_neigh / get_valid_neg ported exactly (spec v1.15 §6, J5).

Toy graph (users 0-2, artists 3-6), every fact symmetric in A:
  train 0-1 (user-user), 0-3, 1-3, 1-4, 2-5; valid 1-6; test 0-4, 2-6.
  A: 0:{1,3,4} 1:{0,3,4,6} 2:{5,6} 3:{0,1} 4:{0,1} 5:{2} 6:{1,2}
  2-hop of head 0: {0,1,3,4,6}; artists minus direct links {3,4} -> candidates [6].
  2-hop of head 2: {1,2}; no artist -> no candidate, so head 2 gets no negative.
"""

from __future__ import annotations

import random
import unittest
from collections import defaultdict

import numpy as np
import torch

from src.data.hgb_lp_negatives import (hgb_test_negatives, hgb_valid_negatives,
                                       symmetric_adjacency, two_hop_candidates)

USERS, ARTISTS = (0, 3), (3, 7)
TRAIN = [(0, 1), (0, 3), (1, 3), (1, 4), (2, 5)]
VALID = [(1, 6)]
TEST = [(0, 4), (2, 6)]


def _edges(pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long).t().reshape(2, -1)


def _toy() -> tuple[torch.Tensor, torch.Tensor]:
    return _edges(TEST), _edges(TRAIN + VALID + TEST)


class TwoHopCandidateTests(unittest.TestCase):
    def test_candidates_by_hand(self) -> None:
        _, adjacency_edges = _toy()
        adjacency = symmetric_adjacency(adjacency_edges, 7)
        cands = two_hop_candidates(adjacency, np.array([0, 1, 2]), ARTISTS)
        self.assertEqual([c.tolist() for c in cands], [[6], [], []])

    def test_test_negatives_by_hand(self) -> None:
        test_pos, adjacency_edges = _toy()
        result = hgb_test_negatives(test_pos, adjacency_edges, 7, ARTISTS)
        self.assertEqual(result.edge_index.tolist(), [[0], [6]])
        self.assertEqual(result.heads_without_candidates.tolist(), [2])
        self.assertEqual(result.positives_of_heads_without, 1)

    def test_head_without_candidates_gets_no_negative(self) -> None:
        """No random fallback (mutation target: a fallback gives head 2 a negative)."""
        test_pos, adjacency_edges = _toy()
        result = hgb_test_negatives(test_pos, adjacency_edges, 7, ARTISTS)
        self.assertNotIn(2, result.edge_index[0].tolist())

    def test_seed_one_and_hgb_choices_order(self) -> None:
        """Heads ascending, random.Random(1).choices per head with k = its positives, candidates
        ascending: the sequence of HGB's `random.seed(1)` + `random.choices`."""
        generator = torch.Generator().manual_seed(0)
        num_nodes, src, dst = 60, (0, 20), (20, 60)
        pairs = {(int(torch.randint(0, 20, (1,), generator=generator)),
                  int(torch.randint(20, 60, (1,), generator=generator))) for _ in range(150)}
        pairs = sorted(pairs)
        test = pairs[:30]
        # head-head facts (as LastFM's user-user): without them a bipartite graph has no 2-hop
        # tail of the tail type
        friends = sorted({(int(torch.randint(0, 20, (1,), generator=generator)),
                           int(torch.randint(0, 20, (1,), generator=generator)))
                          for _ in range(30)} - {(i, i) for i in range(20)})
        pairs = pairs + friends
        test_pos, all_edges = _edges(test), _edges(pairs)
        first = hgb_test_negatives(test_pos, all_edges, num_nodes, dst)
        second = hgb_test_negatives(test_pos, all_edges, num_nodes, dst)
        self.assertTrue(torch.equal(first.edge_index, second.edge_index))
        # independent re-implementation with a dense matrix and HGB's loop
        a = np.zeros((num_nodes, num_nodes), dtype=np.int64)
        for u, v in pairs:
            a[u, v] = a[v, u] = 1
        two = (a @ a > 0).astype(np.int64) - a - np.eye(num_nodes, dtype=np.int64)
        neg_neigh = defaultdict(list)
        for h, t in zip(*np.nonzero(two > 0)):
            if src[0] <= h < src[1] and dst[0] <= t < dst[1]:
                neg_neigh[int(h)].append(int(t))
        pos_neigh = defaultdict(list)
        for h, t in test:
            pos_neigh[h].append(t)
        rng = random.Random(1)
        heads, tails = [], []
        for h in sorted(pos_neigh):
            k = len(pos_neigh[h])
            negs = rng.choices(neg_neigh[h], k=k) if neg_neigh[h] else []
            heads += [h] * len(negs)
            tails += negs
        self.assertEqual(first.edge_index.tolist(), [heads, tails])
        self.assertGreater(len(heads), 0)

    def test_no_candidate_is_known_positive_or_self_loop(self) -> None:
        generator = torch.Generator().manual_seed(3)
        num_nodes = 80
        pairs = sorted({(int(torch.randint(0, 40, (1,), generator=generator)),
                         int(torch.randint(0, 80, (1,), generator=generator))) for _ in range(300)})
        known = {(u, v) for u, v in pairs} | {(v, u) for u, v in pairs}
        result = hgb_test_negatives(_edges(pairs[:50]), _edges(pairs), num_nodes, (0, 80))
        for h, t in result.edge_index.t().tolist():
            self.assertNotEqual(h, t)
            self.assertNotIn((h, t), known)
            self.assertTrue(0 <= t < 80)


class ValidNegativeTests(unittest.TestCase):
    def test_one_uniform_tail_per_positive_in_order(self) -> None:
        valid = _edges([(0, 3), (1, 4), (1, 5), (2, 6)])
        negatives = hgb_valid_negatives(valid, ARTISTS, seed=11)
        rng = random.Random(11)
        self.assertEqual(negatives[0].tolist(), [0, 1, 1, 2])
        self.assertEqual(negatives[1].tolist(), [rng.randrange(3, 7) for _ in range(4)])


def _hgb_evaluate_mrr(edge_list, confidence, labels) -> float:
    """HGB data_loader.evaluate (MRR part), as published."""
    t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, h_id in enumerate(edge_list[0]):
        t_dict[h_id].append(edge_list[1][i])
        labels_dict[h_id].append(labels[i])
        conf_dict[h_id].append(confidence[i])
    mrr_list = []
    for h_id in t_dict.keys():
        rank = np.argsort(-np.array(conf_dict[h_id]))
        sorted_label_array = np.array(labels_dict[h_id])[rank]
        pos_index = np.where(sorted_label_array == 1)[0]
        if len(pos_index) == 0:
            continue
        mrr_list.append(1 / (1 + np.min(pos_index)))
    return float(np.mean(mrr_list))


class MRRTests(unittest.TestCase):
    def test_mrr_hand_example_matches_hgb(self) -> None:
        """Head 0: positive 0.2 below negatives 0.9 and 0.5 -> rank 3, RR 1/3. Head 1: positive
        first -> 1. Head 2: positives only (no negatives) -> 1. MRR = (1/3 + 1 + 1) / 3."""
        from src.utils.misc import compute_link_prediction_metrics

        pos = _edges([(0, 10), (1, 11), (2, 12)])
        neg = _edges([(0, 13), (0, 14), (1, 15)])
        pos_scores = torch.tensor([0.2, 3.0, -1.0])
        neg_scores = torch.tensor([0.9, 0.5, 1.0])
        ours = compute_link_prediction_metrics(pos_scores, neg_scores, pos, neg)["mrr"]
        edge_list = torch.cat([pos, neg], dim=1).tolist()
        hgb = _hgb_evaluate_mrr(edge_list, torch.cat([pos_scores, neg_scores]).tolist(),
                                [1, 1, 1, 0, 0, 0])
        self.assertAlmostEqual(ours, (1 / 3 + 1 + 1) / 3, places=12)
        self.assertAlmostEqual(ours, hgb, places=12)


def _tiny_lastfm_config(root, **data):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({"device": "cpu", "seed": 0, "data": {
        "dataset_name": "lastfm", "root": str(root), "wisp_walks_per_anchor": 2,
        "wisp_max_length": 2, **data}, "model": {"d": 8, "d_r": 4, "classifier_hidden_dim": 8,
                                                  "face_mlp_hidden_dim": 8,
                                                  "edge_mlp_hidden_dim": 8, "K": 1}})


def _raw_pairs(path) -> list[tuple[int, int]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        u, v = line.split("\t")[:2]
        rows.append((int(u), int(v)))
    return rows


class LoaderNegativeTests(unittest.TestCase):
    """The loader on a tiny LastFM-format release (raw ids = global ids: users 0-11, artists
    12-23, tags 24-29)."""

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        from tests.test_task_heads import _write_tiny_lastfm

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_tiny_lastfm(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _data(self):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(_tiny_lastfm_config(self.root))

    def test_test_negatives_are_hgb_get_test_neigh_on_the_release(self) -> None:
        """Independent adjacency: every fact of link.dat (train + validation) and link.dat.test,
        read from the files."""
        from src.data.data_layout import lp_root

        data = self._data()
        folder = lp_root(self.root) / "LastFM"
        facts = _raw_pairs(folder / "link.dat") + _raw_pairs(folder / "link.dat.test")
        expected = hgb_test_negatives(data.test_pos_edge_index, _edges(facts), 30, (12, 24))
        self.assertTrue(torch.equal(data.test_neg_edge_index, expected.edge_index))
        self.assertTrue(torch.equal(data.test_heads_without_negatives,
                                    expected.heads_without_candidates))
        heads_with = set(data.test_neg_edge_index[0].tolist())
        for head in data.test_heads_without_negatives.tolist():
            self.assertNotIn(head, heads_with)  # no fallback negative
        known = set(facts) | {(v, u) for u, v in facts}
        for h, t in data.test_neg_edge_index.t().tolist():
            self.assertTrue(12 <= t < 24)
            self.assertNotIn((h, t), known)
        counts = {h: data.test_pos_edge_index[0].tolist().count(h) for h in heads_with}
        for head, k in counts.items():
            self.assertEqual(data.test_neg_edge_index[0].tolist().count(head), k)

    def test_a_head_without_candidates_gets_no_negative_through_the_loader(self) -> None:
        """User 0 loses its two friendship facts (0-1, 11-0): its neighbours are artists only, so
        it reaches no artist in two hops and its test positive gets no negative (no fallback)."""
        from src.data.data_layout import lp_root

        link = lp_root(self.root) / "LastFM" / "link.dat"
        rows = [r for r in link.read_text(encoding="utf-8").splitlines(keepends=True)
                if r.split("\t")[:3] not in (["0", "1", "1"], ["11", "0", "1"])]
        link.write_text("".join(rows), encoding="utf-8")
        data = self._data()
        self.assertIn(0, data.test_pos_edge_index[0].tolist())
        self.assertIn(0, data.test_heads_without_negatives.tolist())
        self.assertNotIn(0, data.test_neg_edge_index[0].tolist())
        self.assertEqual(data.test_positives_without_negatives,
                         int((data.test_pos_edge_index[0] == 0).sum()))
        self.assertLess(data.test_neg_edge_index.size(1), data.test_pos_edge_index.size(1))

    def test_validation_negatives_are_hgb_get_valid_neg(self) -> None:
        data = self._data()
        config = _tiny_lastfm_config(self.root)
        expected = hgb_valid_negatives(data.val_pos_edge_index, (12, 24),
                                       seed=config.data.acm_split_seed + 101 + 0)
        self.assertTrue(torch.equal(data.val_neg_edge_index, expected))

    def test_written_file_is_read_and_must_match_the_positives(self) -> None:
        import json

        from scripts.lp.write_hgb_negatives import write_task
        from src.data.link_prediction_dataset import HGB_NEGATIVES_DIR, local_hgb_link_dir

        computed = self._data()
        write_task(self.root, "lastfm")
        meta_path = local_hgb_link_dir(self.root, "lastfm") / HGB_NEGATIVES_DIR / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entry = meta["relations"]["0"]
        self.assertEqual(entry["test_positives"], computed.test_pos_edge_index.size(1))
        self.assertEqual(entry["heads_without_candidates_count"],
                         computed.test_heads_without_negatives.numel())
        read = self._data()
        self.assertTrue(torch.equal(read.test_neg_edge_index, computed.test_neg_edge_index))
        self.assertTrue(torch.equal(read.test_heads_without_negatives,
                                    computed.test_heads_without_negatives))
        entry["test_pos_md5"] = "0" * 32
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "other test positives"):
            self._data()


class LinkBatchTests(unittest.TestCase):
    def test_one_optimizer_step_and_one_solve_per_batch(self) -> None:
        """link_batch_size B: ceil(P / B) optimizer steps per epoch, one encode (solve) each."""
        import math
        import tempfile
        from pathlib import Path

        from src.builders import build_optimizer
        from src.data.link_prediction_dataset import build_link_prediction_dataset
        from src.models.h2snn_model import build_link_prediction_model
        from src.trainers.link_prediction_trainer import LinkPredictionTrainer
        from tests.test_task_heads import _write_tiny_lastfm

        self.assertEqual(_tiny_lastfm_config(Path(".")).data.link_batch_size, 8192)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tiny_lastfm(root)
            config = _tiny_lastfm_config(root, link_batch_size=10)
            data = build_link_prediction_dataset(config)
            torch.manual_seed(0)
            model = build_link_prediction_model(config, data)
            optimizer = build_optimizer(model, config)
            trainer = LinkPredictionTrainer(model, optimizer, None, config, logger=None,
                                            checkpointer=None)
            encodes, steps = [0], [0]
            original_encode, original_step = model.encode, optimizer.step

            def counting_encode(*args, **kwargs):
                encodes[0] += 1
                return original_encode(*args, **kwargs)

            def counting_step(*args, **kwargs):
                steps[0] += 1
                return original_step(*args, **kwargs)

            model.encode, optimizer.step = counting_encode, counting_step
            metrics = trainer._train_epoch(data, epoch=0)
            expected = math.ceil(data.train_pos_edge_index.size(1) / 10)
            self.assertGreater(expected, 1)
            self.assertEqual((steps[0], encodes[0]), (expected, expected))
            self.assertEqual(metrics["optimizer_steps"], float(expected))
            self.assertTrue(math.isfinite(metrics["loss"]))


if __name__ == "__main__":
    unittest.main()
