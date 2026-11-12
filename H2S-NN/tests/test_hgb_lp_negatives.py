"""HGB's LP negatives (spec v1.16 §6, K1-K2; cards Part M M7, Part N N1).

Toy graph (users 0-2, artists 3-6): train 0-1 (user-user), 0-3, 1-3, 1-4, 2-5 and test 0-4, 2-6
enter HGB's adjacency in both orientations; validation 1->6 as stored only:
  A: 0:{1,3,4} 1:{0,3,4,6} 2:{5,6} 3:{0,1} 4:{0,1} 5:{2} 6:{2}
  2-hop of head 0: via 1 -> {0,3,4,6}; artists minus direct links {3,4} -> candidates [6].
  2-hop of head 2: via 5, 6 -> {2}; no artist -> no candidate, so head 2 gets no 2-hop negative.
If the same validation fact were stored as 6->1, row 1 would lose 6 and head 0 would have no
candidate: the orientation matters (K2).
Random set: known neighbours of 0 = {1,3,4}, of 2 = {5,6}; one tail per positive, redrawn while
known, seed 1.
"""

from __future__ import annotations

import random
import unittest
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from src.data.hgb_lp_negatives import (NO_TWO_HOP_PATH_REASON, hgb_adjacency,
                                       hgb_random_test_negatives, hgb_test_negatives,
                                       hgb_valid_negatives, two_hop_candidates)

USERS, ARTISTS = (0, 3), (3, 7)
TRAIN = [(0, 1), (0, 3), (1, 3), (1, 4), (2, 5)]
VALID = [(1, 6)]
TEST = [(0, 4), (2, 6)]


def _edges(pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long).t().reshape(2, -1)


def _toy(valid=VALID):
    return _edges(TEST), _edges(TRAIN + TEST), _edges(valid)


def _hgb_pos_links(both, one, n):
    """HGB's `pos_links` (dense): train/test + transpose, validation as stored (counts)."""
    a = np.zeros((n, n), dtype=np.int64)
    for u, v in both:
        a[u, v] += 1
        a[v, u] += 1
    for u, v in one:
        a[u, v] += 1
    return a


def _hgb_get_test_neigh(test, both, one, n, src, dst):
    """Independent re-implementation of HGB's get_test_neigh loop (dense)."""
    a = _hgb_pos_links(both, one, n)
    two = (a @ a > 0).astype(np.int64) - a - np.eye(n, dtype=np.int64)
    neg_neigh = defaultdict(list)
    for h, t in zip(*np.nonzero(two > 0)):
        if src[0] <= h < src[1] and dst[0] <= t < dst[1]:
            neg_neigh[int(h)].append(int(t))
    pos_neigh = defaultdict(list)
    for h, t in sorted(set(test)):
        pos_neigh[h].append(t)
    random.seed(1)
    heads, tails = [], []
    for h in sorted(pos_neigh):
        k = len(pos_neigh[h])
        negs = random.choices(neg_neigh[h], k=k) if neg_neigh[h] else []
        heads += [h] * len(negs)
        tails += negs
    return [heads, tails]


def _hgb_get_test_neigh_w_random(test, both, one, n, dst):
    """Independent re-implementation of HGB's get_test_neigh_w_random loop."""
    a = _hgb_pos_links(both, one, n)
    known = defaultdict(set)
    for h, t in zip(*np.nonzero(a)):
        known[int(h)].add(int(t))
    random.seed(1)
    heads, tails = [], []
    for h, _ in sorted(set(test)):
        neg = random.randrange(dst[0], dst[1])
        while neg in known[h]:
            neg = random.randrange(dst[0], dst[1])
        heads.append(h)
        tails.append(neg)
    return [heads, tails]


class TwoHopCandidateTests(unittest.TestCase):
    def test_candidates_by_hand(self) -> None:
        _, both, one = _toy()
        adjacency = hgb_adjacency(both, one, 7)
        cands = two_hop_candidates(adjacency, np.array([0, 1, 2]), ARTISTS)
        self.assertEqual([c.tolist() for c in cands], [[6], [], []])

    def test_test_negatives_by_hand(self) -> None:
        test_pos, both, one = _toy()
        result = hgb_test_negatives(test_pos, both, one, 7, ARTISTS)
        self.assertEqual(result.edge_index.tolist(), [[0], [6]])
        self.assertEqual(result.heads_without_candidates.tolist(), [2])
        self.assertEqual(result.positives_of_heads_without, 1)

    def test_validation_orientation_changes_the_candidates_as_hgb(self) -> None:
        """The validation fact 1->6 stored as 6->1 removes head 0's only candidate, exactly as
        HGB's loop (mutation target: validation added in both orientations)."""
        test_pos, both, _ = _toy()
        flipped = hgb_test_negatives(test_pos, both, _edges([(6, 1)]), 7, ARTISTS)
        self.assertEqual(flipped.edge_index.tolist(), [[], []])
        self.assertEqual(flipped.heads_without_candidates.tolist(), [0, 2])
        for valid in ([(1, 6)], [(6, 1)]):
            ours = hgb_test_negatives(test_pos, both, _edges(valid), 7, ARTISTS)
            self.assertEqual(ours.edge_index.tolist(),
                             _hgb_get_test_neigh(TEST, TRAIN + TEST, valid, 7, USERS, ARTISTS))

    def test_head_without_candidates_gets_no_negative(self) -> None:
        """No random fallback (mutation target: a fallback gives head 2 a negative)."""
        test_pos, both, one = _toy()
        result = hgb_test_negatives(test_pos, both, one, 7, ARTISTS)
        self.assertNotIn(2, result.edge_index[0].tolist())

    def test_seed_one_and_hgb_loop_on_a_random_graph(self) -> None:
        """Random graph with a one-way validation set: equal to the independent re-implementation
        of HGB's loop (heads ascending, random.seed(1), random.choices, candidates ascending)."""
        generator = torch.Generator().manual_seed(0)
        n, src, dst = 60, (0, 20), (20, 60)
        pairs = sorted({(int(torch.randint(0, 20, (1,), generator=generator)),
                         int(torch.randint(20, 60, (1,), generator=generator)))
                        for _ in range(150)})
        friends = sorted({(int(torch.randint(0, 20, (1,), generator=generator)),
                           int(torch.randint(0, 20, (1,), generator=generator)))
                          for _ in range(30)} - {(i, i) for i in range(20)})
        test, valid, train = pairs[:30], pairs[30:45], pairs[45:] + friends
        first = hgb_test_negatives(_edges(test), _edges(train + test), _edges(valid), n, dst)
        second = hgb_test_negatives(_edges(test), _edges(train + test), _edges(valid), n, dst)
        self.assertTrue(torch.equal(first.edge_index, second.edge_index))
        expected = _hgb_get_test_neigh(test, train + test, valid, n, src, dst)
        self.assertEqual(first.edge_index.tolist(), expected)
        self.assertGreater(len(expected[0]), 0)

    def test_no_candidate_is_known_positive_or_self_loop(self) -> None:
        generator = torch.Generator().manual_seed(3)
        n = 80
        pairs = sorted({(int(torch.randint(0, 40, (1,), generator=generator)),
                         int(torch.randint(0, 80, (1,), generator=generator))) for _ in range(300)})
        known = {(u, v) for u, v in pairs} | {(v, u) for u, v in pairs}
        result = hgb_test_negatives(_edges(pairs[:50]), _edges(pairs), _edges([]), n, (0, 80))
        for h, t in result.edge_index.t().tolist():
            self.assertNotEqual(h, t)
            self.assertNotIn((h, t), known)
            self.assertTrue(0 <= t < 80)


class RandomSetTests(unittest.TestCase):
    def test_random_set_by_hand(self) -> None:
        test_pos, both, one = _toy()
        negatives = hgb_random_test_negatives(test_pos, both, one, 7, ARTISTS)
        self.assertEqual(negatives[0].tolist(), [0, 2])           # one per positive, sorted
        self.assertNotIn(int(negatives[1, 0]), {1, 3, 4})          # known neighbours of 0
        self.assertNotIn(int(negatives[1, 1]), {5, 6})             # known neighbours of 2
        self.assertEqual(negatives.tolist(),
                         _hgb_get_test_neigh_w_random(TEST, TRAIN + TEST, VALID, 7, ARTISTS))

    def test_seed_one_redraw_and_count_on_a_random_graph(self) -> None:
        """Independent re-implementation of HGB's loop; no negative is a known neighbour (train,
        validation as stored, test); count = test positives; deterministic (mutation target: the
        redraw removed)."""
        generator = torch.Generator().manual_seed(5)
        n, dst = 40, (10, 18)
        pairs = sorted({(int(torch.randint(0, 10, (1,), generator=generator)),
                         int(torch.randint(10, 18, (1,), generator=generator)))
                        for _ in range(60)})
        test, valid, train = pairs[:15], pairs[15:22], pairs[22:]
        ours = hgb_random_test_negatives(_edges(test), _edges(train + test), _edges(valid), n,
                                         dst)
        again = hgb_random_test_negatives(_edges(test), _edges(train + test), _edges(valid), n,
                                          dst)
        self.assertTrue(torch.equal(ours, again))
        self.assertEqual(ours.size(1), len(test))
        self.assertEqual(ours.tolist(),
                         _hgb_get_test_neigh_w_random(test, train + test, valid, n, dst))
        a = _hgb_pos_links(train + test, valid, n)
        for h, t in ours.t().tolist():
            self.assertEqual(a[h, t], 0)
            self.assertTrue(dst[0] <= t < dst[1])


class EmptySetMetricTests(unittest.TestCase):
    def test_an_empty_2hop_set_gives_none_with_the_reason(self) -> None:
        from src.utils.misc import lp_test_set_metrics

        data = SimpleNamespace(
            test_pos_edge_index=_edges([(0, 4), (2, 6)]), test_pos_edge_relation=torch.zeros(2),
            test_neg_edge_index=_edges([]), test_neg_edge_relation=torch.zeros(0),
            test_2hop_empty_reason=NO_TWO_HOP_PATH_REASON,
            test_neg_random_edge_index=_edges([(0, 5), (2, 3)]),
            test_neg_random_edge_relation=torch.zeros(2))
        scores = {(0, 4): 2.0, (2, 6): 1.0, (0, 5): -1.0, (2, 3): 0.5}

        def score(edges, relation):
            return torch.tensor([scores[(int(u), int(v))] for u, v in edges.t().tolist()])

        metrics = lp_test_set_metrics(score, data)
        for metric in ("auroc", "aupr", "mrr"):
            self.assertIsNone(metrics[f"2hop_{metric}"])
            self.assertIsNotNone(metrics[f"random_{metric}"])
        self.assertEqual(metrics["2hop_reason"], NO_TWO_HOP_PATH_REASON)
        self.assertEqual((metrics["2hop_negatives"], metrics["random_negatives"]), (0, 2))


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

    OFFSETS = {"user": 0, "artist": 12, "tag": 24}

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

    def _hgb_parts(self):
        """HGB's pos_links pieces from the loader's split (every relation): training and test
        facts (both orientations), validation positives as stored."""
        from src.data.link_prediction_dataset import _load_local_hgb_link_data

        split = _load_local_hgb_link_data(_tiny_lastfm_config(self.root))

        def pairs(by_relation):
            out = []
            for (src, _, dst), index in by_relation.items():
                for u, v in index.t().tolist():
                    out.append((u + self.OFFSETS[src], v + self.OFFSETS[dst]))
            return out

        train = pairs(split.train_edge_index_by_relation)
        test = pairs(split.test_edge_index_by_relation)
        valid = pairs(split.valid_edge_index_by_relation)
        return train, test, valid

    def test_both_test_sets_are_hgb_on_the_release(self) -> None:
        data = self._data()
        train, test, valid = self._hgb_parts()
        test_pos = sorted(map(tuple, data.test_pos_edge_index.t().tolist()))
        self.assertEqual(data.test_neg_edge_index.tolist(),
                         _hgb_get_test_neigh(test_pos, train + test, valid, 30, (0, 12),
                                             (12, 24)))
        self.assertEqual(data.test_neg_random_edge_index.tolist(),
                         _hgb_get_test_neigh_w_random(test_pos, train + test, valid, 30,
                                                      (12, 24)))
        self.assertEqual(data.test_neg_random_edge_index.size(1), len(test_pos))
        self.assertIsNone(data.test_2hop_empty_reason)
        heads_with = set(data.test_neg_edge_index[0].tolist())
        for head in data.test_heads_without_negatives.tolist():
            self.assertNotIn(head, heads_with)  # no fallback negative

    def test_a_head_without_candidates_gets_no_negative_through_the_loader(self) -> None:
        """User 0 loses its two friendship facts (0-1, 11-0): its neighbours are artists only, so
        it reaches no artist in two hops and its test positive gets no 2-hop negative (no
        fallback); the random set still gives it one."""
        from src.data.data_layout import lp_root

        link = lp_root(self.root) / "LastFM" / "link.dat"
        rows = [r for r in link.read_text(encoding="utf-8").splitlines(keepends=True)
                if r.split("\t")[:3] not in (["0", "1", "1"], ["11", "0", "1"])]
        link.write_text("".join(rows), encoding="utf-8")
        data = self._data()
        self.assertIn(0, data.test_pos_edge_index[0].tolist())
        self.assertIn(0, data.test_heads_without_negatives.tolist())
        self.assertNotIn(0, data.test_neg_edge_index[0].tolist())
        self.assertIn(0, data.test_neg_random_edge_index[0].tolist())
        self.assertEqual(data.test_positives_without_negatives,
                         int((data.test_pos_edge_index[0] == 0).sum()))

    def test_validation_negatives_are_hgb_get_valid_neg(self) -> None:
        data = self._data()
        config = _tiny_lastfm_config(self.root)
        expected = hgb_valid_negatives(data.val_pos_edge_index, (12, 24),
                                       seed=config.data.acm_split_seed + 101 + 0)
        self.assertTrue(torch.equal(data.val_neg_edge_index, expected))

    def test_written_files_are_read_and_must_match_the_positives(self) -> None:
        import json

        from scripts.lp.write_hgb_negatives import write_task
        from src.data.link_prediction_dataset import HGB_NEGATIVES_DIR, local_hgb_link_dir

        computed = self._data()
        write_task(self.root, "lastfm")
        folder = local_hgb_link_dir(self.root, "lastfm") / HGB_NEGATIVES_DIR
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        entry = meta["relations"]["0"]
        self.assertEqual(entry["test_positives"], computed.test_pos_edge_index.size(1))
        self.assertEqual(entry["test_negatives_random"], computed.test_pos_edge_index.size(1))
        self.assertEqual(entry["heads_without_candidates_count"],
                         computed.test_heads_without_negatives.numel())
        self.assertTrue((folder / entry["test_neg_2hop_file"]).exists())
        self.assertTrue((folder / entry["test_neg_random_file"]).exists())
        read = self._data()
        for name in ("test_neg_edge_index", "test_neg_random_edge_index",
                     "test_heads_without_negatives"):
            self.assertTrue(torch.equal(getattr(read, name), getattr(computed, name)), name)
        entry["test_pos_md5"] = "0" * 32
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
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
