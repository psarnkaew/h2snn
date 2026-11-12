"""Part N-fix, F1: HGB's own LP code as the oracle of our ports (spec v1.16 K1-K3).

Tiny datasets are written in HGB's file format (`node.dat`, `link.dat`, `link.dat.test`,
`info.dat`) and loaded with HGB's `data_loader` class (`vendor/HGB/LP/benchmark/scripts/
data_loader.py`, imported from its file path under a private module name so it cannot collide
with any other vendored `data_loader`). Our functions get the same split (HGB's `links` after
its split, `links_test`, `valid_pos`) and must return identical (head, tail) lists in identical
order. HGB draws from Python's global `random`; ours from `random.Random(seed)`: the global
generator is seeded with the same value right before each HGB call (the state is restored
afterwards).

The tests skip only when `vendor/HGB` is absent (it is cloned, never committed: HGB declares no
licence). The Part N tests (`test_hgb_lp_negatives.py`) stay: they test our code on our hand
graphs; these are added.
"""

from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.hgb_lp_negatives import (hgb_random_test_negatives, hgb_test_negatives,
                                       hgb_train_negatives, hgb_valid_negatives)
from src.data.link_prediction_dataset import _split_edge_index_hgb_by_head

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HGB_LOADER = PROJECT_ROOT / "vendor" / "HGB" / "LP" / "benchmark" / "scripts" / "data_loader.py"
SKIP_REASON = ("vendor/HGB is absent: clone https://github.com/THUDM/HGB into H2S-NN/vendor/HGB "
               "(commit in vendor/HGB/CLONED_COMMIT.txt) to run HGB's code as the oracle")
SEEDS = (0, 1, 2, 3, 7)


def _hgb_module():
    spec = importlib.util.spec_from_file_location("_hgb_oracle_lp_data_loader", HGB_LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ------------------------------------------------------------------------------- tiny datasets
# Each: node types (count per type, ids contiguous by type, as HGB's loader assumes), training
# facts (link.dat) and test facts (link.dat.test) as (head, tail, relation).


def _dataset_one_way_validation() -> dict:
    """Two types (user 0-5, item 6-15), one relation user->item: every validation fact is stored
    one way only (user->item), so its orientation decides which 2-hop paths exist."""
    rng = np.random.default_rng(11)
    counts = (6, 10)
    links = set()
    for user in range(6):
        for item in rng.choice(np.arange(6, 16), size=6, replace=False):
            links.add((user, int(item), 0))
    test = set()
    for user in range(6):
        for item in range(6, 16):
            if (user, item, 0) not in links and len(test) < 2 * (user + 1):
                test.add((user, item, 0))
    return {"counts": counts, "links": sorted(links), "test": sorted(test)}


def _dataset_head_without_candidates() -> dict:
    """user 0-4, item 5-12; rel 0 user->item (target), rel 1 item->item among items 5-8 only.
    User 4 links only items 11, 12 (no item-item fact), so its 2-hop row has no item: no
    candidate; the other users reach items through rel 1."""
    counts = (5, 8)
    links = {(0, 5, 0), (0, 6, 0), (0, 7, 0), (0, 9, 0), (1, 5, 0), (1, 8, 0), (1, 9, 0),
             (1, 10, 0), (2, 6, 0), (2, 7, 0), (2, 8, 0), (2, 10, 0), (3, 5, 0), (3, 7, 0),
             (3, 9, 0), (3, 10, 0), (4, 11, 0), (4, 12, 0),
             (5, 6, 1), (6, 7, 1), (7, 8, 1), (8, 5, 1), (5, 7, 1), (6, 8, 1)}
    test = {(0, 8, 0), (1, 6, 0), (2, 5, 0), (3, 6, 0), (4, 10, 0)}
    return {"counts": counts, "links": sorted(links), "test": sorted(test)}


def _dataset_two_types_two_relations() -> dict:
    """user 0-7, item 8-19; rel 0 user->item (target), rel 1 user->user (directed, stored one
    way): a validation fact of rel 1 enters `pos_links` as stored only."""
    rng = np.random.default_rng(5)
    counts = (8, 12)
    links = set()
    for user in range(8):
        for item in rng.choice(np.arange(8, 20), size=5, replace=False):
            links.add((user, int(item), 0))
        for friend in rng.choice([u for u in range(8) if u != user], size=3, replace=False):
            links.add((user, int(friend), 1))
    test = set()
    for user in range(8):
        for item in rng.choice(np.arange(8, 20), size=4, replace=False):
            if (user, int(item), 0) not in links:
                test.add((user, int(item), 0))
    return {"counts": counts, "links": sorted(links), "test": sorted(test)}


DATASETS = {"one_way_validation": _dataset_one_way_validation,
            "head_without_candidates": _dataset_head_without_candidates,
            "two_types_two_relations": _dataset_two_types_two_relations}


def _write(folder: Path, spec: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    node_id = 0
    with (folder / "node.dat").open("w", encoding="utf-8") as handle:
        for node_type, count in enumerate(spec["counts"]):
            for _ in range(count):
                handle.write(f"{node_id}\tn{node_id}\t{node_type}\n")
                node_id += 1
    for name, rows in (("link.dat", spec["links"]), ("link.dat.test", spec["test"])):
        with (folder / name).open("w", encoding="utf-8") as handle:
            for head, tail, relation in rows:
                handle.write(f"{head}\t{tail}\t{relation}\t1.0\n")
    (folder / "info.dat").write_text("tiny HGB-format dataset (Part N-fix F1 oracle)\n",
                                     encoding="utf-8")


def _load_hgb(module, folder: Path, seed: int):
    """HGB's loader with the global generator seeded first (its split draws `random.random()`
    at construction, data_loader.py:62)."""
    state = random.getstate()
    random.seed(seed)
    try:
        return module.data_loader(str(folder))
    finally:
        random.setstate(state)


def _seeded(seed: int, call):
    state = random.getstate()
    random.seed(seed)
    try:
        return call()
    finally:
        random.setstate(state)


def _edge_index(pairs) -> torch.Tensor:
    rows = [[int(h) for h in pairs[0]], [int(t) for t in pairs[1]]]
    return torch.tensor(rows, dtype=torch.long).reshape(2, -1)


def _facts_both_ways(dl) -> torch.Tensor:
    """HGB's `links` (training facts after its split) and `links_test`, every relation: the
    `both_ways` part of `pos_links` (data_loader.py:352-355)."""
    parts = []
    for matrix in list(dl.links["data"].values()) + list(dl.links_test["data"].values()):
        row, col = matrix.nonzero()
        parts.append(torch.tensor(np.stack([row, col]), dtype=torch.long))
    return torch.cat(parts, dim=1)


def _valid_one_way(dl) -> torch.Tensor:
    """HGB's `valid_pos`, every relation, as stored (data_loader.py:356-359)."""
    parts = [_edge_index(dl.valid_pos[r]) for r in dl.valid_pos]
    return torch.cat(parts, dim=1)


def _tail_range(dl, relation: int, links_key: str = "links_test") -> tuple[int, int]:
    _, t_type = getattr(dl, links_key)["meta"][relation]
    start = dl.nodes["shift"][t_type]
    return start, start + dl.nodes["count"][t_type]


def _hgb_negatives(test_neigh, test_label, relation: int) -> list[tuple[int, int]]:
    heads, tails = test_neigh[relation]
    return [(int(h), int(t)) for h, t, y in zip(heads, tails, test_label[relation]) if y == 0]


def _ours(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    return [(int(h), int(t)) for h, t in edge_index.t().tolist()]


@unittest.skipUnless(HGB_LOADER.exists(), SKIP_REASON)
class HGBOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hgb = _hgb_module()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folders = {}
        for name, build in DATASETS.items():
            folder = Path(cls.tmp.name) / name
            _write(folder, build())
            cls.folders[name] = folder

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _cases(self):
        for name, folder in self.folders.items():
            for seed in SEEDS:
                yield name, seed, _load_hgb(self.hgb, folder, seed)

    def test_datasets_cover_the_cards_cases(self) -> None:
        """The three cases of the card hold on HGB's own loading of the files."""
        dl = _load_hgb(self.hgb, self.folders["head_without_candidates"], 0)
        test_neigh, test_label = _seeded(0, dl.get_test_neigh)
        positives = {int(h) for h, y in zip(test_neigh[0][0], test_label[0]) if y == 1}
        negatives = {int(h) for h, y in zip(test_neigh[0][0], test_label[0]) if y == 0}
        self.assertIn(4, positives)
        self.assertNotIn(4, negatives)          # a head without 2-hop candidates
        self.assertTrue(negatives)              # and heads with candidates
        two = _load_hgb(self.hgb, self.folders["two_types_two_relations"], 0)
        self.assertEqual(len(two.nodes["count"]), 2)
        self.assertEqual(sorted(two.links["meta"]), [0, 1])
        one_way = [_load_hgb(self.hgb, self.folders["one_way_validation"], s) for s in SEEDS]
        self.assertTrue(any(len(dl.valid_pos[0][0]) > 0 for dl in one_way))

    def test_two_hop_set_equals_get_test_neigh(self) -> None:
        for name, seed, dl in self._cases():
            for relation in dl.test_types:
                with self.subTest(dataset=name, seed=seed, relation=relation):
                    test_neigh, test_label = dl.get_test_neigh()  # seeds random with 1 itself
                    expected = _hgb_negatives(test_neigh, test_label, relation)
                    ours = hgb_test_negatives(
                        _edge_index(dl.links_test["data"][relation].nonzero()),
                        _facts_both_ways(dl), _valid_one_way(dl), dl.nodes["total"],
                        _tail_range(dl, relation))
                    self.assertEqual(_ours(ours.edge_index), expected)

    def test_random_set_equals_get_test_neigh_w_random(self) -> None:
        for name, seed, dl in self._cases():
            for relation in dl.test_types:
                with self.subTest(dataset=name, seed=seed, relation=relation):
                    test_neigh, test_label = dl.get_test_neigh_w_random()  # seeds with 1
                    expected = _hgb_negatives(test_neigh, test_label, relation)
                    ours = hgb_random_test_negatives(
                        _edge_index(dl.links_test["data"][relation].nonzero()),
                        _facts_both_ways(dl), _valid_one_way(dl), dl.nodes["total"],
                        _tail_range(dl, relation))
                    self.assertEqual(_ours(ours), expected)

    def test_training_negatives_equal_get_train_neg(self) -> None:
        for name, seed, dl in self._cases():
            for relation in dl.links["data"]:
                with self.subTest(dataset=name, seed=seed, relation=relation):
                    expected = _seeded(seed, lambda: dl.get_train_neg(edge_types=[relation]))
                    ours = hgb_train_negatives(_edge_index(dl.train_pos[relation]),
                                               _tail_range(dl, relation, "links"), seed)
                    self.assertEqual(_ours(ours), [(int(h), int(t)) for h, t in
                                                   zip(*expected[relation])])

    def test_validation_negatives_equal_get_valid_neg(self) -> None:
        for name, seed, dl in self._cases():
            for relation in dl.links["data"]:
                with self.subTest(dataset=name, seed=seed, relation=relation):
                    expected = _seeded(seed, lambda: dl.get_valid_neg(edge_types=[relation]))
                    ours = hgb_valid_negatives(_edge_index(dl.valid_pos[relation]),
                                               _tail_range(dl, relation, "links"), seed)
                    self.assertEqual(_ours(ours), [(int(h), int(t)) for h, t in
                                                   zip(*expected[relation])])

    def test_validation_split_equals_get_train_valid_pos(self) -> None:
        """HGB splits at construction: relations in `link.dat` order, one global stream seeded
        here with `seed`; ours (`_split_edge_index_hgb_by_head`) gets `seed` for the first
        relation (our loader's `split_seed + relation_id * 9973` with relation id 0), the ratio
        1 - train_ratio = 0.10 (data_loader.py:44) and the same stored facts. Only the first
        relation is compared: HGB continues one stream across relations, ours seeds each."""
        for name, seed, dl in self._cases():
            first = next(iter(dl.links["data"]))
            with self.subTest(dataset=name, seed=seed, relation=first):
                stored = [row for row in DATASETS[name]()["links"] if row[2] == first]
                edge_index = torch.tensor([[h for h, _, _ in stored], [t for _, t, _ in stored]])
                train, valid = _split_edge_index_hgb_by_head(edge_index, 0.10, seed)
                self.assertEqual(_ours(valid), [(int(h), int(t)) for h, t in
                                                zip(*dl.valid_pos[first])])
                self.assertEqual(_ours(train), [(int(h), int(t)) for h, t in
                                                zip(*dl.train_pos[first])])


if __name__ == "__main__":
    unittest.main()
