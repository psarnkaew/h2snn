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

    def test_hgb_rule_replayed_over_our_facts_equals_get_train_valid_pos(self) -> None:
        """The documented difference (spec draft v1.18 Z-4; author decision (a), 2026-11-12):
        HGB's carve (data_loader.py:44-73) replayed over our sorted facts with one
        `random.Random(seed)` stream over the relations in `link.dat` order -- first fact of a
        head to training, else `u >= 0.9` to validation (`random.random() < 0.9` keeps it in
        training, :62), no minimum -- equals HGB's `get_train_valid_pos` exactly. Fact order and
        rule agree; only the draw differs from our fixed files (our carve draws `u < 0.10`, one
        seeded stream per relation, at least one fact)."""
        for name, seed, dl in self._cases():
            rows = DATASETS[name]()["links"]   # written to link.dat in this order
            relation_order = list(dict.fromkeys(r for _, _, r in rows))
            rng = random.Random(seed)
            for relation in relation_order:
                with self.subTest(dataset=name, seed=seed, relation=relation):
                    facts = sorted({(h, t) for h, t, r in rows if r == relation})
                    train, valid, last_head = [], [], None
                    for head, tail in facts:
                        if head != last_head:
                            train.append((head, tail))
                            last_head = head
                        elif rng.random() < 0.9:
                            train.append((head, tail))
                        else:
                            valid.append((head, tail))
                    self.assertEqual(valid, [(int(h), int(t))
                                             for h, t in zip(*dl.valid_pos[relation])])
                    self.assertEqual(train, [(int(h), int(t))
                                             for h, t in zip(*dl.train_pos[relation])])
            self.assertEqual(list(dl.links["data"]), relation_order)


#: The six local LP tasks and their targets (spec v1.16 §6); on the three derived tasks whose
#: target relation has a stored reverse relation, the validation file also lists the reverse
#: copies (decision 5 of the Part N review, confirmed).
LOCAL_TASKS: dict[str, tuple[tuple[str, str, str], bool]] = {
    "acm_pa": (("paper", "rel_2", "author"), True),
    "dblp_ap": (("author", "rel_0", "paper"), True),
    "imdb_ma": (("movie", "rel_2", "actor"), True),
    "freebase_bb": (("book", "rel_0", "book"), False),
    "pubmed": (("disease", "rel_2", "disease"), False),
    "lastfm": (("user", "rel_0", "artist"), False),
}


def _pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(h), int(t)) for h, t in edge_index.reshape(2, -1).t().tolist()}


class FixedValidationFileTests(unittest.TestCase):
    """Our side of the documented difference (draft v1.18 Z-4): every local LP task reads its
    fixed `link.dat.valid`, which lists the target relation only (spec 1.9 F13); the loader's
    validation facts of the target relation are exactly the file's rows; on acm_pa, dblp_ap and
    imdb_ma the stored reverse relation takes those rows flipped (`_file_validation_facts`:
    the partner's rows flipped; decision 5 of the Part N review); every fact of every other
    relation in `link.dat` stays in the message graph."""

    def test_loader_reads_the_fixed_validation_file_exactly(self) -> None:
        from src.configs.config import TrainConfig
        from src.data.link_prediction_dataset import (_load_local_hgb_link_data,
                                                      local_hgb_link_dir)

        root = PROJECT_ROOT / "data"
        for task, (target, has_reverse) in LOCAL_TASKS.items():
            with self.subTest(task=task):
                folder = local_hgb_link_dir(root, task)
                valid_file = folder / "link.dat.valid"
                self.assertTrue(valid_file.exists(), f"{task}: no link.dat.valid")
                config = TrainConfig.from_dict({"device": "cpu", "data": {
                    "dataset_name": task, "task_type": "link_prediction", "root": str(root),
                    "target_relation": list(target), "wisp_walks_per_anchor": 1,
                    "wisp_max_length": 2}})
                split = _load_local_hgb_link_data(config)

                file_rows: dict[tuple[str, str, str], set[tuple[int, int]]] = {}
                count = 0
                with valid_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        head, tail, relation = (int(x) for x in line.split("\t")[:3])
                        key = (split.raw_to_type_name[head], f"rel_{relation}",
                               split.raw_to_type_name[tail])
                        file_rows.setdefault(key, set()).add(
                            (split.raw_to_local[head], split.raw_to_local[tail]))
                        count += 1
                self.assertEqual(count, sum(len(v) for v in file_rows.values()))  # no dupes
                self.assertEqual(set(file_rows), {target})   # F13: the target relation only

                expected = {target: file_rows[target]}
                if has_reverse:
                    reverse = [r for r in split.link_edge_index_by_relation
                               if (r[0], r[2]) == (target[2], target[0])]
                    self.assertEqual(len(reverse), 1, f"{task}: stored reverse relation")
                    expected[reverse[0]] = {(t, h) for h, t in file_rows[target]}

                for relation, link_edges in split.link_edge_index_by_relation.items():
                    valid = split.valid_edge_index_by_relation.get(relation)
                    valid_pairs = _pairs(valid) if valid is not None else set()
                    train_pairs = _pairs(split.train_edge_index_by_relation[relation])
                    self.assertEqual(valid_pairs, expected.get(relation, set()),
                                     f"{task} {relation}: validation != file rows")
                    if relation in expected:
                        self.assertFalse(train_pairs & valid_pairs)
                        self.assertEqual(train_pairs | valid_pairs, _pairs(link_edges))
                    else:  # every other relation complete in the message graph
                        self.assertEqual(train_pairs, _pairs(link_edges),
                                         f"{task} {relation}: message graph incomplete")


if __name__ == "__main__":
    unittest.main()
