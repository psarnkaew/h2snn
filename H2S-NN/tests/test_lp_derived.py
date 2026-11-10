"""J1 tests (task card Part J): derived link-prediction tasks, their split, negatives and loader.

Every property is tested on a small synthetic HGB-format release built in a temporary folder. The
real derived tasks under `data/lp_derived/` are also checked when they have been built.
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.lp import build_derived_lp as bd
from scripts.lp.apply_feature_rule import apply_rule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED_ROOT = PROJECT_ROOT / "data" / "lp_derived"

# Synthetic release laid out like HGB ACM: type 0 paper, 1 author, 2 subject, 3 term; relation
# 2 = paper->author (target), 3 = its reverse, 4/5 paper<->subject, 0/1 cite/ref.
N_PAPER, N_AUTHOR, N_SUBJECT, N_TERM = 60, 40, 4, 6
PAPER0, AUTHOR0, SUBJECT0, TERM0 = 0, 60, 100, 104
N_TARGET_FACTS = 200


def write_release(folder: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    nodes = []
    for i in range(N_PAPER):
        nodes.append(f"{PAPER0 + i}\tpaper{i}\t0\t0.1,0.{i % 10}\n")
    for i in range(N_AUTHOR):
        nodes.append(f"{AUTHOR0 + i}\tauthor{i}\t1\t0.3,0.{i % 10}\n")
    for i in range(N_SUBJECT):
        nodes.append(f"{SUBJECT0 + i}\tsubject{i}\t2\t0.5,0.6\n")
    for i in range(N_TERM):
        nodes.append(f"{TERM0 + i}\tterm{i}\t3\n")
    (folder / "node.dat").write_bytes("".join(nodes).encode())
    writes = rng.sample([(p, a) for p in range(N_PAPER) for a in range(N_AUTHOR)], N_TARGET_FACTS)
    cites = rng.sample([(p, q) for p in range(N_PAPER) for q in range(N_PAPER) if p != q], 50)
    links = []
    for p, a in writes:
        links.append(f"{PAPER0 + p}\t{AUTHOR0 + a}\t2\t1.0\n")
    for p, q in cites:
        links.append(f"{PAPER0 + p}\t{PAPER0 + q}\t0\t1.0\n")
    for p in range(N_PAPER):
        links.append(f"{PAPER0 + p}\t{SUBJECT0 + p % N_SUBJECT}\t4\t1.0\n")
    for p, a in writes:
        links.append(f"{AUTHOR0 + a}\t{PAPER0 + p}\t3\t1.0\n")
    for p, q in cites:
        links.append(f"{PAPER0 + q}\t{PAPER0 + p}\t1\t1.0\n")
    for p in range(N_PAPER):
        links.append(f"{SUBJECT0 + p % N_SUBJECT}\t{PAPER0 + p}\t5\t1.0\n")
    (folder / "link.dat").write_bytes("".join(links).encode())
    (folder / "info.dat").write_text(json.dumps({"synthetic": True}), encoding="utf-8")


def node_ids(task_dir: Path) -> set[int]:
    return {int(line.split(b"\t", 1)[0]) for line in (task_dir / "node.dat").read_bytes().splitlines()}


class SplitPropertyMixin:
    """The J1 split properties, checked on one built task directory."""

    task_dir: Path
    relation: int
    reverse: int
    tolerance_pp: float

    def test_parts_are_disjoint_by_fact_in_either_orientation(self) -> None:
        split = bd.read_split(self.task_dir, self.relation)
        parts = {"train": split.train, "valid": split.valid, "test": split.test}
        for name_a, a in parts.items():
            for name_b, b in parts.items():
                if name_a < name_b:
                    both = a | {(v, u) for u, v in a}
                    self.assertFalse(both & b, f"{name_a} and {name_b} share a fact")

    def test_proportions_are_81_9_10(self) -> None:
        split = bd.read_split(self.task_dir, self.relation)
        total = len(split.train) + len(split.valid) + len(split.test)
        for part, target in ((split.train, 81.0), (split.valid, 9.0), (split.test, 10.0)):
            self.assertLessEqual(abs(100.0 * len(part) / total - target), self.tolerance_pp)

    def test_every_node_of_a_test_fact_is_in_node_dat(self) -> None:
        ids = node_ids(self.task_dir)
        split = bd.read_split(self.task_dir, self.relation)
        self.assertTrue(all(u in ids and v in ids for u, v in split.test))

    def test_no_test_fact_is_left_in_link_dat(self) -> None:
        self.assertEqual(bd.test_fact_leaks(self.task_dir, self.relation, self.reverse), 0)


class SyntheticSplitTests(SplitPropertyMixin, unittest.TestCase):
    relation, reverse, tolerance_pp = 2, 3, 0.5

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        write_release(root / "release")
        self.source = root / "release"
        self.task_dir = root / "lp_derived" / "acm_pa"
        self.meta = bd.build_task(self.source, self.task_dir, 2, seed=42, val_frac=0.09,
                                  test_frac=0.10)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_counts_and_reverse_relation(self) -> None:
        self.assertEqual(self.meta["reverse_relation"], 3)
        self.assertEqual((self.meta["train"], self.meta["valid"], self.meta["test"]),
                         (162, 18, 20))

    def test_same_seed_gives_byte_identical_files(self) -> None:
        again = self.task_dir.parent / "again"
        other = self.task_dir.parent / "other_seed"
        bd.build_task(self.source, again, 2, seed=42, val_frac=0.09, test_frac=0.10)
        bd.build_task(self.source, other, 2, seed=43, val_frac=0.09, test_frac=0.10)
        for name in ("node.dat", "info.dat", "link.dat", "link.dat.test", "link.dat.valid",
                     "split_meta.json"):
            self.assertEqual((self.task_dir / name).read_bytes(), (again / name).read_bytes(), name)
        self.assertNotEqual((self.task_dir / "link.dat.test").read_bytes(),
                            (other / "link.dat.test").read_bytes())  # the seed matters

    def test_link_dat_is_the_release_minus_the_test_rows(self) -> None:
        source = (self.source / "link.dat").read_bytes().splitlines(keepends=True)
        derived = (self.task_dir / "link.dat").read_bytes().splitlines(keepends=True)
        self.assertEqual(len(derived), len(source) - 2 * self.meta["test"])
        rest = iter(source)
        self.assertTrue(all(any(line == candidate for candidate in rest) for line in derived),
                        "link.dat is not an order-preserving subset of the release's rows")
        valid = bd.read_split(self.task_dir, 2).valid
        target = bd.read_split(self.task_dir, 2).link_pairs_by_relation[2]
        self.assertTrue(valid <= target)  # validation facts stay in link.dat (HGB format)

    def test_a_reverse_only_leak_is_caught_at_file_level(self) -> None:
        """Route 2 of the pre-registration: a test fact back in link.dat only as an r' row."""
        u, v = sorted(bd.read_split(self.task_dir, 2).test)[0]
        with (self.task_dir / "link.dat").open("ab") as handle:
            handle.write(f"{v}\t{u}\t3\t1.0\n".encode())
        self.assertEqual(bd.test_fact_leaks(self.task_dir, 2, 3), 1)


def derived_config(root: Path, dataset: str, target: tuple[str, str, str], seed: int = 3):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({"device": "cpu", "seed": seed, "data": {
        "dataset_name": dataset, "task_type": "link_prediction", "root": str(root),
        "target_relation": list(target), "fact_layer": "tied", "path_sampler": "walk",
        "face_sample_k": 4}})


class SyntheticLoaderTests(unittest.TestCase):
    """J1.3: the registered derived task loads through the existing leak checks."""

    TARGET = ("paper", "rel_2", "author")

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_release(self.root / "release")
        self.task_dir = self.root / "lp_derived" / "acm_pa"
        bd.build_task(self.root / "release", self.task_dir, 2, seed=42, val_frac=0.09,
                      test_frac=0.10)
        apply_rule("acm_pa", self.task_dir)  # F17/F20: the loader needs the rule applied

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _load(self, seed: int = 3):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(derived_config(self.root, "acm_pa", self.TARGET, seed))

    def test_loads_and_its_test_set_is_link_dat_test(self) -> None:
        data = self._load()
        src0, dst0 = data.src_node_offset, data.dst_node_offset
        loaded = {(int(u) - src0 + PAPER0, int(v) - dst0 + AUTHOR0)
                  for u, v in data.test_pos_edge_index.t().tolist()}
        self.assertEqual(loaded, bd.read_split(self.task_dir, 2).test)

    def test_validation_split_does_not_depend_on_the_training_seed(self) -> None:
        a, b = self._load(seed=3), self._load(seed=11)
        for name in ("val_pos_edge_index", "val_neg_edge_index", "test_pos_edge_index",
                     "test_neg_edge_index", "train_pos_edge_index"):
            self.assertTrue(bool((getattr(a, name) == getattr(b, name)).all()), name)

    def _append(self, rows: list[str]) -> None:
        with (self.task_dir / "link.dat").open("ab") as handle:
            handle.write("".join(rows).encode())

    def test_mutation_a_test_fact_back_in_link_dat_is_caught(self) -> None:
        u, v = sorted(bd.read_split(self.task_dir, 2).test)[0]
        self._append([f"{u}\t{v}\t2\t1.0\n", f"{v}\t{u}\t3\t1.0\n"])  # as HGB stores a fact
        with self.assertRaisesRegex(ValueError, "leak|overlap"):
            self._load()

    def test_mutation_a_test_fact_back_as_its_target_row_is_caught(self) -> None:
        u, v = sorted(bd.read_split(self.task_dir, 2).test)[0]
        self._append([f"{u}\t{v}\t2\t1.0\n"])
        with self.assertRaisesRegex(ValueError, "leak|overlap"):
            self._load()


N_BOOK, N_FILM = 50, 10
N_BOOK_PAIRS, N_BOOK_LOOPS = 120, 5


def write_freebase_like(folder: Path, seed: int = 0) -> None:
    """A release laid out like HGB Freebase: type 0 book, 1 film, no features; relation 0 =
    book-book with no stored reverse relation, 120 non-loop pairs of which about 60 % are stored in
    both orientations, plus 5 self-loops; relation 1 = book->film."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    nodes = [f"{i}\tbook{i}\t0\n" for i in range(N_BOOK)]
    nodes += [f"{N_BOOK + i}\tfilm{i}\t1\n" for i in range(N_FILM)]
    (folder / "node.dat").write_bytes("".join(nodes).encode())
    pairs = rng.sample([(a, b) for a in range(N_BOOK) for b in range(a + 1, N_BOOK)], N_BOOK_PAIRS)
    links = []
    for a, b in pairs:
        r = rng.random()
        if r < 0.6:
            links += [f"{a}\t{b}\t0\t1.0\n", f"{b}\t{a}\t0\t1.0\n"]
        else:
            links.append(f"{a}\t{b}\t0\t1.0\n" if r < 0.8 else f"{b}\t{a}\t0\t1.0\n")
    links += [f"{i}\t{i}\t0\t1.0\n" for i in range(N_BOOK_LOOPS)]
    links += [f"{a}\t{N_BOOK + a % N_FILM}\t1\t1.0\n" for a in range(N_BOOK)]
    (folder / "link.dat").write_bytes("".join(links).encode())
    (folder / "info.dat").write_text(json.dumps({"synthetic": True}), encoding="utf-8")


def canonical(rows: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Own helper (not the build script's): each row as its unordered pair."""
    return {(min(u, v), max(u, v)) for u, v in rows}


class FreebasePairSplitTests(unittest.TestCase):
    """J4.3 (spec v1.8 F10): BOOK-and-BOOK split 81/9/10 by unordered pair, self-loops in no part."""

    TARGET = ("book", "rel_0", "book")

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_freebase_like(self.root / "release")
        self.task_dir = self.root / "lp_derived" / "freebase_bb"
        self.meta = bd.build_task_by_pair(self.root / "release", self.task_dir, 0, seed=42,
                                          val_frac=0.09, test_frac=0.10)
        apply_rule("freebase_bb", self.task_dir)  # F17/F20

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _parts(self) -> dict[str, set[tuple[int, int]]]:
        split = bd.read_split(self.task_dir, 0)
        train = {p for p in split.train if p[0] != p[1]}
        return {"train": train, "valid": split.valid, "test": split.test}

    def test_no_unordered_pair_has_rows_in_two_parts(self) -> None:
        parts = {name: canonical(rows) for name, rows in self._parts().items()}
        self.assertFalse(parts["train"] & parts["valid"])
        self.assertFalse(parts["train"] & parts["test"])
        self.assertFalse(parts["valid"] & parts["test"])
        self.assertEqual(sum(len(p) for p in parts.values()), N_BOOK_PAIRS)

    def test_reciprocal_leak_no_held_out_pair_in_link_dat_either_orientation(self) -> None:
        link = bd.read_split(self.task_dir, 0).link_pairs_by_relation[0]
        for u, v in bd.read_split(self.task_dir, 0).test:
            self.assertNotIn((u, v), link)
            self.assertNotIn((v, u), link)

    def test_every_stored_row_of_a_held_out_pair_is_in_its_file(self) -> None:
        source = {(r.src, r.dst) for r in bd.read_link_rows(self.root / "release" / "link.dat")
                  if r.relation == 0}
        split = bd.read_split(self.task_dir, 0)
        for rows in (split.test, split.valid):
            pairs = canonical(rows)
            self.assertEqual({p for p in source if p[0] != p[1] and canonical({p}) <= pairs}, rows)

    def test_self_loops_are_in_no_part_and_stay_in_link_dat(self) -> None:
        split = bd.read_split(self.task_dir, 0)
        self.assertFalse(any(u == v for u, v in split.valid | split.test))
        loops = {(i, i) for i in range(N_BOOK_LOOPS)}
        self.assertTrue(loops <= split.link_pairs_by_relation[0])
        self.assertEqual(self.meta["rows"]["self_loops_in_link_dat"], N_BOOK_LOOPS)

    def test_proportions_are_81_9_10_by_pair(self) -> None:
        counts = (self.meta["train"], self.meta["valid"], self.meta["test"])
        self.assertEqual(counts, (97, 11, 12))
        for part, share in zip(counts, (81.0, 9.0, 10.0)):
            self.assertLessEqual(abs(100.0 * part / N_BOOK_PAIRS - share), 0.5)

    def test_same_seed_gives_byte_identical_files(self) -> None:
        again = self.task_dir.parent / "again"
        bd.build_task_by_pair(self.root / "release", again, 0, seed=42, val_frac=0.09,
                              test_frac=0.10)
        apply_rule("freebase_bb", again)
        for name in ("node.dat", "info.dat", "link.dat", "link.dat.test", "link.dat.valid",
                     "split_meta.json"):
            self.assertEqual((self.task_dir / name).read_bytes(), (again / name).read_bytes(), name)

    def test_loader_leak_checks_pass_and_positives_exclude_self_loops(self) -> None:
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        data = build_link_prediction_dataset(derived_config(self.root, "freebase_bb",
                                                            self.TARGET))
        for name in ("train_pos_edge_index", "val_pos_edge_index", "test_pos_edge_index"):
            index = getattr(data, name)
            self.assertFalse(bool((index[0] == index[1]).any()), name)
        message = {tuple(p) for p in data.edge_index.t().tolist()}
        off = data.src_node_offset
        self.assertTrue({(off + i, off + i) for i in range(N_BOOK_LOOPS)} <= message)
        held = {tuple(p) for p in torch.cat([data.val_pos_edge_index,
                                             data.test_pos_edge_index], 1).t().tolist()}
        self.assertFalse(held & message)
        self.assertFalse({(v, u) for u, v in held} & message)
        self.assertEqual(data.train_pos_edge_index.size(1), self.meta["rows"]["train"])

    def _load(self):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(derived_config(self.root, "freebase_bb", self.TARGET))

    def test_loader_raises_on_a_reversed_test_pair_in_link_dat(self) -> None:
        """Independent review, finding 3 (A): the reverse row of a one-row test pair put back."""
        test = bd.read_split(self.task_dir, 0).test
        u, v = next((u, v) for u, v in sorted(test) if (v, u) not in test)
        with (self.task_dir / "link.dat").open("ab") as handle:
            handle.write(f"{v}\t{u}\t0\t1.0\n".encode())
        with self.assertRaisesRegex(ValueError, "leak"):
            self._load()

    def test_loader_raises_on_a_one_orientation_validation_file(self) -> None:
        """Independent review, finding 3 (B): link.dat.valid lists one row of a two-row pair."""
        lines = (self.task_dir / "link.dat.valid").read_bytes().splitlines(keepends=True)
        rows = [tuple(map(int, line.split(b"\t")[:2])) for line in lines]
        u, v = next((u, v) for u, v in rows if (v, u) in set(rows))
        kept = [line for line, row in zip(lines, rows) if row != (v, u)]
        (self.task_dir / "link.dat.valid").write_bytes(b"".join(kept))
        with self.assertRaisesRegex(ValueError, "leak"):
            self._load()


def _real_task(name: str) -> Path | None:
    task_dir = DERIVED_ROOT / name
    return task_dir if (task_dir / "split_meta.json").is_file() else None


def _real_split_case(name: str) -> type:
    task_dir = _real_task(name)

    @unittest.skipUnless(task_dir is not None, f"data/lp_derived/{name} not built")
    class RealSplitTests(SplitPropertyMixin, unittest.TestCase):
        tolerance_pp = 0.5

        @classmethod
        def setUpClass(cls) -> None:
            meta = json.loads((task_dir / "split_meta.json").read_text(encoding="utf-8"))
            cls.task_dir = task_dir
            cls.relation, cls.reverse = meta["target_relation"], meta["reverse_relation"]

    RealSplitTests.__name__ = RealSplitTests.__qualname__ = f"RealSplitTests_{name}"
    return RealSplitTests


@unittest.skipUnless(_real_task("freebase_bb") is not None, "data/lp_derived/freebase_bb not built")
class RealFreebasePairTests(unittest.TestCase):
    """The built freebase_bb: disjoint by unordered pair, no reciprocal leak, no self-loop held
    out, 81/9/10 by pair."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.task_dir = DERIVED_ROOT / "freebase_bb"
        cls.meta = json.loads((cls.task_dir / "split_meta.json").read_text(encoding="utf-8"))
        cls.split = bd.read_split(cls.task_dir, 0)

    def test_disjoint_by_unordered_pair(self) -> None:
        train = canonical({p for p in self.split.train if p[0] != p[1]})
        valid, test = canonical(self.split.valid), canonical(self.split.test)
        self.assertFalse(train & valid or train & test or valid & test)
        self.assertEqual((len(train), len(valid), len(test)),
                         (self.meta["train"], self.meta["valid"], self.meta["test"]))

    def test_no_reciprocal_leak_and_no_held_out_self_loop(self) -> None:
        self.assertEqual(bd.test_fact_leaks(self.task_dir, 0, 0), 0)
        self.assertFalse(any(u == v for u, v in self.split.valid | self.split.test))

    def test_proportions_by_pair(self) -> None:
        total = self.meta["facts"]
        for key, share in (("train", 81.0), ("valid", 9.0), ("test", 10.0)):
            self.assertLessEqual(abs(100.0 * self.meta[key] / total - share), 0.5)


RealSplitTestsACM = _real_split_case("acm_pa")
RealSplitTestsDBLP = _real_split_case("dblp_ap")
RealSplitTestsIMDB = _real_split_case("imdb_ma")


if __name__ == "__main__":
    unittest.main()
