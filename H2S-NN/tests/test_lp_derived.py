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

from scripts.lp import build_derived_lp as bd

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


RealSplitTestsACM = _real_split_case("acm_pa")
RealSplitTestsDBLP = _real_split_case("dblp_ap")
RealSplitTestsIMDB = _real_split_case("imdb_ma")


if __name__ == "__main__":
    unittest.main()
