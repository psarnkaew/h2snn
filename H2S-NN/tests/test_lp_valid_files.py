"""J4.2 (task card Part J, spec v1.8 F9) and J5.4 (spec v1.9 F13): the loader reads `link.dat.valid`, guards the
reverse-only leak; the files written for PubMed and LastFM list only the target relation, with the
carve's target facts (J4.2 wrote every relation's carve; still reachable with --mode carve).

Synthetic HGB-format releases in temporary folders carry every property; the real PubMed,
LastFM and derived tasks are checked when they are on disk.
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
from tests.test_lp_derived import AUTHOR0, PAPER0, derived_config, write_release
from src.data.data_layout import REPO_DATA_ROOT, lp_derived_root, lp_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_DATA_ROOT


def write_pubmed_like(folder: Path, seed: int = 0) -> None:
    """A release laid out like HGB PubMed: type 1 = disease; relation 2 = disease-disease,
    self-inverse (both orientations stored) with two self-loops; relation 0 = type-0 pairs,
    self-inverse; relation 1 = type 0 -> disease (no inverse). `link.dat.test` holds 12
    disease pairs in both orientations, none in `link.dat`."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    n0, n1 = 30, 40  # raw ids: type 0 = 0..29, disease = 30..69
    nodes = [f"{i}\tc{i}\t0\t0.1,0.2\n" for i in range(n0)]
    nodes += [f"{n0 + i}\td{i}\t1\t0.3,0.{i % 10}\n" for i in range(n1)]
    (folder / "node.dat").write_bytes("".join(nodes).encode())
    pairs = rng.sample([(a, b) for a in range(n1) for b in range(a + 1, n1)], 112)
    train_pairs, test_pairs = pairs[:100], pairs[100:]
    links = []
    for a, b in train_pairs:
        links += [f"{n0 + a}\t{n0 + b}\t2\t3\n", f"{n0 + b}\t{n0 + a}\t2\t3\n"]
    links += [f"{n0 + 5}\t{n0 + 5}\t2\t1\n", f"{n0 + 7}\t{n0 + 7}\t2\t1\n"]
    for a, b in rng.sample([(a, b) for a in range(n0) for b in range(a + 1, n0)], 40):
        links += [f"{a}\t{b}\t0\t1\n", f"{b}\t{a}\t0\t1\n"]
    for a, b in rng.sample([(a, b) for a in range(n0) for b in range(n1)], 60):
        links.append(f"{a}\t{n0 + b}\t1\t2\n")
    (folder / "link.dat").write_bytes("".join(links).encode())
    test = []
    for a, b in test_pairs:
        test += [f"{n0 + a}\t{n0 + b}\t2\t1\n", f"{n0 + b}\t{n0 + a}\t2\t1\n"]
    (folder / "link.dat.test").write_bytes("".join(test).encode())


def pubmed_config(root: Path, seed: int = 3):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({"device": "cpu", "seed": seed, "data": {
        "dataset_name": "pubmed", "root": str(root), "fact_layer": "tied",
        "path_sampler": "walk", "face_sample_k": 4}})


def split_of(root: Path, dataset: str, config) -> tuple[dict, dict]:
    """(train, validation) per relation exactly as the loader builds them, with the file (if
    any) and with the carve (file ignored)."""
    from src.data.fact_layer import inverse_relations_typed
    from src.data.link_prediction_dataset import (_LOCAL_HGB_LINK_DATASETS,
                                                  _load_local_hgb_nodes,
                                                  _read_local_hgb_link_file, _split_validation)

    spec = _LOCAL_HGB_LINK_DATASETS[dataset]
    folder = lp_root(root) / spec["folder"]
    nodes = _load_local_hgb_nodes(folder, spec["node_type_names"])
    link = _read_local_hgb_link_file(folder / "link.dat", nodes)
    test = _read_local_hgb_link_file(folder / "link.dat.test", nodes)
    relations = sorted(set(link) | set(test))
    rho = inverse_relations_typed(link, {t: int(nodes.data[t].num_nodes)
                                         for t in nodes.data.node_types})
    valid_path = folder / "link.dat.valid"
    file_valid = _read_local_hgb_link_file(valid_path, nodes) if valid_path.exists() else None
    kwargs = dict(val_ratio=config.data.link_val_ratio, split_seed=config.data.acm_split_seed)
    return (_split_validation(link, rho, relations, file_valid, **kwargs),
            _split_validation(link, rho, relations, None, **kwargs))


def assert_same_split(case: unittest.TestCase, a: tuple[dict, dict], b: tuple[dict, dict]) -> None:
    for part_a, part_b in zip(a, b):
        case.assertEqual(sorted(part_a), sorted(part_b))
        for relation in part_a:
            case.assertTrue(torch.equal(part_a[relation], part_b[relation]), relation)


class FileWrittenFromTheCarveTests(unittest.TestCase):
    """J4.2 (c) on a PubMed-like release: the file reproduces the carve, tensor for tensor."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_pubmed_like(lp_root(self.root) / "PubMed")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _load(self):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(pubmed_config(self.root))

    def test_written_file_reproduces_the_loaders_carve(self) -> None:
        from scripts.lp.write_valid_files import carve_rows

        before = self._load()
        content, summary = carve_rows(self.root, "pubmed")
        self.assertGreater(summary["rows_per_relation"]["rel_2"], 0)
        (lp_root(self.root) / "PubMed" / "link.dat.valid").write_bytes(content)
        with_file, carve = split_of(self.root, "pubmed", pubmed_config(self.root))
        assert_same_split(self, with_file, carve)
        after = self._load()
        for name in ("train_pos_edge_index", "val_pos_edge_index", "val_neg_edge_index",
                     "test_pos_edge_index", "test_neg_edge_index", "edge_index", "edge_type"):
            self.assertTrue(torch.equal(getattr(before, name), getattr(after, name)), name)

    def test_self_inverse_relation_listed_in_one_orientation_raises(self) -> None:
        from scripts.lp.write_valid_files import carve_rows

        content, _ = carve_rows(self.root, "pubmed")
        rows = [r for r in content.decode().splitlines(keepends=True)
                if r.split("\t")[2] != "2"]
        one = next(r for r in content.decode().splitlines(keepends=True)
                   if r.split("\t")[2] == "2" and r.split("\t")[0] != r.split("\t")[1])
        (lp_root(self.root) / "PubMed" / "link.dat.valid").write_bytes("".join(rows + [one]).encode())
        with self.assertRaisesRegex(ValueError, "one orientation"):
            self._load()


class DerivedTaskValidFileTests(unittest.TestCase):
    """J4.2 (a), (b) on a synthetic derived task (ACM-like, target paper->author)."""

    TARGET = ("paper", "rel_2", "author")

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_release(self.root / "release")
        self.task_dir = lp_derived_root(self.root) / "acm_pa"
        bd.build_task(self.root / "release", self.task_dir, 2, seed=42, val_frac=0.09,
                      test_frac=0.10)
        apply_rule("acm_pa", self.task_dir)  # F17/F20

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _load(self):
        from src.data.link_prediction_dataset import build_link_prediction_dataset

        return build_link_prediction_dataset(derived_config(self.root, "acm_pa", self.TARGET))

    def test_validation_is_the_file_and_the_split_is_81_9_10(self) -> None:
        data = self._load()
        src0, dst0 = data.src_node_offset, data.dst_node_offset
        split = bd.read_split(self.task_dir, 2)
        loaded = {(int(u) - src0 + PAPER0, int(v) - dst0 + AUTHOR0)
                  for u, v in data.val_pos_edge_index.t().tolist()}
        self.assertEqual(loaded, split.valid)
        counts = (data.train_pos_edge_index.size(1), data.val_pos_edge_index.size(1),
                  data.test_pos_edge_index.size(1))
        self.assertEqual(counts, (162, 18, 20))

    def test_validation_facts_leave_the_message_graph_in_both_orientations(self) -> None:
        data = self._load()
        val = {tuple(p) for p in data.val_pos_edge_index.t().tolist()}
        message = {tuple(p) for p in data.edge_index.t().tolist()}
        self.assertFalse(val & message)
        self.assertFalse({(v, u) for u, v in val} & message)

    def test_non_target_relations_absent_from_the_file_stay_whole(self) -> None:
        with_file, _ = split_of(self.root, "acm_pa",
                                derived_config(self.root, "acm_pa", self.TARGET))
        train, valid = with_file
        for relation, edges in valid.items():
            if relation[1] not in ("rel_2", "rel_3"):
                self.assertEqual(edges.size(1), 0, relation)

    def test_a_validation_fact_missing_from_link_dat_raises(self) -> None:
        split = bd.read_split(self.task_dir, 2)
        stored = split.link_pairs_by_relation[2] | split.test
        u, v = next((PAPER0 + p, AUTHOR0 + a) for p in range(60) for a in range(40)
                    if (PAPER0 + p, AUTHOR0 + a) not in stored)
        with (self.task_dir / "link.dat.valid").open("ab") as handle:
            handle.write(f"{u}\t{v}\t2\t1.0\n".encode())
        with self.assertRaisesRegex(ValueError, "not in link.dat"):
            self._load()

    def test_reverse_only_fixture_raises(self) -> None:
        """J1.3 finding 2: a test fact back in link.dat only as its r' row breaks rho; the
        guard raises before the split."""
        u, v = sorted(bd.read_split(self.task_dir, 2).test)[0]
        with (self.task_dir / "link.dat").open("ab") as handle:
            handle.write(f"{v}\t{u}\t3\t1.0\n".encode())
        with self.assertRaisesRegex(ValueError, "reverse-typed"):
            self._load()


def _real(dataset_dir: Path) -> bool:
    return (dataset_dir / "link.dat").is_file() and (dataset_dir / "link.dat.valid").is_file()


def assert_target_only(case: unittest.TestCase, with_file: tuple[dict, dict],
                       carve: tuple[dict, dict], link: dict, target: str) -> None:
    """Spec v1.9 F13: the target's split equals the carve's; every other relation keeps all its
    link.dat facts (no validation)."""
    (train_f, valid_f), (train_c, valid_c) = with_file, carve
    case.assertEqual(sorted(train_f), sorted(train_c))
    for relation in train_f:
        if relation[1] == target:
            case.assertTrue(torch.equal(train_f[relation], train_c[relation]), relation)
            case.assertTrue(torch.equal(valid_f[relation], valid_c[relation]), relation)
        else:
            case.assertEqual(valid_f[relation].size(1), 0, relation)
            case.assertEqual(train_f[relation].size(1),
                             train_c[relation].size(1) + valid_c[relation].size(1), relation)
            if relation in link:
                case.assertEqual(train_f[relation].size(1), link[relation].size(1), relation)


def loaded_split(config) -> tuple[dict, dict]:
    """(train, validation) per relation through the loader's own path (it reads the file)."""
    from src.data.link_prediction_dataset import _load_local_hgb_link_data

    split = _load_local_hgb_link_data(config)
    return split.train_edge_index_by_relation, split.valid_edge_index_by_relation


def link_edges_of(root: Path, dataset: str) -> dict:
    from src.data.link_prediction_dataset import (_LOCAL_HGB_LINK_DATASETS,
                                                  _load_local_hgb_nodes,
                                                  _read_local_hgb_link_file)

    spec = _LOCAL_HGB_LINK_DATASETS[dataset]
    nodes = _load_local_hgb_nodes(lp_root(root) / spec["folder"], spec["node_type_names"])
    return _read_local_hgb_link_file(lp_root(root) / spec["folder"] / "link.dat", nodes)


class TargetOnlyFileTests(unittest.TestCase):
    """J5.4 (spec v1.9 F13) on a PubMed-like release: the target-only file keeps the carve's
    target split and leaves every other relation complete."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_pubmed_like(lp_root(self.root) / "PubMed")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_target_only_file_keeps_the_target_split_and_completes_the_rest(self) -> None:
        from scripts.lp.write_valid_files import carve_rows, target_rows

        content, summary = target_rows(self.root, "pubmed")
        carve, _ = carve_rows(self.root, "pubmed")
        self.assertEqual(summary["target_relation_ids"], [2])
        rows = content.decode().splitlines(keepends=True)
        self.assertTrue(rows and all(r.split("\t")[2] == "2" for r in rows))
        self.assertEqual(rows, [r for r in carve.decode().splitlines(keepends=True)
                                if r.split("\t")[2] == "2"])  # the same bytes as the carve's
        (lp_root(self.root) / "PubMed" / "link.dat.valid").write_bytes(content)
        _, carve_split = split_of(self.root, "pubmed", pubmed_config(self.root))
        assert_target_only(self, loaded_split(pubmed_config(self.root)), carve_split,
                           link_edges_of(self.root, "pubmed"), "rel_2")


class RealValidFileTests(unittest.TestCase):
    """The real files: PubMed and LastFM list only the target relation, with the carve's target
    facts (spec v1.9 F13); the derived tasks evaluate the J1.2 split (validation =
    link.dat.valid)."""

    def _check_target_only(self, dataset: str, folder: str, target: str) -> None:
        from src.configs.config import TrainConfig

        if not _real(lp_root(DATA_ROOT) / folder):
            self.skipTest(f"data/{folder} or its link.dat.valid is absent")
        config = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": dataset,
                                                                 "root": str(DATA_ROOT)}})
        _, carve = split_of(DATA_ROOT, dataset, config)
        assert_target_only(self, loaded_split(config), carve, link_edges_of(DATA_ROOT, dataset),
                           target)

    def test_pubmed_file_is_target_only(self) -> None:
        self._check_target_only("pubmed", "PubMed", "rel_2")

    def test_lastfm_file_is_target_only(self) -> None:
        self._check_target_only("lastfm", "LastFM", "rel_0")

    def test_derived_tasks_evaluate_81_9_10(self) -> None:
        from src.data.link_prediction_dataset import _load_local_hgb_link_data

        targets = {"acm_pa": ("paper", "rel_2", "author"), "dblp_ap": ("author", "rel_0", "paper"),
                   "imdb_ma": ("movie", "rel_2", "actor")}
        for name, target in targets.items():
            task_dir = lp_derived_root(DATA_ROOT) / name
            if not _real(task_dir):
                self.skipTest(f"data/lp_derived/{name} not built")
            meta = json.loads((task_dir / "split_meta.json").read_text(encoding="utf-8"))
            split = _load_local_hgb_link_data(derived_config(DATA_ROOT, name, target))
            counts = (split.train_edge_index_by_relation[target].size(1),
                      split.valid_edge_index_by_relation[target].size(1),
                      split.test_edge_index_by_relation[target].size(1))
            self.assertEqual(counts, (meta["train"], meta["valid"], meta["test"]), name)
            total = sum(counts)
            for part, share in zip(counts, (81.0, 9.0, 10.0)):
                self.assertLessEqual(abs(100.0 * part / total - share), 0.5, name)


if __name__ == "__main__":
    unittest.main()
