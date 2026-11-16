"""Card 1d H3b: the baseline blockers found by the dry run (card 2 D2), each reproduced off real data.

1. HAN NC imdb (`hgb_han.py` `fit_task`: "too many values to unpack (expected 11)"): HAN's IMDB loader returns HGB's
   `dl` as a 12th value (HAN/utils.py:333) and HGB runs IMDB with `main_multi.py` / `model_hetero_multi.HAN`
   (HAN/README.md). On a tiny release in HGB IMDB's raw format, `fit_task` runs one epoch with that model.
2. HAN LP and H2SGNN LP imdb_ma (`_target_block.py`: "target types actor, movie not contiguous"): IMDB lays out movie,
   director, actor, keyword, so the movie-actor block is two ranges. On a tiny imdb_ma task built by the project's
   pipeline, the block is exactly the movie and actor ids and both adapters run one epoch.
3. ChebNetII NC on the H2GB datasets, SeHGNN NC ieee-cis-g (`bundles.py` `_base_config`: "data.batch_size (B) is
   required in mini-batch training"): `fit_cell` builds the bundle configuration of a mini-batch cell with the run's
   B; the configuration is validated as `build_bundle` validates it, nothing is loaded.
No real data are read (`unreachable_real_data`).
"""

from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.lp import build_derived_lp as bd
from scripts.lp.apply_feature_rule import apply_rule
from src.baselines.protocol import LINK_PREDICTION, NODE_CLASSIFICATION, BaselineParams
from src.data.data_layout import lp_derived_root
from tests.partMB_fixtures import restore_real_data, unreachable_real_data

#: IMDB's HGB layout: 0 movie, 1 director, 2 actor, 3 keyword; relations 0/1 movie-director, 2/3 movie-actor,
#: 4/5 movie-keyword (stored both ways, HAN/utils.py:292).
COUNTS = {"movie": 24, "director": 6, "actor": 12, "keyword": 5}
MOVIE_DIM, NUM_CLASSES = 14, 5


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def write_imdb_like(folder: Path, seed: int = 0) -> None:
    """node.dat (movie: 14 attributes; others: 3), link.dat (every movie has a director, two actors and a keyword),
    label.dat / label.dat.test (multi-label over 5 classes, each file covering all 5)."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    first: dict[str, int] = {}
    nodes, next_id = [], 0
    for type_id, (name, count) in enumerate(COUNTS.items()):
        first[name] = next_id
        dim = MOVIE_DIM if name == "movie" else 3
        for i in range(count):
            attrs = ",".join(f"{rng.random():.3f}" for _ in range(dim))
            nodes.append(f"{next_id}\t{name}{i}\t{type_id}\t{attrs}\n")
            next_id += 1
    (folder / "node.dat").write_bytes("".join(nodes).encode())
    links = []
    for m in range(COUNTS["movie"]):
        movie = first["movie"] + m
        director = first["director"] + m % COUNTS["director"]
        actors = {first["actor"] + m % COUNTS["actor"], first["actor"] + (m * 5 + 1) % COUNTS["actor"]}
        keyword = first["keyword"] + m % COUNTS["keyword"]
        links += [f"{movie}\t{director}\t0\t1.0\n", f"{director}\t{movie}\t1\t1.0\n"]
        for actor in sorted(actors):
            links += [f"{movie}\t{actor}\t2\t1.0\n", f"{actor}\t{movie}\t3\t1.0\n"]
        links += [f"{movie}\t{keyword}\t4\t1.0\n", f"{keyword}\t{movie}\t5\t1.0\n"]
    (folder / "link.dat").write_bytes("".join(links).encode())
    train, test = [], []
    for m in range(COUNTS["movie"]):
        labels = movie_labels(m)
        line = f"{m}\tmovie{m}\t0\t{','.join(map(str, labels))}\n"
        (test if m % 4 == 3 else train).append(line)
    (folder / "label.dat").write_bytes("".join(train).encode())
    (folder / "label.dat.test").write_bytes("".join(test).encode())
    (folder / "info.dat").write_text(json.dumps({"synthetic": True}), encoding="utf-8")


def movie_labels(m: int) -> list[int]:
    return sorted({m % NUM_CLASSES, (m * 3 + 1) % NUM_CLASSES})


class HANImdbNodeClassificationTests(unittest.TestCase):
    def test_fit_task_runs_hgb_imdb_path(self) -> None:
        from src.baselines.adapters import hgb_han
        from src.baselines.bundles import NodeBundle
        from src.data.data_layout import hgb_nc_raw_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_imdb_like(hgb_nc_raw_dir(root, "imdb"))
            movies = COUNTS["movie"]
            target_y = torch.zeros(movies, NUM_CLASSES)
            masks = {name: torch.zeros(movies, dtype=torch.bool) for name in ("train", "val", "test")}
            for m in range(movies):
                target_y[m, movie_labels(m)] = 1.0
                masks["test" if m % 4 == 3 else "val" if m % 4 == 2 else "train"][m] = True
            data = SimpleNamespace(target_y=target_y, target_train_mask=masks["train"],
                                   target_val_mask=masks["val"], target_test_mask=masks["test"])
            bundle = NodeBundle(dataset="imdb", data=data, num_classes=NUM_CLASSES, is_multilabel=True,
                                metadata={"data_root": str(root)})
            record = hgb_han.HANAdapter().fit_task(  # the dry run's error: unpacking HAN's 12 IMDB values
                bundle, BaselineParams({"epochs": 1, "early_stopping": 1, "hidden_dim": 8, "num_heads": 2}),
                task=NODE_CLASSIFICATION, seed=0, device=torch.device("cpu"))
            loaded = hgb_han._load_vendor_graph("imdb", root)
            self.assertEqual(len(loaded), 11)
            self.assertTrue(torch.equal(loaded[2], target_y))
            with hgb_han.pushd_sys_path(*hgb_han.VENDORS):
                self.assertEqual(hgb_han._vendor_model_classes("imdb")[0].__module__, "model_hetero_multi")
                self.assertEqual(hgb_han._vendor_model_classes("acm")[0].__module__, "model_hetero")
        self.assertEqual(record["num_meta_paths"], 3.0)
        self.assertTrue(any(k.startswith("test_") and math.isfinite(float(v)) for k, v in record.items()
                            if isinstance(v, (int, float))))


class ImdbMovieActorBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.baselines.bundles import build_bundle

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        write_imdb_like(root / "release_imdb")
        task_dir = lp_derived_root(root) / "imdb_ma"
        bd.build_task(root / "release_imdb", task_dir, 2, seed=42, val_frac=0.09, test_frac=0.10)
        apply_rule("imdb_ma", task_dir)
        cls.bundle = build_bundle(LINK_PREDICTION, "imdb_ma", overrides={"device": "cpu", "data": {"root": str(root)}})

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_block_is_movie_and_actor_and_both_adapters_run(self) -> None:
        from src.baselines.registry import get_adapter

        for name in ("han", "h2sgnn"):  # the dry run's error: "target types ... not contiguous"
            with self.subTest(adapter=name):
                record = get_adapter(name).fit_task(self.bundle, BaselineParams({"epochs": 1, "early_stopping": 1}),
                                                    task=LINK_PREDICTION, seed=0, device=torch.device("cpu"))
                self.assertTrue(math.isfinite(float(record["test_random_auroc"])))

        from src.baselines.adapters._target_block import block_local_ids, scatter_to_full_graph, target_block

        slices = dict(self.bundle.data.node_slices)
        block = target_block(self.bundle)
        expected = torch.cat([torch.arange(*slices["movie"]), torch.arange(*slices["actor"])])
        self.assertTrue(torch.equal(block, expected))
        self.assertGreater(slices["actor"][0], slices["movie"][1])  # director lies between: two ranges
        num_nodes = int(self.bundle.data.x.size(0))
        local = block_local_ids(block, num_nodes)
        self.assertTrue(torch.equal(local[block], torch.arange(block.numel())))
        self.assertTrue(bool((local[torch.arange(*slices["director"])] == -1).all()))
        rows = torch.arange(block.numel(), dtype=torch.float32).unsqueeze(1) + 1.0
        full = scatter_to_full_graph(rows, block, num_nodes)
        self.assertTrue(torch.equal(full[block], rows))
        self.assertEqual(int((full != 0).sum()), block.numel())


class MinibatchBundleBatchSizeTests(unittest.TestCase):
    CELLS = [("chebnetii", d) for d in ("ogbn-mag", "mag-year", "oag-cs", "oag-eng", "oag-chem")] + [
        ("sehgnn", "ieee-cis-g")]

    def test_fit_cell_builds_a_valid_minibatch_configuration(self) -> None:
        from src.baselines import bundles
        from src.baselines.campaign import fit_cell

        class Built(Exception):
            pass

        seen: list[int] = []

        def validate_only(task, dataset, *, run_index=0, seed=42, overrides=None):
            config = bundles._base_config(dataset, run_index=run_index, seed=seed, overrides=overrides)
            seen.append(int(config.data.batch_size))
            raise Built

        for model, dataset in self.CELLS:
            with self.subTest(cell=(model, dataset)), mock.patch.object(bundles, "build_bundle", validate_only):
                raw_base = {"model": model, "task": NODE_CLASSIFICATION, "dataset": dataset, "seed": 0}
                with self.assertRaises(Built):
                    fit_cell(raw_base, {"batch_size": 1024}, source="tuned", device=torch.device("cpu"))
                self.assertEqual(seen[-1], 1024)
                with self.assertRaises(Built):
                    fit_cell(raw_base, {}, source="tuned", device=torch.device("cpu"))
                self.assertEqual(seen[-1], 256)  # minibatch_plan's default


if __name__ == "__main__":
    unittest.main()
