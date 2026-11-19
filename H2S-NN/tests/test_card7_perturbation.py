"""Card 7 T3/T4 (`docs/H2S-NN_card7_card.md` V3): the perturbation generator on toy HGB releases
(`tests/card7_fixtures.py`, written in HGB's own file format under a temporary data root; the real data root is
unreachable).

T3: a reached level is within ±0.01 of its target, an unreached one stopped at its bound (50 % removal cap or the
addition bound); added facts are typed, new, not self-loops, from the M_2 relations only, and each has a
different-class meta-path partner (mutation 1's target); no other relation is touched; removal <= 50 % per
relation and only of different-class facts; the random control has equal counts per relation; h is the reference
code's value (`compute_heterophily.analyse` on the files equals the search value).
T4: a perturbed graph loads through H2S-NN's loader (`data.root` = the graph's data root) and through a baseline's
(the bundle's default `data` root from the graph folder, then HAN's own HGB loader) with only the data directory
changed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from tests.card7_fixtures import write_toy_hgb
from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "perturbation"))
sys.path.insert(0, str(ROOT))

import make_graphs as mg  # noqa: E402

MAX_ADD = 2.0


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class _ToyCase(unittest.TestCase):
    dataset = "dblp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "data"
        write_toy_hgb(cls.root, cls.dataset)
        cls.graph = mg.load_graph(cls.root, cls.dataset)
        cls.h0 = mg.heterophily(cls.graph, cls.graph.facts)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def link_lines(self, root: Path) -> Counter:
        name = mg.ref.HGB_DATASETS[self.dataset]
        path = root / "NC" / self.dataset / "raw" / name / "link.dat"
        return Counter(path.read_text(encoding="utf-8").splitlines())


class TargetedTests(_ToyCase):
    def test_toy_lies_between_the_levels(self) -> None:
        self.assertGreater(self.h0, 0.21)
        self.assertLess(self.h0, 0.59)

    def test_reached_within_tolerance_or_stopped_at_the_bound(self) -> None:
        for level in mg.LEVELS:
            p = mg.targeted(self.graph, level, MAX_ADD, self.h0)
            h = mg.heterophily(self.graph, p.facts(self.graph))
            self.assertAlmostEqual(h, p.search_h, places=12)
            if p.reached:
                self.assertLessEqual(abs(h - level), mg.TOLERANCE, level)
                continue
            # unreached: the bound itself misses the level too, and the kept graph is the closest one
            self.assertIsNotNone(p.bound_h, level)
            self.assertGreater(abs(p.bound_h - level), mg.TOLERANCE, level)
            self.assertLessEqual(abs(h - level), abs(p.bound_h - level), level)
            for rel in self.graph.relations:
                n = self.graph.facts[rel.name].shape[0]
                if p.direction == "raise":
                    self.assertLessEqual(p.added[rel.name].shape[0], int(round(MAX_ADD * n)), level)
                else:
                    self.assertLessEqual(p.removed[rel.name].shape[0], int(mg.REMOVAL_CAP * n), level)

    def test_raising_adds_only_typed_new_different_class_facts(self) -> None:
        p = mg.targeted(self.graph, 0.60, MAX_ADD, self.h0)
        self.assertEqual(p.direction, "raise")
        labels = self.graph.labels
        total = 0
        for rel in self.graph.relations:
            added, partners = p.added[rel.name], p.partners[rel.name]
            original = {tuple(f) for f in self.graph.facts[rel.name].tolist()}
            self.assertEqual(len({tuple(f) for f in added.tolist()}), added.shape[0], "duplicate")
            for (t, x), q in zip(added.tolist(), partners.tolist()):
                total += 1
                self.assertNotIn((t, x), original)
                self.assertTrue(0 <= t < self.graph.num_nodes[rel.target_type])
                self.assertTrue(0 <= x < self.graph.num_nodes[rel.other_type])
                self.assertNotEqual(labels[q], labels[t], "the meta-path partner has the node's own class")
                if rel.target_target:
                    self.assertNotEqual(t, x)
                    self.assertEqual(x, q)
                else:
                    self.assertIn((q, x), original)
        self.assertGreater(total, 0)
        self.assertTrue(all(v.size == 0 for v in p.removed.values()))

    def test_lowering_removes_at_most_half_and_only_different_class_facts(self) -> None:
        p = mg.targeted(self.graph, 0.20, MAX_ADD, self.h0)
        self.assertEqual(p.direction, "lower")
        self.assertTrue(all(v.size == 0 for v in p.added.values()))
        for rel in self.graph.relations:
            facts = self.graph.facts[rel.name]
            removed = {tuple(f) for f in p.removed[rel.name].tolist()}
            self.assertLessEqual(len(removed), int(mg.REMOVAL_CAP * facts.shape[0]))
            candidates = {tuple(facts[i]) for i in mg.removal_candidates(self.graph, rel,
                                                                         np.random.default_rng(1)).tolist()}
            self.assertLessEqual(removed, candidates)

    def test_random_control_has_equal_counts_per_relation(self) -> None:
        for level in (0.20, 0.60):
            t = mg.targeted(self.graph, level, MAX_ADD, self.h0)
            r = mg.random_control(self.graph, t)
            self.assertEqual(t.counts(), r.counts())
            self.assertEqual((r.kind, r.seed), ("random", mg.graph_seed(self.dataset, level, "random")))
            for rel in self.graph.relations:
                original = {tuple(f) for f in self.graph.facts[rel.name].tolist()}
                for t_, x in r.added[rel.name].tolist():
                    self.assertNotIn((t_, x), original)
                self.assertLessEqual({tuple(f) for f in r.removed[rel.name].tolist()}, original)


class WrittenGraphTests(_ToyCase):
    def test_only_m2_relations_are_touched_and_h_is_the_reference_value(self) -> None:
        # DBLP: M_2 = author-paper-author; paper-term and paper-venue (link types 1, 2, 4, 5) stay as they are
        self.assertEqual([r.name for r in self.graph.relations], ["author-to-paper"])
        original = self.link_lines(self.root)
        records = mg.generate(self.root, self.dataset, MAX_ADD, levels=(0.20, 0.60), log=lambda *_: None)
        self.assertEqual(len(records), 4)
        for record in records:
            root = Path(record["data_root"])
            lines = self.link_lines(root)
            for k in ("1", "2", "4", "5"):
                self.assertEqual(Counter(l for l in lines.elements() if l.split("\t")[2] == k),
                                 Counter(l for l in original.elements() if l.split("\t")[2] == k), k)
            self.assertAlmostEqual(record["h"], record["search_h"], places=5)
            self.assertEqual(record["h"], mg.reference_values(root, self.dataset)["h"])
            c = record["facts"]["author-to-paper"]
            self.assertEqual(sum(lines.values()), sum(original.values()) + 2 * (c["added"] - c["removed"]))
            self.assertEqual(record["link_sha256"], mg.fingerprint(root, self.dataset))
        original_h = mg.reference_values(self.root, self.dataset)["h"]
        self.assertAlmostEqual(original_h, self.h0, places=6)


class AcmToyTests(_ToyCase):
    dataset = "acm"

    def test_every_acm_relation_is_m2_and_raising_scales_every_relation(self) -> None:
        self.assertEqual(sorted(r.name for r in self.graph.relations),
                         sorted(["paper-cite-paper", "paper-to-author", "paper-to-subject", "paper-to-term"]))
        p = mg.targeted(self.graph, 0.60, MAX_ADD, self.h0)
        for rel in self.graph.relations:
            n = self.graph.facts[rel.name].shape[0]
            self.assertEqual(p.added[rel.name].shape[0], int(round(p.alpha * n)), rel.name)


class LoaderTests(_ToyCase):
    """T4: one perturbed graph through H2S-NN's loader and a baseline's, with only the data directory changed."""

    def test_perturbed_graph_loads_through_h2snn_and_han_by_data_directory(self) -> None:
        from src.baselines.adapters.hgb_han import _load_vendor_graph
        from src.baselines.bundles import NODE_CLASSIFICATION, build_bundle, bundle_data_root
        from src.configs.config import TrainConfig
        from src.data.dataset import build_dataset

        record = mg.generate(self.root, self.dataset, MAX_ADD, levels=(0.60,), log=lambda *_: None)[0]
        graph_root = Path(record["data_root"])
        lines = sum(self.link_lines(graph_root).values())
        added = record["facts"]["author-to-paper"]["added"]
        # H2S-NN: data.root = the graph's data root
        small = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                 "classifier_hidden_dim": 16}
        cfg = TrainConfig.from_dict({"device": "cpu", "seed": 1, "model": small,
                                     "data": {"dataset_name": "dblp", "root": str(graph_root),
                                              "wisp_max_length": 3, "wisp_walks_per_anchor": 4}})
        data = build_dataset(cfg)
        base = build_dataset(TrainConfig.from_dict({"device": "cpu", "seed": 1, "model": small,
                                                    "data": {"dataset_name": "dblp", "root": str(self.root),
                                                             "wisp_max_length": 3, "wisp_walks_per_anchor": 4}}))
        self.assertEqual(int(data.edge_index.size(1)) - int(base.edge_index.size(1)), added)
        # A baseline: its launcher's bundle reads `./data` (the default root) from the graph folder
        cwd = os.getcwd()
        try:
            os.chdir(graph_root.parent)
            bundle = build_bundle(NODE_CLASSIFICATION, "dblp", run_index=0, seed=0)
        finally:
            os.chdir(cwd)
        self.assertEqual(bundle_data_root(bundle), graph_root.resolve())
        loaded = _load_vendor_graph("dblp", bundle_data_root(bundle))
        self.assertEqual(int(loaded[0].num_edges()), lines)


class ManifestTests(_ToyCase):
    """The perturbation manifest (amendments 1, 3, 4, 7): 8 methods on every ACM/DBLP graph at the levels 0.2-0.6,
    S2 on the targeted graphs only, seeds 1-5, the fixed configuration source per baseline (HetSheaf / CSNN "needs
    tuned config" without a sweep), every command through the fingerprint check, IMDB and 0.80 excluded, no
    out-of-scope arm."""

    def test_manifest_rows(self) -> None:
        import json as _json

        import yaml

        import baseline_configs as bc
        import manifest
        from tests.test_card7_ablation import out_of_scope_in

        records = mg.generate(self.root, self.dataset, MAX_ADD, levels=(0.20, 0.60), log=lambda *_: None)
        # graphs on disk that get no rows: Phase A's 0.80 and an IMDB graph
        for name, dataset, level in (("DBLP_pert/targeted_h0.80", "dblp", 0.8), ("IMDB_pert/targeted_h0.40", "imdb",
                                                                                   0.4)):
            folder = self.root / "NC" / name
            folder.mkdir(parents=True)
            (folder / "graph.json").write_text(_json.dumps({**records[0], "dataset": dataset, "level": level}),
                                               encoding="utf-8")
        full = ROOT / "configs" / "experiments" / "dblp_h2snn_v3b.yaml"
        baselines = Path(self.tmp.name) / "baselines"
        trials = baselines / "hetsheaf_node_classification_dblp" / "tuning_trials_run0.json"
        trials.parent.mkdir(parents=True)
        trials.write_text("[]", encoding="utf-8")
        out = Path(self.tmp.name) / "manifest"
        rows, missing = manifest.build(self.root / "NC", {"dblp": full}, baselines, out)
        self.assertEqual(manifest.LEVELS, mg.LEVELS)
        self.assertEqual(sorted(missing["excluded_graphs"]), ["dblp/targeted_h0.80", "imdb/targeted_h0.40"])
        graphs = {(r["graph_kind"], r["level"]) for r in rows}
        self.assertEqual(graphs, {(k, l) for k in ("targeted", "random") for l in ("0.20", "0.60")})
        expected_source = {"gcn": "hgb_command", "han": "hgb_command", "sehgnn": "published",
                           "hetero2net": "published", "hetsheaf": "tuned", "csnn": bc.NEEDS_TUNED}
        by_graph = {r["fingerprint"] for r in rows}
        self.assertEqual(by_graph, {r["edge_fingerprint"] for r in records})
        for kind, level in graphs:
            here = [r for r in rows if (r["graph_kind"], r["level"]) == (kind, level)]
            h2snn = {"h2snn_full", "h2snn_no_path_cells"} | ({"h2snn_S2"} if kind == "targeted" else set())
            self.assertEqual({r["method"] for r in here if r["method"].startswith("h2snn")}, h2snn)
            self.assertEqual({r["method"]: r["configuration"] for r in here if not r["method"].startswith("h2snn")},
                             expected_source)
            record = [g for g in records if (g["kind"], f"{g['level']:.2f}") == (kind, level)][0]
            for r in here:
                self.assertEqual((r["fingerprint"], r["h"], r["h2_index"]),
                                 (record["edge_fingerprint"], f"{record['h']:.4f}", f"{record['h2_index']:.4f}"))
                if r["configuration"] == bc.NEEDS_TUNED:
                    self.assertEqual((r["command"], r["summary"]), ("", ""))
                    continue
                checker = f"run_checked.py --graph_json {Path(record['data_root']).parent / 'graph.json'} "
                self.assertIn(checker, r["command"])
                self.assertIn(f"--expected_fingerprint {record['edge_fingerprint']} ", r["command"])
                self.assertIn("fingerprint_checks.jsonl -- ", r["command"])
                if r["method"].startswith("h2snn"):
                    self.assertIn("--base_seed 1 --num_runs 5", r["command"])
                    self.assertTrue(r["summary"].endswith("runs_summary.json"))
                    cfg = r["command"].split("--config ")[1].split(" ")[0]
                    self.assertEqual(yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))["data"]["root"],
                                     r["graph_root"])
                else:
                    self.assertEqual(Path(r["cwd"]) / "data", Path(r["graph_root"]))
                    launcher = {"hgb_command": "run_hgb_command.py --model", "published": "run_baseline_published.py",
                                "tuned": "run_baseline_tuning.py"}[r["configuration"]]
                    self.assertIn(launcher, r["command"])
                    seeds = ("--final_seeds 1 2 3 4 5" if r["configuration"] == "tuned"
                             else "--num_runs 5 --base_seed 1")
                    self.assertIn(seeds, r["command"])
        self.assertEqual(missing["needs_tuned_config"], ["csnn/dblp"])
        self.assertEqual(missing["original_graphs_missing"], ["acm", "dblp"])
        manifest.write(rows, missing, out)
        self.assertEqual(out_of_scope_in((out / "manifest.tsv").read_text(encoding="utf-8")), [])
        summary = _json.loads((out / "missing.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["rows"], 4 * 8 + 2)
        self.assertEqual(summary["runnable_rows"], 4 * 7 + 2)
        # amendment 8: the original graph -> the six baselines, seeds 1-5, same source and checked launcher; no H2S-NN
        original = mg.write_original(self.root, self.dataset, log=lambda *_: None)
        self.assertEqual(original["link_sha256"], mg.fingerprint(self.root, self.dataset))
        self.assertEqual(original["edge_fingerprint"], mg.fp.from_raw(mg.raw_dir_of(self.root, self.dataset))["sha256"])
        rows2, missing2 = manifest.build(self.root / "NC", {"dblp": full}, baselines, out)
        reference = [r for r in rows2 if r["graph_kind"] == "original"]
        self.assertEqual(rows2[:len(rows)], rows)
        self.assertEqual(missing2["original_graphs_missing"], ["acm"])
        self.assertEqual({r["method"]: r["configuration"] for r in reference}, expected_source)
        folder = str((self.root / "NC" / "DBLP_pert" / "original").resolve())
        for r in reference:
            self.assertEqual((r["level"], r["seeds"], r["cwd"]), ("", "1 2 3 4 5", folder))
            self.assertEqual((r["fingerprint"], r["h"]), (original["edge_fingerprint"], f"{original['h']:.4f}"))
            if r["command"]:
                self.assertIn(f"run_checked.py --graph_json {folder}/graph.json ", r["command"])
                self.assertIn("/runs/dblp/original/baselines/", r["summary"])
        manifest.write(rows2, missing2, out)
        summary = _json.loads((out / "missing.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["rows"], summary["reference_rows"], summary["runnable_rows"]),
                         (4 * 8 + 2 + 6, 6, 4 * 7 + 2 + 5))


if __name__ == "__main__":
    unittest.main()
