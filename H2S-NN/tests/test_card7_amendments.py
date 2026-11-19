"""Card 7 amendments after the Phase A review (author 2026-11-19), the parts without a card-5 hook:

1. levels 0.20-0.60 in steps of 0.1 (seeds of Phase A's levels unchanged), C and the ceiling 1 - 1/C per graph, the
   perturbation figure draws a dashed line at 1 - 1/C per dataset (crafted inputs, no training);
3. the edge-list fingerprint: the generator's (from the written files) equals the one of the graph H2S-NN's loader
   builds; the run check (`run_checked.py`) on a toy perturbed graph through H2S-NN (`run_h2snn_runs.py --dry_run`)
   and one baseline (GCN at its HGB command, `run_hgb_command.py`) from the graph's folder (the `./data` route); a
   wrong graph (fingerprint) and a wrong working directory (data root) fail the run;
4, 7. the baseline configurations: GCN / HAN mapped from HGB's files (values read back from the vendor files),
   every mapped key is an adapter key, SeHGNN / Hetero2Net published, CSNN has no published ACM/DBLP configuration.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

from tests.card7_fixtures import write_toy_hgb
from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "perturbation"))
sys.path.insert(0, str(ROOT))

import baseline_configs as bc  # noqa: E402
import fingerprint as fp  # noqa: E402
import make_graphs as mg  # noqa: E402
import run_checked  # noqa: E402

MAX_ADD = 2.0
TASK = "node_classification"
SMALL = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16, "classifier_hidden_dim": 16}


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def h2snn_config(dataset: str, root: Path) -> dict:
    return {"device": "cpu", "seed": 1, "model": dict(SMALL),
            "data": {"dataset_name": dataset, "root": str(root), "wisp_max_length": 3, "wisp_walks_per_anchor": 4}}


def load(dataset: str, root: Path):
    from src.configs.config import TrainConfig
    from src.data.dataset import build_dataset

    return build_dataset(TrainConfig.from_dict(h2snn_config(dataset, root)))


class LevelTests(unittest.TestCase):
    def test_levels_and_seeds(self) -> None:
        import manifest

        self.assertEqual(mg.LEVELS, (0.20, 0.30, 0.40, 0.50, 0.60))
        self.assertEqual(manifest.LEVELS, mg.LEVELS)
        self.assertEqual(manifest.DATASETS, ("acm", "dblp"))
        # Phase A's graphs keep their seeds (job 17814: ACM targeted 0.40 = 10, DBLP random 0.60 = 1021)
        self.assertEqual((mg.graph_seed("acm", 0.40, "targeted"), mg.graph_seed("dblp", 0.60, "random"),
                          mg.graph_seed("acm", 0.20, "random")), (10, 1021, 1))
        self.assertEqual((mg.graph_seed("acm", 0.30, "targeted"), mg.graph_seed("dblp", 0.50, "random")),
                         (40, 1051))
        seeds = {mg.graph_seed(d, lv, k) for d in ("acm", "dblp") for lv in mg.LEVELS for k in mg.KINDS}
        self.assertEqual(len(seeds), 2 * 5 * 2)


class _Toy(unittest.TestCase):
    dataset = "dblp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "data"
        write_toy_hgb(cls.root, cls.dataset)
        cls.record = mg.generate(cls.root, cls.dataset, MAX_ADD, levels=(0.60,), log=lambda *_: None)[0]
        cls.graph_json = Path(cls.record["data_root"]).parent / "graph.json"
        cls.original = fp.from_raw(mg.raw_dir_of(cls.root, cls.dataset))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()


class FingerprintTests(_Toy):
    def test_generator_fingerprint_equals_the_loaded_graph(self) -> None:
        graph_root = Path(self.record["data_root"])
        self.assertEqual(json.loads(self.graph_json.read_text(encoding="utf-8"))["edge_fingerprint"],
                         self.record["edge_fingerprint"])
        loaded = fp.from_prepared(load(self.dataset, graph_root))
        self.assertEqual((loaded["sha256"], loaded["pairs"]),
                         (self.record["edge_fingerprint"], self.record["edge_pairs"]))
        self.assertEqual(fp.from_prepared(load(self.dataset, self.root)), self.original)
        self.assertNotEqual(self.original["sha256"], self.record["edge_fingerprint"])
        added = sum(c["added"] for c in self.record["facts"].values())
        self.assertEqual(self.record["edge_pairs"] - self.original["pairs"], added)
        self.assertEqual((self.record["num_classes"], self.record["ceiling"]), (4, 0.75))

    def test_acm_shape_with_self_mirrored_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            write_toy_hgb(root, "acm")
            self.assertEqual(fp.from_prepared(load("acm", root)), fp.from_raw(mg.raw_dir_of(root, "acm")))
            self.assertAlmostEqual(mg.class_ceiling(root, "acm")["ceiling"], 2.0 / 3.0)

    def test_backfill_adds_the_fields_and_refuses_a_changed_link_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            write_toy_hgb(root, "dblp")
            record = mg.generate(root, "dblp", MAX_ADD, levels=(0.20,), log=lambda *_: None)[0]
            path = Path(record["data_root"]).parent / "graph.json"
            stripped = {k: v for k, v in record.items()
                        if k not in ("edge_fingerprint", "edge_pairs", "num_classes", "ceiling")}
            path.write_text(json.dumps(stripped), encoding="utf-8")
            done = mg.backfill(root, "dblp", log=lambda *_: None)
            self.assertEqual(len(done), 2)
            refreshed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(refreshed["edge_fingerprint"], record["edge_fingerprint"])
            link = mg.raw_dir_of(Path(record["data_root"]), "dblp") / "link.dat"
            link.unlink()                                          # a hard link to nothing shared: safe to rewrite
            link.write_text("", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                mg.backfill(root, "dblp", log=lambda *_: None)


class CheckTests(_Toy):
    """Amendment 3: the run check through H2S-NN and GCN on the toy perturbed graph."""

    def run_checked(self, expected: str, log: Path, launcher: str, *args: str) -> None:
        run_checked.main(["--graph_json", str(self.graph_json), "--expected_fingerprint", expected, "--log",
                          str(log), "--", str(ROOT / launcher), *args])

    def lines(self, log: Path) -> list[dict]:
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    def test_h2snn_run_logs_and_matches(self) -> None:
        import yaml

        out = Path(self.tmp.name) / "h2snn"
        config = out / "config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(yaml.safe_dump(h2snn_config(self.dataset, Path(self.record["data_root"]))),
                          encoding="utf-8")
        log = out / "checks.jsonl"
        self.run_checked(self.record["edge_fingerprint"], log, "scripts/run_h2snn_runs.py", "--config", str(config),
                         "--base_seed", "1", "--num_runs", "1", "--exp_name", "toy", "--output_dir", str(out),
                         "--dry_run", "--device", "cpu")
        checks = self.lines(log)
        self.assertGreaterEqual(len(checks), 1)
        self.assertTrue(all(c["match"] and c["fingerprint"] == self.record["edge_fingerprint"] for c in checks))
        self.assertTrue((out / "toy" / "runs_summary.json").exists())

    def test_gcn_from_the_graph_folder_logs_and_matches(self) -> None:
        out = Path(self.tmp.name) / "gcn"
        log = out / "checks.jsonl"
        cwd = os.getcwd()
        try:
            os.chdir(self.graph_json.parent)                      # the baselines' `./data` route
            self.run_checked(self.record["edge_fingerprint"], log, "scripts/perturbation/run_hgb_command.py",
                             "--model", "gcn", "--dataset", "dblp", "--num_runs", "1", "--base_seed", "1",
                             "--output_dir", str(out), "--device", "cpu")
        finally:
            os.chdir(cwd)
        checks = self.lines(log)
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0]["match"])
        self.assertEqual(Path(checks[0]["data_root"]), Path(self.record["data_root"]).resolve())
        record = json.loads((out / "gcn_node_classification_dblp" / "hgb_command_run0.json").read_text("utf-8"))
        self.assertEqual((record["configuration"], record["params_source"]), ("hgb_command", "published"))
        self.assertEqual(record["sweep_params"], bc.hgb_command_params("gcn", "dblp"))
        self.assertTrue((out / "gcn_node_classification_dblp" / "hgb_command_summary.json").exists())

    def test_wrong_graph_fails_before_training(self) -> None:
        from src.baselines.bundles import NODE_CLASSIFICATION, build_bundle

        log = Path(self.tmp.name) / "wrong" / "checks.jsonl"
        cwd = os.getcwd()
        try:
            os.chdir(self.graph_json.parent)
            with run_checked.checked_loading(self.graph_json, self.original["sha256"], log):
                with self.assertRaises(run_checked.FingerprintMismatch):
                    build_bundle(NODE_CLASSIFICATION, "dblp", run_index=0, seed=1)
            os.chdir(self.root.parent)                            # `./data` is the original toy, not the graph
            with run_checked.checked_loading(self.graph_json, self.record["edge_fingerprint"], log):
                with self.assertRaises(run_checked.FingerprintMismatch):
                    build_bundle(NODE_CLASSIFICATION, "dblp", run_index=0, seed=1)
        finally:
            os.chdir(cwd)
        first, second = self.lines(log)
        self.assertEqual((first["match"], first["failed"], first["fingerprint"]),
                         (False, "fingerprint", self.record["edge_fingerprint"]))
        self.assertEqual((second["match"], second["failed"]), (False, "data_root"))
        import src.builders
        import src.data.dataset

        self.assertEqual(src.builders.build_dataset.__name__, "build_dataset")   # loader restored
        self.assertIs(src.builders.build_dataset, src.data.dataset.build_dataset)

    def test_a_launcher_that_never_loads_fails(self) -> None:
        script = Path(self.tmp.name) / "noop.py"
        script.write_text("print('no load')\n", encoding="utf-8")
        with self.assertRaises(run_checked.FingerprintMismatch):
            run_checked.main(["--graph_json", str(self.graph_json), "--expected_fingerprint", "x", "--log",
                              str(Path(self.tmp.name) / "noop.jsonl"), "--", str(script)])


class BaselineConfigTests(unittest.TestCase):
    def read(self, rel: str) -> str:
        return (ROOT / rel).read_text(encoding="utf-8")

    def test_gcn_values_are_hgb_s(self) -> None:
        readme, run = self.read(bc.HGB_GNN_README), self.read(bc.HGB_GNN_RUN)

        def default(flag: str) -> float:
            return float(re.search(rf"add_argument\('{flag}', type=\w+, default=([0-9.e-]+)", run).group(1))

        for dataset in ("acm", "dblp"):
            entry = bc.HGB_COMMANDS[("gcn", dataset)]
            self.assertIn(entry["command"], readme)
            flags = dict(re.findall(r"--([a-z-]+)[ =]([^ ]+)", entry["command"]))
            params = entry["params"]
            self.assertEqual(params["lr"], float(flags["lr"]))
            self.assertEqual(params["weight_decay"], float(flags["weight-decay"]))
            self.assertEqual(params["feats_type"], int(flags.get("feats-type", default("--feats-type"))))
            for flag, key in (("--hidden-dim", "hidden_dim"), ("--num-layers", "num_layers"),
                              ("--dropout", "dropout"), ("--epoch", "epochs"), ("--patience", "early_stopping")):
                self.assertEqual(params[key], default(flag), flag)
        self.assertEqual(bc.HGB_COMMANDS[("gcn", "acm")]["params"]["feats_type"], 0)

    def test_han_values_are_hgb_s(self) -> None:
        main, utils = self.read(bc.HGB_HAN_MAIN), self.read(bc.HGB_HAN_UTILS)
        block = utils[utils.index("default_configure = {"):utils.index("}", utils.index("default_configure = {"))]
        conf = {k: v for k, v in re.findall(r"'(\w+)': ([^,]+),", block)}
        for dataset in ("acm", "dblp"):
            self.assertIn(f"python main.py --dataset {dataset.upper()}", self.read(bc.HGB_HAN_README))
            params = bc.HGB_COMMANDS[("han", dataset)]["params"]
            self.assertEqual((params["lr"], params["weight_decay"], params["hidden_dim"], params["dropout"]),
                             (float(conf["lr"]), float(conf["weight_decay"]), int(conf["hidden_units"]),
                              float(conf["dropout"])))
            self.assertEqual(f"[{params['num_heads']}]", conf["num_heads"])
            for flag, key in (("--num_epochs", "epochs"), ("--patience", "early_stopping")):
                value = re.search(rf"add_argument\('{flag}', type=int, default=(\d+)\)", main).group(1)
                self.assertEqual(params[key], int(value), flag)

    def test_every_mapped_key_is_an_adapter_key(self) -> None:
        from src.baselines.registry import get_adapter

        for (model, dataset), entry in bc.HGB_COMMANDS.items():
            known = set(get_adapter(model).search_space(TASK, dataset)) | {"epochs", "early_stopping"}
            self.assertEqual(set(entry["params"]) - known, set(), model)
            mapped = {key for m, _, _, key, _ in bc.MAPPING if m == model}
            self.assertEqual(set(entry["params"]) - mapped, set(), model)

    def test_sources(self) -> None:
        from src.baselines.registry import get_adapter

        self.assertEqual(set(bc.CONFIG_SOURCE), {"gcn", "han", "sehgnn", "hetero2net", "hetsheaf", "csnn"})
        for dataset in ("acm", "dblp"):
            for model in ("sehgnn", "hetero2net"):
                self.assertIsNotNone(get_adapter(model).published_params(TASK, dataset), model)
            for model in ("gcn", "han", "hetsheaf", "csnn"):
                self.assertIsNone(get_adapter(model).published_params(TASK, dataset), model)
        scripts = {p.name.lower() for p in (ROOT / "vendor" / "csnn" / "exp" / "scripts").iterdir()}
        self.assertFalse(any(d in name for name in scripts for d in ("acm", "dblp")))
        self.assertEqual((bc.CONFIG_SOURCE["csnn"], bc.CONFIG_SOURCE["hetsheaf"]), ("tuned", "tuned"))


class FigureTests(unittest.TestCase):
    def test_ceiling_lines_and_collection(self) -> None:
        import csv

        import figure
        import manifest

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = []
            for dataset in ("acm", "dblp"):
                for kind in ("targeted", "random"):
                    for i, level in enumerate((0.2, 0.4, 0.6)):
                        for method in ("h2snn_full", "gcn"):
                            summary = tmp / f"{dataset}_{kind}_{level}_{method}.json"
                            if not (dataset == "dblp" and method == "gcn" and level == 0.6):
                                summary.write_text(json.dumps({"summary": {"metrics": {"test_macro_f1": {
                                    "mean": 0.9 - 0.1 * i, "std": 0.01}}}}), encoding="utf-8")
                            rows.append({"graph_kind": kind, "dataset": dataset, "level": f"{level:.2f}",
                                         "h": f"{level + 0.003:.4f}", "h2_index": f"{1.5 * level:.4f}",
                                         "method": method, "configuration": "x", "summary": str(summary)})
            for dataset in ("acm", "dblp"):                       # amendment 8: the baselines' original-graph rows
                summary = tmp / f"{dataset}_original_gcn.json"
                summary.write_text(json.dumps({"summary": {"metrics": {"test_macro_f1": {
                    "mean": 0.93, "std": 0.02}}}}), encoding="utf-8")
                rows.append({"graph_kind": "original", "dataset": dataset, "level": "", "h": "0.2620",
                             "h2_index": "0.3900", "method": "gcn", "configuration": "x", "summary": str(summary)})
                rows.append({"graph_kind": "original", "dataset": dataset, "level": "", "h": "0.2620",
                             "h2_index": "0.3900", "method": "csnn", "configuration": "needs tuned config",
                             "summary": ""})
            path = tmp / "manifest.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest.FIELDS), delimiter="\t",
                                        extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerows(rows)
            for dataset, c in (("acm", 3), ("dblp", 4)):
                folder = tmp / "NC" / f"{dataset.upper()}_pert" / "targeted_h0.20"
                folder.mkdir(parents=True)
                (folder / "graph.json").write_text(json.dumps({"dataset": dataset, "ceiling": 1 - 1 / c}),
                                                   encoding="utf-8")
            result = figure.collect(path)
            self.assertEqual(len(result["points"]), 2 * 2 * 3 * 2 - 2)
            self.assertEqual({(m["dataset"], m["method"], m["level"]) for m in result["missing"]},
                             {("dblp", "gcn", 0.6), ("acm", "csnn", None), ("dblp", "csnn", None)})
            self.assertEqual(result["points"][0]["h"], 0.203)
            self.assertEqual({(r["dataset"], r["method"], r["h"], r["mean"]) for r in result["reference"]},
                             {("acm", "gcn", 0.262, 0.93), ("dblp", "gcn", 0.262, 0.93)})
            ceiling = figure.ceilings(tmp / "NC")
            self.assertAlmostEqual(ceiling["acm"], 2 / 3)
            # amendment 8: H2S-NN reference = ablation configuration (a), seeds 1-5 only (seeds 6-10 and (b) ignored)
            table = tmp / "ablation.csv"
            lines = ["dataset,row,configuration,seed,test_macro_f1,source"]
            for seed in range(1, 11):
                lines += [f"acm,full,a,{seed},{0.90 + 0.01 * seed},x", f"acm,full,b,{seed},0.1,x",
                          f"acm,S2,a,{seed},{0.80 + 0.01 * seed},x", f"acm,endpoints,a,{seed},0.5,x"]
            lines += [f"acm,no_path_cells,a,{seed},0.7,x" for seed in (1, 2, 3, 4)]       # seed 5 missing
            table.write_text("\n".join(lines) + "\n", encoding="utf-8")
            points, missing = figure.h2snn_reference(table, {"acm": 0.262})
            got = {p["method"]: p for p in points}
            self.assertEqual(sorted(got), ["h2snn_S2", "h2snn_full"])
            self.assertAlmostEqual(got["h2snn_full"]["mean"], 0.93)
            self.assertAlmostEqual(got["h2snn_full"]["std"], 0.0158113883, places=8)       # sample std of 1-5
            self.assertEqual(got["h2snn_full"]["h"], 0.262)
            self.assertEqual({(m["dataset"], m["method"]) for m in missing},
                             {("acm", "h2snn_no_path_cells")} | {("dblp", f"h2snn_{a}") for a in
                                                                  ("full", "no_path_cells", "S2")})
            result["reference"] += points
            drawn = figure.figure(result, ceiling, tmp / "fig.pdf")
            self.assertEqual(len(drawn), 4)                                    # 2 datasets x 2 kinds
            for line in drawn:
                self.assertEqual(line["linestyle"], "--")
                self.assertAlmostEqual(line["x"], 1 - 1 / {"acm": 3, "dblp": 4}[line["dataset"]])
            self.assertGreater((tmp / "fig.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
