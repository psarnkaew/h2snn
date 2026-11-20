"""Card 9 W9 (pre-registration v3, spec 1.29): one test per new mechanism of the v3 scripts. Fixtures and crafted
numbers only; the real data root is unreachable.

- T1 rule (`scripts/ablation/stats.py`): 4 wins with a positive mean -> better; 3 wins -> not better; 4 wins with a
  negative mean -> not better; a tie is no win; a missing seed -> `None`; C2-w is a claim; a (b) run name raises.
  Mutations m1 (`wins >= 3`) and m2 (no `mean_difference > 0`) target `RuleTests.test_T1_rule_edge_cases`.
- T2 `uniform` row: the emitted `uniform.yaml` builds; one training step on the small ACM-like fixture
  (`tests/partO_fixtures`) has a finite loss, `g` is absent (its parameter count is the scalar `b`), `b` moves, every
  face carries one weight.
- T3 manifest (`seed_runs.py`): 8 rows, `full` seed 5, every other row seeds 1-5, no `tune` row, no (b) row.
- T4 scale-free margin (`mechanism_common.py`, `face_effect.py`): x* times 3 leaves it unchanged (raw margins triple);
  the F1a-F3 rules on crafted seeds. Mutation m3 (margin not divided by D) targets
  `ScaleFreeMarginTests.test_T4_scaling_x_star_leaves_the_scale_free_margin_unchanged`.
- T5 `e_tilde_v` (`two_regime.py`): x* times 3 and e_v times 9 leave it unchanged; where wrong nodes carry larger
  e_tilde_v (but smaller raw e_v), R2 holds; R1 and R3 (S2 partner needed). Mutation m4 (`e_tilde_v` = raw e_v)
  targets `NormalisedEnergyTests.test_T5_scaling_leaves_e_tilde_unchanged_and_r2_reads_it`.
- T6 perturbation rules (`scripts/perturbation/rules.py`, `figure.py`): E1 holds on ACM but not DBLP -> false; E2;
  an absent method; per-seed values from the launchers' files (a mean-only summary raises). Mutation m5 (E1 on one
  dataset only) targets `PerturbationRuleTests.test_T6_e1_needs_both_datasets_e2_and_absent`.
- T7 q3 (`scripts/analysis/q3_schema.py`): seeds 1-5 only (seed 0 ignored), no permutation key, `favour` only at
  5 of 5; the homophily scatter.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
for sub in (("scripts", "ablation"), ("scripts", "analysis"), ("scripts", "perturbation"), ("scripts",)):
    path = str(ROOT.joinpath(*sub))
    if path not in sys.path:
        sys.path.insert(0, path)

import stats  # noqa: E402

DATASETS = ("acm", "dblp", "imdb", "freebase")


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


# ----------------------------------------------------------------------------------------------------- T1
def _rows(dataset: str, row: str, values: dict[int, float]) -> list[dict]:
    return [{"dataset": dataset, "row": row, "configuration": "a", "seed": s, stats.METRIC: v}
            for s, v in values.items()]


def _long_table(differences: dict[str, list[float | None]], row: str = "uniform") -> list[dict]:
    """`full` = 0.80 on seeds 1-5 of every dataset; `row` = full - difference (None: that seed is missing)."""
    out = []
    for dataset in DATASETS:
        out += _rows(dataset, "full", {s: 0.80 for s in range(1, 6)})
        diffs = differences[dataset]
        out += _rows(dataset, row, {s: 0.80 - d for s, d in zip(range(1, 6), diffs) if d is not None})
    return out


class RuleTests(unittest.TestCase):
    CASES = {
        "acm": [0.01, 0.01, 0.01, 0.01, -0.001],        # 4 wins, mean > 0             -> better
        "dblp": [0.01, 0.01, 0.01, -0.001, -0.001],     # 3 wins, mean > 0             -> not better
        "imdb": [0.01, 0.01, 0.01, 0.01, -0.10],        # 4 wins, mean < 0             -> not better
        "freebase": [0.01, 0.01, 0.01, 0.0, -0.001],    # 3 wins + 1 tie, mean > 0     -> not better (a tie is no win)
    }

    def test_T1_rule_edge_cases(self) -> None:
        result = {c["claim"]: c for c in stats.analyse(_long_table(self.CASES))["claims"]}
        claim = result["C2-w"]
        self.assertEqual(claim["row"], "uniform")
        per = claim["per_dataset"]
        self.assertEqual({d: per[d]["wins"] for d in DATASETS}, {"acm": 4, "dblp": 3, "imdb": 4, "freebase": 3})
        self.assertEqual(claim["better"], {"acm": True, "dblp": False, "imdb": False, "freebase": False})
        self.assertGreater(per["dblp"]["mean_difference"], 0)
        self.assertLess(per["imdb"]["mean_difference"], 0)
        self.assertGreater(per["freebase"]["mean_difference"], 0)
        self.assertFalse(claim["holds"])                                    # 1 of 4
        for got, want in zip(per["acm"]["differences"], self.CASES["acm"]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(per["freebase"]["differences"][3], 0.0)          # the tie is exact
        self.assertEqual(per["acm"]["seeds"], [1, 2, 3, 4, 5])
        self.assertEqual((per["acm"]["a"]["n"], per["acm"]["b"]["n"]), (5, 5))

    def test_T1_missing_seed_is_none_never_false(self) -> None:
        cases = dict(self.CASES)
        cases["acm"] = [0.01, 0.01, 0.01, 0.01, None]                      # seed 5 of the row missing
        claim = {c["claim"]: c for c in stats.analyse(_long_table(cases))["claims"]}["C2-w"]
        self.assertIsNone(claim["per_dataset"]["acm"]["better"])
        self.assertFalse(claim["per_dataset"]["acm"]["complete"])
        self.assertEqual(claim["per_dataset"]["acm"]["seeds"], [1, 2, 3, 4])
        self.assertIsNone(claim["holds"])
        cases["acm"] = [0.01] * 5
        cases["dblp"] = cases["imdb"] = [0.01] * 5
        claim = {c["claim"]: c for c in stats.analyse(_long_table(cases))["claims"]}["C2-w"]
        self.assertTrue(claim["holds"])                                     # 3 of 4

    def test_T1_c2w_present_and_b_name_raises(self) -> None:
        self.assertIn(("C2-w", "uniform", "3of4"), [(c.name, c.row, c.rule) for c in stats.CLAIMS])
        self.assertIn("uniform", stats.KNOWN_ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "acm__ablation_a_uniform" / "acm__ablation_a_uniform_run0"
            run.mkdir(parents=True)
            (run / "best_test_metrics_run0.json").write_text(json.dumps(
                {"exp_name": "acm__ablation_a_uniform_run0", "seed": 1, stats.METRIC: 0.5}), encoding="utf-8")
            rows = stats.collect([Path(tmp)], [])
            self.assertEqual([(r["dataset"], r["row"], r["configuration"], r["seed"]) for r in rows],
                             [("acm", "uniform", "a", 1)])
            bad = Path(tmp) / "acm__ablation_b_endpoints_seeds" / "acm__ablation_b_endpoints_seeds_run0"
            bad.mkdir(parents=True)
            (bad / "best_test_metrics_run0.json").write_text(json.dumps(
                {"exp_name": "acm__ablation_b_endpoints_seeds_run0", "seed": 1, stats.METRIC: 0.5}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration \\(a\\) only"):
                stats.collect([Path(tmp)], [])
        with self.assertRaisesRegex(ValueError, "configuration \\(a\\) only"):
            stats.analyse([{"dataset": "acm", "row": "endpoints", "configuration": "b", "seed": 1,
                            stats.METRIC: 0.5}])


# ----------------------------------------------------------------------------------------------------- T2
FULL_CONFIG = ROOT / "configs" / "experiments" / "acm_h2snn_v3b.yaml"
SMALL = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16, "classifier_hidden_dim": 16,
         "propagation": "anchored"}


class UniformRowTests(unittest.TestCase):
    def test_T2_emitted_uniform_yaml_builds(self) -> None:
        import arms
        import yaml

        from src.configs.config import TrainConfig

        with tempfile.TemporaryDirectory() as tmp:
            paths = arms.emit_configurations_a(FULL_CONFIG, "acm", Path(tmp))
            raw = yaml.safe_load(paths["uniform"].read_text(encoding="utf-8"))
            cfg = TrainConfig.from_dict(raw)
            self.assertEqual(cfg.ablation.path_weight_input, "uniform")
            self.assertEqual((cfg.model.arm, bool(cfg.ablation.fact_maps_identity)), ("full", False))
            self.assertTrue(raw["exp_name"].endswith("__ablation_a_uniform"))

    def test_T2_one_training_step_of_the_uniform_model(self) -> None:
        import torch

        import param_table
        from src.analysis.mechanism_hooks import cells_per_target
        from src.builders import build_optimizer
        from src.configs.config import TrainConfig
        from src.data.dataset import build_dataset
        from src.models.h2snn_model import build_model
        from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

        cfg = TrainConfig.from_dict({"device": "cpu", "seed": 3,
                                     "data": {"dataset_name": "acm", "wisp_max_length": 3,
                                              "wisp_walks_per_anchor": 5},
                                     "model": dict(SMALL), "ablation": {"path_weight_input": "uniform"}})
        with patched_acm_loader(acm_like_hetero(seed=9)):
            data = build_dataset(cfg)
        torch.manual_seed(0)
        model = build_model(cfg, data)
        names = dict(model.named_parameters())
        self.assertFalse([n for n in names if n.startswith(("path_encoder.", "path_gate."))], "g must be absent")
        self.assertEqual(tuple(names["path_log_weight"].shape), ())
        counts = param_table.counts(model)
        self.assertEqual((counts["g"], counts["g_form"]), (1, "scalar b (path_log_weight)"))
        b_before = names["path_log_weight"].detach().clone()
        optimizer = build_optimizer(model, cfg)
        model.train()
        logits = model(data, epoch=0).logits
        mask = data.target_train_mask
        loss = torch.nn.functional.cross_entropy(logits[mask], data.target_y[mask])
        self.assertTrue(bool(torch.isfinite(loss)))
        optimizer.zero_grad()
        loss.backward()
        self.assertIsNotNone(names["path_log_weight"].grad)
        self.assertTrue(bool(torch.isfinite(names["path_log_weight"].grad)))
        optimizer.step()
        self.assertFalse(torch.equal(names["path_log_weight"].detach(), b_before), "b must move")
        model.eval()
        model._warm_start = None
        groups = cells_per_target(model, data)
        weights = torch.cat([g["w"].reshape(-1).double().cpu() for g in groups])
        self.assertGreater(weights.numel(), 1)
        self.assertTrue(bool(torch.all(weights == weights[0])), "every face carries one weight")
        bound = float(model.log_weight_bound)
        expected = torch.exp(bound * torch.tanh(names["path_log_weight"].detach().double() / bound))
        self.assertAlmostEqual(float(weights[0]), float(expected), places=5)


# ----------------------------------------------------------------------------------------------------- T3
class ManifestTests(unittest.TestCase):
    def test_T3_eight_rows_seeds_and_no_tune_row(self) -> None:
        import seed_runs

        rows_expected = {"full", "no_path_cells", "S2", "S0", "uniform", "endpoints", "instance_features",
                         "literature"}
        with tempfile.TemporaryDirectory() as tmp:
            rows = seed_runs.build_manifest("acm", FULL_CONFIG, Path(tmp) / "main", Path(tmp), None)
            seed_runs.write_manifest(rows, Path(tmp) / "acm" / "manifest.tsv")
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["arm"] for r in rows}, rows_expected)
            self.assertEqual({(r["kind"], r["configuration"]) for r in rows}, {("runs", "a")})
            for row in rows:
                seeds = list(range(row["base_seed"], row["base_seed"] + row["num_runs"]))
                self.assertEqual(seeds, [5] if row["arm"] == "full" else [1, 2, 3, 4, 5], row["arm"])
            text = (Path(tmp) / "acm" / "manifest.tsv").read_text(encoding="utf-8")
            self.assertNotIn("\ttune\t", text)
            self.assertNotIn("ablation_b_", text)
            lookup = json.loads((Path(tmp) / "acm" / "lookup.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(lookup["full_seeds_1_4_from_main_table"]), ["1", "2", "3", "4"])


# ----------------------------------------------------------------------------------------------------- T4
class ScaleFreeMarginTests(unittest.TestCase):
    def test_T4_scaling_x_star_leaves_the_scale_free_margin_unchanged(self) -> None:
        import torch

        import face_effect as fe
        import mechanism_common as mc

        g = torch.Generator().manual_seed(0)
        x = torch.randn(30, 4, generator=g, dtype=torch.float64)
        labels = torch.arange(30) % 3
        x = x + 2.0 * torch.nn.functional.one_hot(labels, 4).double()
        centres = mc.centroids(x, labels, 3)
        centres3 = mc.centroids(3 * x, labels, 3)
        torch.testing.assert_close(mc.margins(3 * x, labels, centres3), 3 * mc.margins(x, labels, centres))
        torch.testing.assert_close(mc.scale_free_margins(3 * x, labels, centres3),
                                   mc.scale_free_margins(x, labels, centres), rtol=0, atol=1e-12)
        self.assertAlmostEqual(mc.centroid_scale(centres3), 3 * mc.centroid_scale(centres), places=12)
        group = np.zeros(30, dtype=np.int64)
        a, b = fe.f2_entry(x, labels, centres, group), fe.f2_entry(3 * x, labels, centres3, group)
        self.assertAlmostEqual(b["raw"]["all"], 3 * a["raw"]["all"], places=12)
        self.assertAlmostEqual(b["scale_free"]["all"], a["scale_free"]["all"], places=12)
        # D on a hand example: centroids (0,0), (3,0), (0,4) -> pairwise 3, 4, 5 -> D = 4; a class without nodes
        hand = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [float("nan"), float("nan")]], dtype=torch.float64)
        self.assertAlmostEqual(mc.centroid_scale(hand), 4.0, places=12)

    def test_T4_face_rules_on_crafted_seeds(self) -> None:
        import face_effect as fe

        def entry(seed: int, net_all: float, net_g3: float, net_g1: float, d_all: float, d_g3: float,
                  d_g1: float) -> dict:
            f1 = {"G0": {"nodes": 0, "net_fix_rate": None}, "G1": {"nodes": 5, "net_fix_rate": net_g1},
                  "G2": {"nodes": 5, "net_fix_rate": 0.0}, "G3": {"nodes": 5, "net_fix_rate": net_g3},
                  "none": {"nodes": 0, "net_fix_rate": None}, "all": {"nodes": 15, "net_fix_rate": net_all}}
            e = {"seed": seed, "F1": f1,
                 "F2_difference": {"G0": None, "G1": d_g1, "G2": 0.0, "G3": d_g3, "all": d_all}}
            if seed <= 3:
                e["F3"] = {"by_group": {"all": {"right_to_wrong_share": 0.1 if seed != 2 else 0.0}}}
            return e

        per_seed = [entry(s, 0.1, 0.3, 0.1, 0.2, 0.5, 0.1) for s in (1, 2, 3, 4)]
        per_seed.append(entry(5, -0.1, 0.0, 0.1, -0.2, 0.0, 0.1))
        result = fe.rules("acm", per_seed)
        self.assertEqual(result["G_low"][1], "G1")                          # ACM: G0 empty -> G1
        for name in ("F1a", "F1b", "F2a", "F2b"):
            self.assertEqual((result[name]["count"], result[name]["holds"]), (4, True), name)
        self.assertEqual(result["F3"]["count"], 2)
        self.assertIs(result["F3"]["holds"], False)                         # seed 2: no right->wrong
        self.assertNotIn("F2a", fe.rules("imdb", per_seed))                 # F2: single-label datasets only
        self.assertIsNone(fe.rules("acm", per_seed[:4])["F1a"]["holds"])    # seed 5 missing -> None


# ----------------------------------------------------------------------------------------------------- T5
class NormalisedEnergyTests(unittest.TestCase):
    def test_T5_scaling_leaves_e_tilde_unchanged_and_r2_reads_it(self) -> None:
        import two_regime as tr

        rng = np.random.default_rng(0)
        x = rng.normal(size=(12, 4))
        e = rng.uniform(0.1, 1.0, size=12)
        np.testing.assert_allclose(tr.normalised_node_energy(9 * e, 3 * x), tr.normalised_node_energy(e, x),
                                   rtol=1e-12, atol=0)
        # wrong nodes: e_v 1, ||x||^2 1 (e_tilde 1); correct nodes: e_v 4, ||x||^2 16 (e_tilde 0.25)
        wrong = np.array([True] * 4 + [False] * 8)
        x = np.where(wrong[:, None], np.array([[1.0, 0, 0, 0]]), np.array([[4.0, 0, 0, 0]]))
        e = np.where(wrong, 1.0, 4.0)
        eta = np.linspace(0.1, 0.9, 12)
        group = np.array([1, 2, 3] * 4)
        records = []
        for seed in range(1, 6):
            e_tilde = tr.normalised_node_energy(e, x)
            r2 = {**tr.r2(e_tilde, wrong, eta, group), "raw": tr.r2(e, wrong, eta, group)}
            records.append({"dataset": "acm", "arm": "full", "seed": seed, "graph": "original",
                            "eta_group_counts": {"G0": 0, "G1": 4, "G2": 4, "G3": 4}, "R2": r2,
                            "R1": {"cells": {"auroc_different": 0.4 if seed == 1 else 0.6}}})
        self.assertEqual(records[0]["R2"]["auroc_wrong"], 1.0)
        self.assertEqual(records[0]["R2"]["raw"]["auroc_wrong"], 0.0)          # raw e_v points the other way
        result = tr.rules(records)["acm"]
        self.assertEqual((result["R2"]["count"], result["R2"]["holds"]), (5, True))
        self.assertEqual((result["R1"]["count"], result["R1"]["holds"]), (4, True))
        self.assertNotIn("R3", result)

    def test_T5_r3_needs_the_s2_partner(self) -> None:
        import two_regime as tr

        def rec(arm: str, seed: int, g3: float, g1: float) -> dict:
            return {"dataset": "imdb", "arm": arm, "seed": seed, "graph": "original",
                    "eta_group_counts": {"G0": 0, "G1": 3, "G2": 3, "G3": 3},
                    "R3": {"probe_x_star": {"by_group": {"G1": {"macro_f1": g1}, "G3": {"macro_f1": g3}}}}}

        records = [rec("full", s, 0.6, 0.7) for s in range(1, 6)] + [rec("S2", s, 0.4, 0.7) for s in range(1, 6)]
        self.assertIs(tr.rules(records)["imdb"]["R3"]["holds"], True)          # gain 0.2 in G3 > 0.0 in G1
        result = tr.rules(records[:-1])["imdb"]["R3"]                           # S2 seed 5 missing
        self.assertIsNone(result["per_seed"][5])
        self.assertIsNone(result["holds"])


# ----------------------------------------------------------------------------------------------------- T6
class PerturbationRuleTests(unittest.TestCase):
    H = 0.6

    def _values(self, drops: dict[tuple[str, str], float], gaps: dict[tuple[str, str], float]) -> dict:
        """Per-seed values: method m on dataset d has F1 random(h_max) = 0.80 and targeted = 0.80 - drop (+ a seed
        jitter that keeps every per-seed comparison strict); `h2snn_no_path_cells` = full - gap on the original graph
        and on targeted h_max."""
        import rules

        values = {}
        for (dataset, method), d in drops.items():
            values[(dataset, "random", self.H, method)] = {s: 0.80 + 0.001 * s for s in rules.SEEDS}
            values[(dataset, "targeted", self.H, method)] = {s: 0.80 + 0.001 * s - d for s in rules.SEEDS}
        for (dataset, kind), g in gaps.items():
            level = None if kind == "original" else self.H
            full = values.setdefault((dataset, kind, level, rules.FULL), {s: 0.80 for s in rules.SEEDS})
            values[(dataset, kind, level, rules.NO_CELLS)] = {s: full[s] - g for s in rules.SEEDS}
        return values

    def test_T6_e1_needs_both_datasets_e2_and_absent(self) -> None:
        import rules

        drops = {}
        for dataset in rules.DATASETS:
            drops[(dataset, rules.FULL)] = 0.01
            drops[(dataset, "han")] = drops[(dataset, "sehgnn")] = drops[(dataset, "hetero2net")] = 0.05
            drops[(dataset, "gcn")] = 0.05 if dataset == "acm" else 0.0            # E1 holds on ACM only
        gaps = {(d, k): (0.01 if k == "original" else 0.03) for d in rules.DATASETS for k in ("original", "targeted")}
        values = self._values(drops, gaps)
        runnable = {(d, m): True for d in rules.DATASETS for m in ("gcn", "han", "sehgnn", "hetero2net")}
        result = rules.rules(values, runnable, {d: self.H for d in rules.DATASETS})
        gcn = result["E1"]["gcn"]
        self.assertIs(gcn["per_dataset"]["acm"]["holds"], True)
        self.assertIs(gcn["per_dataset"]["dblp"]["holds"], False)
        self.assertIs(gcn["holds"], False)                                   # E1 needs ACM and DBLP
        self.assertIs(result["E1"]["han"]["holds"], True)
        for s, v in result["drop"]["acm"][rules.FULL].items():
            self.assertAlmostEqual(v, 0.01, places=12, msg=s)
        self.assertIs(result["E2"]["holds"], True)
        self.assertEqual(result["E2"]["per_dataset"]["acm"]["count"], 5)
        self.assertEqual(result["absent"], {d: ["csnn", "hetsheaf"] for d in rules.DATASETS})
        self.assertEqual(result["E1_reported"]["hetsheaf"]["per_dataset"]["acm"], {"status": "absent"})
        self.assertNotIn("holds", result["E1_reported"]["hetero2net"]["per_dataset"]["acm"])   # no verdict
        rules.markdown(result)
        gaps[("dblp", "targeted")] = 0.0                                    # gap shrinks on DBLP -> E2 false
        result = rules.rules(self._values(drops, gaps), runnable, {d: self.H for d in rules.DATASETS})
        self.assertIs(result["E2"]["holds"], False)
        del drops[("acm", "gcn")]                                            # a missing row -> undecided, no raise
        result = rules.rules(self._values(drops, gaps), runnable, {d: self.H for d in rules.DATASETS})
        self.assertIsNone(result["E1"]["gcn"]["per_dataset"]["acm"]["holds"])

    def test_T6_per_seed_values_from_the_launchers_files(self) -> None:
        import rules

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            published = tmp / "gcn_node_classification_acm" / "hgb_command_summary.json"
            published.parent.mkdir(parents=True)
            published.write_text(json.dumps({"summary": {"metrics": {}}, "units": [
                {"unit": f"run{i}", rules.METRIC: 0.5 + 0.01 * i} for i in range(5)]}), encoding="utf-8")
            tuned = tmp / "hetsheaf_node_classification_acm" / "final_seeds_summary.json"
            tuned.parent.mkdir(parents=True)
            tuned.write_text(json.dumps({"summary": {}, "units": [
                {"final_seed": s, rules.METRIC: 0.1 * s} for s in (3, 1, 2, 5, 4)]}), encoding="utf-8")
            exp = "acm__pert_targeted_h0.60_full"
            for i in range(5):
                run = tmp / "h2snn" / f"{exp}_run{i}"
                run.mkdir(parents=True)
                (run / f"best_test_metrics_run{i}.json").write_text(json.dumps(
                    {"seed": i + 1, rules.METRIC: 0.7 + 0.01 * i}), encoding="utf-8")
            (tmp / "h2snn" / exp).mkdir()
            (tmp / "h2snn" / exp / "runs_summary.json").write_text("{}", encoding="utf-8")
            seeds = "1 2 3 4 5"
            got = rules.per_seed({"method": "gcn", "summary": str(published), "seeds": seeds})
            self.assertEqual(sorted(got), [1, 2, 3, 4, 5])
            self.assertAlmostEqual(got[3], 0.52, places=12)                    # run2 = seed 3
            got = rules.per_seed({"method": "hetsheaf", "summary": str(tuned), "seeds": seeds})
            self.assertAlmostEqual(got[5], 0.5, places=12)
            got = rules.per_seed({"method": "h2snn_full", "summary": str(tmp / "h2snn" / exp / "runs_summary.json"),
                                  "seeds": seeds})
            self.assertAlmostEqual(got[1], 0.7, places=12)
            self.assertEqual(sorted(got), [1, 2, 3, 4, 5])
            self.assertIsNone(rules.per_seed({"method": "han", "summary": str(tmp / "none.json"), "seeds": seeds}))
            mean_only = tmp / "mean_only.json"
            mean_only.write_text(json.dumps({"metrics": {rules.METRIC: {"mean": 0.5, "std": 0.1}}}),
                                 encoding="utf-8")
            with self.assertRaises(rules.NoPerSeedValues):
                rules.per_seed({"method": "han", "summary": str(mean_only), "seeds": seeds})

    def test_T6_figure_names_absent_methods(self) -> None:
        import figure

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.tsv"
            rows = [("acm", m, "cmd" if m not in ("hetsheaf", "csnn") else "") for m in figure.BASELINES]
            manifest.write_text("dataset\tmethod\tcommand\n" + "".join(f"{d}\t{m}\t{c}\n" for d, m, c in rows),
                                encoding="utf-8")
            absent = figure.absent_methods(manifest)
            self.assertEqual(absent, {"acm": ["hetsheaf", "csnn"]})
            result = {"points": [{"dataset": "acm", "kind": "targeted", "method": "gcn", "h": 0.6, "mean": 0.5,
                                  "std": 0.01}], "reference": [], "absent": absent}
            figure.figure(result, {"acm": 0.667}, Path(tmp) / "f.pdf")
            self.assertTrue((Path(tmp) / "f.pdf").stat().st_size > 0)


# ----------------------------------------------------------------------------------------------------- T7
def _keys(obj) -> set[str]:
    if isinstance(obj, dict):
        return {str(k) for k in obj} | {k for v in obj.values() for k in _keys(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in _keys(v)}
    return set()


class Q3Tests(unittest.TestCase):
    @staticmethod
    def _record(seed: int, rhos: list[float], literature: list[float]) -> dict:
        schemas = [{"name": f"s{i}", "relations": f"paper -[r{i} +]-> x -[r{i} -]-> paper",
                    "signature": [0, i, 2, 1, i, 0, 0], "span": 2, "c": 0.05, "I": 0.05 * r, "rho": r,
                    "mean_g": 0.0, "num_cells": 1} for i, r in enumerate(rhos)]
        return {"dataset": "acm", "seed": seed, "run": f"acm_{seed}", "path_source": "wisp", "schemas": schemas,
                "literature": [{"listed": f"L{i}", "relations": f"lit{i}", "rho": r}
                               for i, r in enumerate(literature)]}

    def test_T7_seeds_1_to_5_only_no_permutation_and_favour_at_5_of_5(self) -> None:
        import q3_schema as q3

        rhos = [2.0, 1.5, 1.2, 1.0, 0.8, 0.5]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            # seed 0 (main-table tuning seed): reversed ranking, literature all > 1 -> it must change nothing
            for seed in range(0, 6):
                ranking = rhos[::-1] if seed == 0 else rhos
                literature = [1.5, 1.5 if seed != 3 else 0.9, 0.5] if seed else [2.0, 2.0, 2.0]
                path = tmp / f"acm_seed{seed}.json"
                path.write_text(json.dumps(self._record(seed, ranking, literature)), encoding="utf-8")
                paths.append(path)
            ignored: list = []
            records = q3.load_records(paths, ignored)
            self.assertEqual([r["seed"] for r in records["acm"]], [1, 2, 3, 4, 5])
            self.assertEqual([x["seed"] for x in ignored], [0])
            result = q3.analyse(records, tmp / "out", None, make_figures=False, ignored=ignored)
            entry = result["datasets"]["acm"]
            self.assertEqual(entry["seeds"], [1, 2, 3, 4, 5])
            stability = entry["stability"]
            self.assertEqual((stability["pairs"], stability["stable"]), (10, True))
            self.assertAlmostEqual(stability["mean_tau_b_rho"], 1.0, places=12)   # seed 0's reversal not read
            keys = _keys(result)
            self.assertFalse({k for k in keys if "permutation" in k or k.startswith("p_") or k == "p"}, keys)
            literature = {x["listed"]: x for x in entry["literature"]}
            self.assertEqual((literature["L0"]["seeds_rho_above_1"], literature["L0"]["favour"]), (5, True))
            self.assertEqual((literature["L1"]["seeds_rho_above_1"], literature["L1"]["favour"]), (4, False))
            self.assertEqual((literature["L2"]["seeds_rho_above_1"], literature["L2"]["favour"]), (0, False))
            self.assertNotIn("sampler learns", (tmp / "out" / "q3_schema.md").read_text(encoding="utf-8"))

    def test_T7_homophily_scatter_from_the_link_points(self) -> None:
        import q3_schema as q3

        link = [{"seed": s, "points": 3, "schemas": ["a", "b", "c"], "status": "computed", "spearman": 0.5,
                 "values": [{"schema": n, "rho": 1.0 + 0.1 * i, "homophily": 0.2 * i}
                            for i, n in enumerate("abc")]} for s in range(1, 6)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(q3.homophily_scatter("acm", link, Path(tmp)))
            self.assertTrue(path.exists() and path.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
