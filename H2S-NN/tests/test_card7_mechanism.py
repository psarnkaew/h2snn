"""Card 7 T6 and T7 (Phase B): face effect F1-F3 and the case study (`scripts/analysis/face_effect.py`), the
two-regime analysis R1-R3 (`scripts/analysis/two_regime.py`) and their shared helpers (`mechanism_common.py`).

Fixture ("a tiny trained fixture"): a toy ACM release in HGB's format (`card7_fixtures.write_toy_hgb`; ACM has the
target-target relation P-P), and runs of `full`, `no_path_cells` and S2 at seeds 1-2 trained by
`scripts/run_h2snn_runs.py --dry_run` (one epoch) and reloaded from their run folders by `load_trained_model`; one
`full` run on a perturbed toy graph for the graph axis. Numerical identities use the float64 model of the card 5
tests (`test_card5_protocol_hooks._model`).

T6: flips counted from crafted predictions; the margin sign; F3 with hook (b) zeroes exactly the chosen node's cells
    and reads that node; eta_v equals a brute-force count over the reference meta-path graphs; the mini-batch views
    give the trainer's own evaluation logits; F1-F3 and the case study end to end on the trained runs.
T7: the composed endpoint transport equals the product of the fact transports in path order (crafted, <= 1e-12) and
    the model's own transport (`PathTransportOperator.apply_holonomy`, float64, <= 1e-12); R2 / R3 on crafted inputs;
    R1-R3 end to end, both axes.
"""

from __future__ import annotations

import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from tests.card7_fixtures import write_toy_hgb
from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "perturbation"))

import face_effect as fe  # noqa: E402
import mechanism_common as mc  # noqa: E402
import two_regime as tr  # noqa: E402

SMALL = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16, "classifier_hidden_dim": 16,
         "propagation": "anchored"}
ARMS = {"full": {}, "no_path_cells": {"model": {"arm": "no_path_cells"}},
        "S2": {"ablation": {"fact_maps_identity": True}}}
FIXTURE: dict = {}


def _config(root: Path, arm: str) -> dict:
    raw = {"device": "cpu", "seed": 1, "model": {**SMALL}, "ablation": {},
           "data": {"dataset_name": "acm", "root": str(root), "wisp_max_length": 3, "wisp_walks_per_anchor": 4}}
    for section, change in ARMS[arm].items():
        raw[section] = {**raw[section], **change}
    return raw


def _train(root: Path, arm: str, out: Path, exp: str) -> list[Path]:
    config = out / f"{exp}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(_config(root, arm)), encoding="utf-8")
    argv = ["run_h2snn_runs.py", "--config", str(config), "--base_seed", "1", "--num_runs", "2", "--exp_name", exp,
            "--output_dir", str(out), "--dry_run", "--device", "cpu"]
    with mock.patch.object(sys, "argv", argv):
        runpy.run_path(str(ROOT / "scripts" / "run_h2snn_runs.py"), run_name="__main__")
    return [out / f"{exp}_run{i}" for i in range(2)]


def setUpModule() -> None:
    unreachable_real_data()
    tmp = tempfile.TemporaryDirectory()
    FIXTURE["tmp"] = tmp
    root = Path(tmp.name) / "data"
    write_toy_hgb(root, "acm")
    out = Path(tmp.name) / "runs"
    FIXTURE["root"] = root
    FIXTURE["runs"] = {arm: _train(root, arm, out, f"toy_{arm}") for arm in ARMS}
    import make_graphs as mg

    record = mg.generate(root, "acm", 2.0, levels=(0.60,), log=lambda *_: None)[0]
    FIXTURE["graph"] = record
    FIXTURE["pert_runs"] = _train(Path(record["data_root"]), "full", out, "toy_pert_full")


def tearDownModule() -> None:
    FIXTURE["tmp"].cleanup()
    restore_real_data()


def _card5_model():
    from tests.test_card5_protocol_hooks import _model

    return _model()


# ----------------------------------------------------------------------------------------------------- T6
class FaceEffectUnitTests(unittest.TestCase):
    def test_T6_flips_from_crafted_predictions(self) -> None:
        full = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
        ablated = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
        quartile = np.array([0, 0, 1, 2, -1, 3, 3, 3])
        out = fe.flip_counts(full, ablated, quartile)
        self.assertEqual(out["Q1"], {"nodes": 2, "wrong_to_right": 1, "right_to_wrong": 0})
        self.assertEqual(out["Q2"], {"nodes": 1, "wrong_to_right": 0, "right_to_wrong": 1})
        self.assertEqual(out["Q3"], {"nodes": 1, "wrong_to_right": 0, "right_to_wrong": 0})
        self.assertEqual(out["Q4"], {"nodes": 3, "wrong_to_right": 1, "right_to_wrong": 1})
        self.assertEqual(out["none"], {"nodes": 1, "wrong_to_right": 1, "right_to_wrong": 0})

    def test_T6_margin_sign(self) -> None:
        centres = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
        x = torch.tensor([[1.0, 0.0], [9.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1])
        m = mc.margins(x, labels, centres)
        torch.testing.assert_close(m, torch.tensor([17 ** 0.5 - 1.0, 1.0 - 9.0, 1.0 - 9.0], dtype=torch.float64))
        self.assertGreater(float(m[0]), 0.0)                        # nearer its own centroid: positive
        self.assertLess(float(m[1]), 0.0)
        c = mc.centroids(x, labels, 3)
        torch.testing.assert_close(c[0], torch.tensor([5.0, 0.0], dtype=torch.float64))
        self.assertTrue(bool(torch.isnan(c[2]).all()))              # a class without a node

    def test_T6_quartiles(self) -> None:
        values = np.array([0.0, 0.0, 0.1, 0.2, 0.5, 0.9, np.nan, 1.0])
        edges = mc.quartile_edges(values)
        self.assertEqual(edges, [float(v) for v in np.quantile(values[~np.isnan(values)], [0.25, 0.5, 0.75])])
        q = mc.quartile_of(values, edges)
        self.assertEqual(q[6], -1)
        self.assertTrue(np.all(np.diff(q[np.argsort(np.nan_to_num(values, nan=-1))][1:]) >= 0))

    def test_T6_case_rule(self) -> None:
        rows = [{"node": 5, "right_to_wrong": True, "margin_change": -2.0},
                {"node": 3, "right_to_wrong": True, "margin_change": -2.0},
                {"node": 1, "right_to_wrong": True, "margin_change": -9.0},
                {"node": 7, "right_to_wrong": False, "margin_change": -20.0}]
        self.assertEqual(fe.pick_case(rows, {3, 5, 7})["node"], 3)   # node 1 has no different-class neighbour
        self.assertIsNone(fe.pick_case(rows, {7}))
        self.assertEqual(list(fe.f3_sample(np.arange(10), 20)), list(range(10)))
        drawn = fe.f3_sample(np.arange(100), 7)
        self.assertEqual(len(set(drawn.tolist())), 7)
        self.assertEqual(list(drawn), list(fe.f3_sample(np.arange(100)[::-1], 7)))   # fixed draw


class FaceEffectModelTests(unittest.TestCase):
    def test_T6_f3_knockout_changes_only_the_chosen_nodes_cells(self) -> None:
        from src.analysis.mechanism_hooks import knockout
        from src.analysis.model_loading import nonempty_groups

        run = mc.load_run(FIXTURE["runs"]["full"][0])
        whole = next(mc.views(run))
        base = mc.read(run.model, whole)
        base_betas = [b.clone() for b in run.model.last_cell_betas]
        groups = nonempty_groups(run.data.face_groups)
        test_ids = set(whole.ids.tolist())
        node = next(int(v) for g in groups for v in g.face_node_index.reshape(-1).tolist() if int(v) in test_ids)
        out = fe.knockout_read(run, node, whole)
        hit = 0
        for group, beta, before in zip(groups, out["betas"], base_betas):
            through = (group.face_node_index == node).any(dim=1)
            hit += int(through.sum())
            self.assertTrue(bool((beta[through] == 0).all()))
            self.assertTrue(torch.equal(beta[~through], before[~through]))
        self.assertGreater(hit, 0)
        with knockout(run.model, [node]):                           # the node read is the node knocked out
            manual = mc.cold_forward(run.model, run.data)
        i = int((whole.ids == node).nonzero()[0, 0])
        torch.testing.assert_close(out["knockout"]["logits"][0], manual.logits[whole.positions[i]].double(),
                                   atol=0.0, rtol=0.0)
        again = mc.read(run.model, whole)                           # cold start: nothing carried over
        self.assertTrue(torch.equal(again["logits"], base["logits"]))

    def test_T6_eta_equals_a_brute_force_count(self) -> None:
        values = mc.eta(FIXTURE["root"], "acm")
        data, target, ref = mc.reference_graph(FIXTURE["root"], "acm")
        y = data[target].y
        pairs = set()
        for path in ref.h2gb_metapaths(data, target):
            g = ref.metapath_graph(data, path, 7)
            pairs |= {(int(u), int(v)) for u, v in g.t().tolist() if u != v}
        n = int(data[target].num_nodes)
        expected = np.full(n, np.nan)
        for v in range(n):
            nb = [u for (u, w) in pairs if w == v and int(y[u]) >= 0]
            if nb and int(y[v]) >= 0:
                expected[v] = np.mean([int(y[u]) != int(y[v]) for u in nb])
        np.testing.assert_allclose(values["eta"], expected, rtol=0, atol=1e-15)
        self.assertTrue((values["neighbours"][~np.isnan(expected)] > 0).all())

    def test_T6_minibatch_views_give_the_trainers_logits(self) -> None:
        from tests.test_partMB_trainer import trainer_for

        with trainer_for(8, 16) as (cfg, graph, model, trainer):
            model.eval()
            run = mc.Run(Path("."), model, graph, cfg, "acm", int(cfg.seed), "full", Path(cfg.data.root))
            self.assertTrue(run.minibatch)
            ids, logits, labels = trainer.evaluate(graph, graph.target_test_mask)
            ours = mc.evaluate(run)
            self.assertTrue(torch.equal(ours["ids"], ids.cpu()))
            self.assertTrue(torch.equal(ours["labels"], labels))
            torch.testing.assert_close(ours["logits"], logits.double(), atol=1e-6, rtol=0.0)
            node = int(ids[0])
            single = mc.read(model, next(mc.views(run, targets=torch.tensor([node]))))
            torch.testing.assert_close(single["logits"][0], ours["logits"][0], atol=1e-6, rtol=0.0)

    def test_T6_face_effect_end_to_end(self) -> None:
        runs = [*FIXTURE["runs"]["full"], *FIXTURE["runs"]["no_path_cells"], FIXTURE["runs"]["S2"][0]]
        with tempfile.TemporaryDirectory() as out:
            out = Path(out)
            result = fe.analyse(runs, out, f3_seeds=(1,), f3_max_nodes=6, case=("acm", 1))
            self.assertEqual([s["arm"] for s in result["skipped_runs"]], ["S2"])
            acm = result["datasets"]["acm"]
            self.assertEqual([s["seed"] for s in acm["seeds"]], [1, 2])
            # F1 recounted from the two runs' own predictions
            full, npc = mc.load_run(FIXTURE["runs"]["full"][0]), mc.load_run(FIXTURE["runs"]["no_path_cells"][0])
            a, b = mc.evaluate(full), mc.evaluate(npc)
            ca, cb = mc.correct(a["logits"], a["labels"], False), mc.correct(b["logits"], b["labels"], False)
            f1 = acm["seeds"][0]["F1"]
            self.assertEqual(sum(v["wrong_to_right"] for v in f1.values()), int((ca & ~cb).sum()))
            self.assertEqual(sum(v["right_to_wrong"] for v in f1.values()), int((~ca & cb).sum()))
            self.assertEqual(sum(v["nodes"] for v in f1.values()), int(a["ids"].numel()))
            self.assertIn("F2", acm["seeds"][0])
            self.assertEqual(set(acm["seeds"][0]["F2"]), {"full", "no_path_cells"})
            self.assertNotIn("F3", acm["seeds"][1])                  # F3 on the listed seeds only
            rows = json.loads((out / "f3_acm_seed1.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), min(6, int(a["ids"].numel())))
            for r in rows:
                self.assertEqual(r["right_to_wrong"], r["base_correct"] and not r["knockout_correct"])
                self.assertAlmostEqual(r["margin_change"], r["margin_knockout"] - r["margin_base"], places=12)
            case = result["case_study"]
            self.assertIn(case["status"], ("chosen", "no candidate"))
            if case["status"] == "chosen":
                self.assertTrue((out / "case_study.pdf").stat().st_size > 0)
                self.assertLessEqual(len(case["cells"]), 3)
                self.assertTrue(all(case["node"] in c["node_ids"] for c in case["cells"]))
                self.assertEqual([c["w"] for c in case["cells"]], sorted((c["w"] for c in case["cells"]),
                                                                         reverse=True))
            self.assertTrue((out / "face_effect.md").read_text(encoding="utf-8").startswith("# Face effect"))

    def test_T6_case_study_on_a_crafted_drop(self) -> None:
        """The case-study path itself on the trained run: a node whose knockout result is forced right->wrong."""
        run = mc.load_run(FIXTURE["runs"]["full"][0])
        base = mc.evaluate(run)
        data, target, ref = mc.reference_graph(run.data_root, "acm")
        pool = []
        for v in base["ids"].tolist():
            nb = fe.different_class_neighbours(run, v, data, target, ref)
            if nb["fact"] or nb["metapath"]:
                pool.append(v)
        self.assertTrue(pool)
        rows = [{"node": v, "right_to_wrong": True, "margin_change": -1.0 - 0.1 * (v == pool[-1]),
                 "margin_base": 1.0, "margin_knockout": 0.0, "p_true_base": 0.6, "p_true_knockout": 0.3}
                for v in pool]
        with tempfile.TemporaryDirectory() as out:
            case = fe.case_study(run, base, rows, Path(out))
            self.assertEqual((case["status"], case["node"]), ("chosen", pool[-1]))
            labels = mc.labelled_targets(run.data)
            self.assertNotEqual(case["neighbour"]["label"], labels[case["node"]])
            self.assertTrue((Path(out) / "case_study.pdf").exists())
            self.assertEqual(len(case["pca"]["centroids"]), run.data.num_classes)


# ----------------------------------------------------------------------------------------------------- T7
def _orthogonal(k: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(k, d, d, generator=g, dtype=torch.float64))
    return q


class TwoRegimeUnitTests(unittest.TestCase):
    def test_T7_composition_is_the_product_in_path_order(self) -> None:
        q = _orthogonal(4, 5, 0)
        facts = torch.tensor([[0, 1, 2], [2, 0, 3]])
        signs = torch.tensor([[1, -1, 1], [-1, 1, 1]])
        t = tr.compose(q, facts, signs)
        expected = torch.stack([q[0] @ q[1].T @ q[2], q[2].T @ q[0] @ q[3]])
        self.assertLessEqual(float((t - expected).abs().max()), 1e-12)
        self.assertGreater(float((tr.compose(q, facts, -signs) - expected).abs().max()), 1e-3)
        reversed_order = torch.stack([q[2] @ q[1].T @ q[0], q[3] @ q[0] @ q[2].T])
        self.assertGreater(float((t - reversed_order).abs().max()), 1e-3)

    def test_T7_composition_equals_the_models_transport(self) -> None:
        from src.analysis.mechanism_hooks import transports_per_fact
        from src.analysis.model_loading import nonempty_groups

        _, data, model = _card5_model()
        model.record_solve_state = True
        mc.cold_forward(model, data)
        snap = model.last_solve_state
        model.record_solve_state = False
        op = snap.build(*snap.params).operator
        groups = nonempty_groups(data.face_groups)
        self.assertEqual(len(groups), len(op.groups))
        checked = 0
        for group, cell_group in zip(groups, op.groups):
            self.assertTrue(torch.equal(group.face_node_index, cell_group.face_node_index))
            facts = torch.unique(group.face_edge_index.reshape(-1))
            q = transports_per_fact(model, data, facts)["Q"]
            index = torch.full((int(facts.max()) + 1,), -1, dtype=torch.long)
            index[facts] = torch.arange(facts.numel())
            t = tr.compose(q, index[group.face_edge_index], group.face_edge_sign)
            d = model.d
            columns = [cell_group.transport.apply_holonomy(torch.eye(d, dtype=torch.float64)[k].expand(
                group.num_faces, d).clone()) for k in range(d)]
            holonomy = torch.stack(columns, dim=-1)                       # [P, d, d]: T_{v_L}
            self.assertLessEqual(float((t - holonomy).abs().max()), 1e-12)
            checked += group.num_faces
            if bool((group.face_edge_sign < 0).any()):
                checked += 0
        self.assertGreater(checked, 0)
        self.assertTrue(any(bool((g.face_edge_sign < 0).any()) for g in groups))   # -1 hops are exercised

    def test_T7_r2_on_crafted_inputs(self) -> None:
        e_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        wrong = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
        eta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, np.nan])
        quartile = np.array([0, 0, 1, 1, 2, 2, 3, -1])
        out = tr.r2(e_v, wrong, eta, quartile)
        self.assertEqual(out["auroc_wrong"], 1.0)
        self.assertAlmostEqual(out["spearman_eta"], 1.0)
        self.assertEqual(out["by_quartile"]["Q1"], {"nodes": 2, "mean_e_v": 1.5, "auroc_wrong": None})
        self.assertEqual(out["by_quartile"]["Q4"]["nodes"], 1)
        out = tr.r2(-e_v, wrong, eta, quartile)
        self.assertEqual(out["auroc_wrong"], 0.0)

    def test_T7_r3_on_crafted_inputs(self) -> None:
        x = np.array([[0.0, 1.0], [0.0, -1.0], [10.0, 1.0], [10.0, -1.0]])
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(tr.distance_ratio(x, y, 2), 0.1)
        self.assertAlmostEqual(tr.distance_ratio(x, y, 2, np.array([1, 0, 0, 0], dtype=bool)), 0.1)
        rng = np.random.default_rng(0)
        x_tr = np.concatenate([rng.normal((0, 0), 0.5, (20, 2)), rng.normal((10, 0), 0.5, (20, 2))])
        y_tr = np.repeat([0, 1], 20)
        out = tr.probe(x_tr, y_tr, x, y, False, np.array([0, 0, 1, 1]))
        self.assertEqual((out["macro_f1"], out["accuracy"], out["nodes"]), (1.0, 1.0, 4))
        self.assertEqual(out["by_quartile"]["Q1"]["nodes"], 2)
        multi = tr.probe(x_tr, np.stack([y_tr, 1 - y_tr], axis=1), x, np.stack([y, 1 - y], axis=1), True)
        self.assertEqual(multi["macro_f1"], 1.0)
        self.assertEqual(tr.split_summary(np.array([0.1, 0.2, 0.9, 1.0]), np.array([1, 1, 0, 0], dtype=bool)),
                         {"same": 2, "different": 2, "median_same": 0.15000000000000002, "median_different": 0.95,
                          "auroc_different": 1.0})


class TwoRegimeModelTests(unittest.TestCase):
    def test_T7_r1_cells_on_the_float64_model(self) -> None:
        from src.analysis.model_loading import nonempty_groups

        cfg, data, model = _card5_model()
        run = mc.Run(Path("."), model, data, cfg, "acm", 3, "full", Path("."))
        labels = mc.labelled_targets(data)
        cells = tr.r1_cells(run, labels, chunk=3)                     # chunks smaller than a group
        model.record_solve_state = True
        mc.cold_forward(model, data)
        snap = model.last_solve_state
        model.record_solve_state = False
        op = snap.build(*snap.params).operator
        expected = []
        for group, cell_group in zip(nonempty_groups(data.face_groups), op.groups):
            ids = group.face_node_index
            keep = torch.tensor([int(a) in labels and int(b) in labels for a, b in zip(ids[:, 0], ids[:, -1])])
            d = model.d
            cols = [cell_group.transport.apply_holonomy(torch.eye(d, dtype=torch.float64)[k].expand(
                group.num_faces, d).clone()) for k in range(d)]
            hol = torch.stack(cols, dim=-1)[keep]
            expected.append(tr.distance_to_identity(hol))
        expected = torch.cat(expected).numpy()
        self.assertEqual(len(cells["distance"]), len(expected))
        np.testing.assert_allclose(cells["distance"], expected, rtol=0, atol=1e-12)
        same = [labels[int(a)] == labels[int(b)] for a, b in zip(cells["start"], cells["end"])]
        self.assertEqual(list(cells["same"]), same)
        facts = tr.r1_facts(run, labels, chunk=5)
        self.assertEqual(facts["status"], "computed")
        self.assertEqual(facts["same"] + facts["different"], facts["facts"])

    def test_T7_two_regime_end_to_end_both_axes(self) -> None:
        runs = [*FIXTURE["runs"]["full"], *FIXTURE["runs"]["S2"], FIXTURE["runs"]["no_path_cells"][0],
                *FIXTURE["pert_runs"]]
        with tempfile.TemporaryDirectory() as out:
            out = Path(out)
            result = tr.analyse(runs, out)
            self.assertEqual(result["runs"], 6)
            self.assertEqual([s["arm"] for s in result["skipped_runs"]], ["no_path_cells"])
            records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "runs").glob("*.json"))]
            for r in records:
                self.assertIn("R3", r)
                self.assertGreater(r["tau"], 0.0)
                self.assertTrue(np.isfinite(r["R3"]["energy_tilde"]))
                self.assertEqual("R1" in r, r["arm"] == "full")
                self.assertEqual("R2" in r, r["arm"] == "full")
                if r["arm"] == "full":
                    self.assertEqual(r["R2"]["nodes"], sum(q["nodes"] for q in r["R2"]["by_quartile"].values())
                                     + r["test_nodes_without_eta"])
                    self.assertEqual(r["R1"]["facts"]["status"], "computed")       # ACM: P-P facts
            pert = [r for r in records if r["graph"] != "original"]
            self.assertEqual({r["graph"] for r in pert}, {"targeted_h0.60"})
            self.assertEqual({r["h"] for r in pert}, {FIXTURE["graph"]["h"]})
            summary = {(s["graph"], s["arm"]): s for s in result["summary"]}
            self.assertEqual(set(summary), {("original", "full"), ("original", "S2"), ("targeted_h0.60", "full")})
            self.assertEqual(summary[("original", "full")]["seeds"], [1, 2])
            s2 = [r for r in records if r["arm"] == "S2"]
            run = mc.load_run(FIXTURE["runs"]["S2"][0])
            energy = tr.node_energy(run)
            self.assertAlmostEqual(s2[0]["R3"]["energy_tilde"], energy["energy_tilde"], places=12)
            self.assertTrue((out / "two_regime.md").read_text(encoding="utf-8").startswith("# Two-regime"))


if __name__ == "__main__":
    unittest.main()
