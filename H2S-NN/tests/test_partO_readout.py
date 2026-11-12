"""Part O, O2 (`docs/H2S-NN_partO_card.md`): the schema read-out (`src/analysis/schema_readout.py`,
`scripts/schema_readout.py`).

- hand-built cells with known m_f, w_f: c_sigma, I_sigma, rho_sigma and mean g equal hand values;
- constant g (path gate weights zero) -> rho_sigma = 1 for every sigma (1e-12, float64);
- w_f of a v3-B anchored model (`evaluation_cell_weights`) equal `H2SNN._path_weights` on the draw;
- the ACM literature list maps to the expected typed relation sequences (and PTP, the card's
  example); a misspelled entry raises; a listed path absent from a draw is stated, not dropped;
- the script writes one JSON per run and one table per dataset (synthetic runs; no value is read).

Card mutation: I_sigma computed without m_f fails `test_hand_values`.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.analysis.schema_readout import (
    literature_signature,
    literature_signatures,
    run_record,
    schema_name,
    schema_statistics,
)
from src.configs.config import LITERATURE_METAPATHS, MetaPathConfig
from src.data.structures import FaceGroup
from tests.partO_fixtures import acm_like_hetero, patched_acm_loader


def _group(nodes, relations, signs, m) -> FaceGroup:
    nodes = torch.tensor(nodes)
    return FaceGroup(template_name="hand", node_type_sequence=(), face_node_index=nodes,
                     face_edge_index=torch.zeros(nodes.size(0), nodes.size(1) - 1, dtype=torch.long),
                     relation_index=torch.tensor(relations),
                     face_edge_sign=torch.tensor(signs, dtype=torch.float32),
                     path_weight=torch.tensor(m, dtype=torch.float32))


# Papers 0-2 (type 0), authors 3-4 (type 1); relation 0 stored paper -> author.
HAND_NODE_TYPE = torch.tensor([0, 0, 0, 1, 1])
HAND_GROUPS = (
    _group([[0, 3, 1], [1, 4, 2]], [[0, 0], [0, 0]], [[1, -1], [1, -1]], [0.5, 1.0]),  # P-A-P
    _group([[0, 3, 1, 4]], [[0, 0, 0]], [[1, -1, 1]], [0.25]),                        # P-A-P-A
)
HAND_WEIGHTS = (torch.tensor([2.0, 0.5], dtype=torch.float64),
                torch.tensor([4.0], dtype=torch.float64))
PA, PAP, APA, PAPA = (0, 0, 2, 1), (0, 0, 2, 1, 0, 0, 0), (1, 0, 0, 0, 0, 2, 1), \
    (0, 0, 2, 1, 0, 0, 0, 0, 2, 1)


def _hand_expected() -> dict:
    """By hand. Pair weight w m / (2|f|): P-A-P cells 2*0.5/6 = 1/6 and 0.5*1/6 = 1/12 (pairs PA,
    PA, PAP); the P-A-P-A cell 4*0.25/8 = 1/8 (pairs PA x3, PAP, APA, PAPA). With w = 1: 1/12, 1/6,
    1/32."""
    learned = {PA: 2 * Fraction(1, 6) + 2 * Fraction(1, 12) + 3 * Fraction(1, 8),
               PAP: Fraction(1, 6) + Fraction(1, 12) + Fraction(1, 8),
               APA: Fraction(1, 8), PAPA: Fraction(1, 8)}
    uniform = {PA: 2 * Fraction(1, 12) + 2 * Fraction(1, 6) + 3 * Fraction(1, 32),
               PAP: Fraction(1, 12) + Fraction(1, 6) + Fraction(1, 32),
               APA: Fraction(1, 32), PAPA: Fraction(1, 32)}
    total_w, total_1 = sum(learned.values()), sum(uniform.values())
    two = math.log(2.0)
    mean_g = {PA: (two - two + 2 * two) / 3, PAP: (two - two + 2 * two) / 3, APA: 2 * two,
              PAPA: 2 * two}
    return {key: (float(uniform[key] / total_1), float(learned[key] / total_w),
                  float((learned[key] / total_w) / (uniform[key] / total_1)), mean_g[key])
            for key in learned}


class HandTests(unittest.TestCase):
    def test_hand_values(self) -> None:
        stats = schema_statistics(HAND_GROUPS, HAND_WEIGHTS, HAND_NODE_TYPE)
        expected = _hand_expected()
        self.assertEqual(set(stats), set(expected))
        for key, (c, share, rho, mean_g) in expected.items():
            with self.subTest(schema=key):
                self.assertAlmostEqual(stats[key].coverage, c, places=12)
                self.assertAlmostEqual(stats[key].share, share, places=12)
                self.assertAlmostEqual(stats[key].ratio, rho, places=12)
                self.assertAlmostEqual(stats[key].mean_g, mean_g, places=12)
        self.assertEqual(stats[PA].num_cells, 3)
        self.assertEqual(stats[APA].num_cells, 1)
        self.assertEqual({key: stats[key].span for key in stats}, {PA: 1, PAP: 2, APA: 2, PAPA: 3})

    def test_constant_weights_give_ratio_one(self) -> None:
        from tests.test_wisp import NODE_TYPE, SELF_INVERSE, _fixture_groups

        groups = tuple(g for g in _fixture_groups(4) if g.num_faces)
        weights = tuple(torch.full((g.num_faces,), 3.7, dtype=torch.float64) for g in groups)
        stats = schema_statistics(groups, weights, NODE_TYPE, SELF_INVERSE)
        self.assertGreater(len(stats), 5)
        for key, stat in stats.items():
            with self.subTest(schema=key):
                self.assertLessEqual(abs(stat.ratio - 1.0), 1e-12)
                self.assertLessEqual(abs(stat.mean_g - math.log(3.7)), 1e-12)


def _wisp_config(**data):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({
        "device": "cpu", "seed": 4,
        "data": {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 6, **data},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
        "ablation": {"coupling": "symmetric"}})


def _wisp_model_and_data():
    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model

    cfg = _wisp_config()
    with patched_acm_loader(acm_like_hetero(seed=3)):
        data = build_dataset(cfg)
    torch.manual_seed(0)
    return build_model(cfg, data).eval(), data, cfg


class ModelWeightTests(unittest.TestCase):
    def test_weights_equal_the_models_path_weights(self) -> None:
        from src.analysis.model_loading import evaluation_cell_weights, nonempty_groups

        model, data, _ = _wisp_model_and_data()
        weights = evaluation_cell_weights(model, data)
        groups = nonempty_groups(data.face_groups)
        self.assertEqual(len(weights), len(groups))
        with torch.no_grad():
            h0 = model.encode_nodes(data.x, data.node_type)
            path_input = model.path_inputs(h0, model.geometry_inputs(h0, data.edge_index,
                                                                     data.edge_type))
            for group, w in zip(groups, weights):
                reference = model._path_weights(model._path_latent(path_input, group), group, h0)
                self.assertTrue(torch.equal(w, reference))
        self.assertFalse(model.training or any(m.training for m in model.modules()))

    def test_constant_g_of_a_model_gives_ratio_one(self) -> None:
        from src.analysis.model_loading import evaluation_cell_weights, nonempty_groups

        model, data, _ = _wisp_model_and_data()
        with torch.no_grad():
            model.path_gate.weight.zero_()  # g = its bias b: constant over cells
        groups = nonempty_groups(data.face_groups)
        weights = evaluation_cell_weights(model, data)
        self.assertEqual(len({float(v) for w in weights for v in w}), 1)
        stats = schema_statistics(groups, tuple(w.double() for w in weights), data.node_type,
                                  data.fact_layer.self_inverse)
        for key, stat in stats.items():
            with self.subTest(schema=key):
                self.assertLessEqual(abs(stat.ratio - 1.0), 1e-12)


def _acm_layer():
    from src.data.fact_layer import build_fact_layer
    from src.data.hetero_flatten import flatten_edges, stack_node_features

    hetero = acm_like_hetero(seed=0)
    x, _, type_names, slices, _ = stack_node_features(hetero)
    edge_index, edge_type, names, tuples = flatten_edges(hetero, slices)
    layer = build_fact_layer(edge_index, edge_type, names, x.size(0), "tied")
    return layer, tuple(tuples[r] for r in layer.kept), type_names


class LiteratureMappingTests(unittest.TestCase):
    def test_acm_list_maps_to_the_expected_signatures(self) -> None:
        layer, tuples, types = _acm_layer()
        self.assertEqual(types, ("paper", "author", "subject", "term"))
        self.assertEqual(layer.kept_names, ("paper__cite__paper", "paper__to__author",
                                            "paper__to__subject", "paper__to__term"))
        # by hand: (t0, r1, d1 + 1, t1, ...), the smaller orientation; cite = 0 (ref = cite, -),
        # paper-author = 1, paper-subject = 2, paper-term = 3; paper 0, author 1, subject 2, term 3
        expected = {
            "PAP": (0, 1, 2, 1, 1, 0, 0),
            "PSP": (0, 2, 2, 2, 2, 0, 0),
            "PcPAP": (0, 0, 2, 0, 1, 2, 1, 1, 0, 0),
            "PcPSP": (0, 0, 2, 0, 2, 2, 2, 2, 0, 0),
            "PrPAP": (0, 0, 0, 0, 1, 2, 1, 1, 0, 0),
            "PrPSP": (0, 0, 0, 0, 2, 2, 2, 2, 0, 0),
        }
        self.assertEqual(literature_signatures("acm", layer, tuples, types), expected)
        self.assertEqual([t.name for t in LITERATURE_METAPATHS["acm"]], list(expected))
        # the card's example PTP, as a typed relation sequence
        ptp = MetaPathConfig("PTP", ("paper", "term", "paper"),
                             hops=(("paper__to__term", "+"), ("term__to__paper", "+")))
        self.assertEqual(literature_signature(ptp, layer, tuples, types), (0, 3, 2, 3, 3, 0, 0))
        names = {schema_name(s, types, layer.kept_names)[0] for s in expected.values()}
        self.assertEqual(names, {"paper-author-paper", "paper-subject-paper",
                                 "paper-paper-author-paper", "paper-paper-subject-paper"})

    def test_misspelled_entry_raises(self) -> None:
        layer, tuples, types = _acm_layer()
        relation = MetaPathConfig("PAP", ("paper", "author", "paper"),
                                  hops=(("paper__to__autor", "+"), ("author__to__paper", "+")))
        with self.assertRaises(KeyError):
            literature_signatures("acm", layer, tuples, types, templates=(relation,))
        node_type = MetaPathConfig("PAP", ("paper", "authr", "paper"),
                                   hops=(("paper__to__author", "+"), ("author__to__paper", "+")))
        with self.assertRaises(ValueError):  # the hop does not join paper -> 'authr'
            literature_signatures("acm", layer, tuples, types, templates=(node_type,))

    def test_listed_path_absent_from_the_draw_is_stated(self) -> None:
        layer, tuples, types = _acm_layer()
        literature = literature_signatures("acm", layer, tuples, types)
        record = run_record({}, types, layer.kept_names, literature, {"run": "r"})
        self.assertEqual([item["listed"] for item in record["literature"]],
                         [t.name for t in LITERATURE_METAPATHS["acm"]])
        self.assertTrue(all(not item["in_draw"] and item["c"] == 0.0 and item["rho"] is None
                            for item in record["literature"]))
        self.assertEqual(literature_signatures("ogbn-mag", layer, tuples, types), {})
        self.assertIsNone(run_record({}, types, layer.kept_names, {}, {"run": "r"})["literature"])


class ScriptTests(unittest.TestCase):
    """End to end on two synthetic 1-epoch runs: files and structure only."""

    def test_script_writes_run_json_and_dataset_table(self) -> None:
        import scripts.schema_readout as readout
        from src.builders import build_experiment_components
        from src.configs.config import TrainConfig
        from src.utils.misc import save_config

        hetero = acm_like_hetero(seed=6)
        with tempfile.TemporaryDirectory() as tmp, patched_acm_loader(hetero):
            run_dirs = []
            for seed in (0, 1):
                raw = {"device": "cpu", "seed": seed, "epochs": 1, "exp_name": f"run{seed}",
                       "output_dir": tmp, "checkpoint_metric": "macro_f1",
                       "checkpoint_mode": "max",
                       "data": {"dataset_name": "acm", "wisp_max_length": 3,
                                "wisp_walks_per_anchor": 4},
                       "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16,
                                 "face_mlp_hidden_dim": 16, "classifier_hidden_dim": 16,
                                 "propagation": "anchored", "arm": "full"},
                       "ablation": {"coupling": "symmetric"}}
                cfg = TrainConfig.from_dict(raw)
                cfg.exp_dir.mkdir(parents=True)
                save_config(cfg, cfg.exp_dir / "config.yaml")
                data, _, _, _, logger, _, trainer = build_experiment_components(cfg)
                try:
                    trainer.fit(data)
                finally:
                    logger.close()
                run_dirs.append(cfg.exp_dir)
            out = Path(tmp) / "readout"
            readout.main(["--runs", *map(str, run_dirs), "--out_dir", str(out),
                          "--permutations", "20"])
            for seed, run_dir in enumerate(run_dirs):
                record = json.loads((out / "acm" / f"{run_dir.name}.json").read_text("utf-8"))
                self.assertEqual((record["dataset"], record["seed"]), ("acm", seed))
                self.assertEqual([item["listed"] for item in record["literature"]],
                                 [t.name for t in LITERATURE_METAPATHS["acm"]])
                self.assertEqual(record["num_distinct_schemas"], len(record["schemas"]))
                shares = [s["I"] for s in record["schemas"]]
                self.assertAlmostEqual(sum(shares), 1.0, places=9)
                self.assertEqual(shares, sorted(shares, reverse=True))
                self.assertAlmostEqual(sum(s["c"] for s in record["schemas"]), 1.0, places=9)
            table = (out / "acm" / "schema_readout.md").read_text("utf-8")
            for heading in ("Listed meta-paths", "Top ten length-2", "Distinct sub-path",
                            "Kendall tau"):
                self.assertIn(heading, table)


if __name__ == "__main__":
    unittest.main()
