"""Card 19: MLH and MDE on H2GB's meta-path set (`scripts/heterophily/h2gb_set_measures.py`), synthetic data only.

(a) Toy graph: papers with two paper–paper relations (cite, ref) and one paper–author relation (writes, with its
    reverse). For every H2GB meta-path, the mini-graph MDE equals `het.mde_reference` on that path's `G_P` built by
    hand (support of A₁A₂, self-pairs kept).
(b) MLH from a hand-made ℋ² record (two meta-paths, two fill seeds).
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch  # noqa: E402
from torch_geometric.data import HeteroData  # noqa: E402

from src.analysis import heterophily as het  # noqa: E402
from tests.partMB_fixtures import restore_real_data, unreachable_real_data  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "heterophily" / "h2gb_set_measures.py"


def _script():
    spec = importlib.util.spec_from_file_location("h2gb_set_measures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def toy() -> HeteroData:
    data = HeteroData()
    data["paper"].num_nodes = 5
    data["paper"].x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 0.], [0., 3.]])
    data["author"].num_nodes = 3
    data["paper", "cite", "paper"].edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 0, 0]])
    data["paper", "ref", "paper"].edge_index = torch.tensor([[0, 2, 4, 1], [2, 4, 1, 3]])
    writes = torch.tensor([[0, 1, 1, 2, 3, 4], [0, 0, 1, 1, 2, 2]])
    data["paper", "writes", "author"].edge_index = writes
    data["author", "rev_writes", "paper"].edge_index = writes.flip(0)
    return data


def hand_gp(data: HeteroData, path) -> torch.Tensor:
    (s, r1, m), (_, r2, d) = path
    a1 = torch.zeros(data[s].num_nodes, data[m].num_nodes)
    a2 = torch.zeros(data[m].num_nodes, data[d].num_nodes)
    e1, e2 = data[s, r1, m].edge_index, data[m, r2, d].edge_index
    a1[e1[0], e1[1]] = 1
    a2[e2[0], e2[1]] = 1
    return ((a1 @ a2) > 0).nonzero().t()


class Card19(unittest.TestCase):
    def test_a_mini_graph_mde_equals_reference_per_h2gb_path(self) -> None:
        mod = _script()
        data = toy()
        x = data["paper"].x
        paths = het.h2gb_metapaths(data.edge_types, "paper")
        self.assertEqual(len(paths), 5)  # cite/ref × cite/ref + writes-rev_writes
        rows = mod.mde_paths(data, "paper", x, paths, list(range(len(paths))), both=False)
        for path, row in zip(paths, rows):
            gp = hand_gp(data, path)
            self.assertGreater(gp.size(1), 0, het.metapath_name(path))
            self.assertFalse(row["empty"])
            self.assertAlmostEqual(row["value"], het.mde_reference(gp, x), places=6, msg=het.metapath_name(path))
        summary = mod.summarise_mde(rows, [het.metapath_name(p) for p in paths])
        self.assertAlmostEqual(summary["value"], sum(r["value"] for r in rows) / 5, places=12)

    def test_c_overlap_closed_form_equals_reference(self) -> None:
        mod = _script()
        g = torch.Generator().manual_seed(0)
        n, n_mid = 40, 6
        pairs = torch.stack([torch.randint(0, n, (90,), generator=g), torch.randint(0, n_mid, (90,), generator=g)])
        data = HeteroData()
        data["paper"].num_nodes = n
        data["paper"].x = torch.randn(n, 5, generator=g)
        data["field"].num_nodes = n_mid
        data["paper", "in", "field"].edge_index = pairs
        data["field", "rev_in", "paper"].edge_index = pairs.flip(0)
        path = [("paper", "in", "field"), ("field", "rev_in", "paper")]
        mini = het.labeled_only_metapath_data(data, path, "paper", torch.ones(n, dtype=torch.bool))
        fields, why = mod.overlap_clique_mde(mini, data["paper"].x)
        self.assertIsNotNone(fields, why)
        self.assertGreater(fields["k_max"], 2)  # cliques overlap (levels ≥ 3 used)
        gp = hand_gp(data, path)
        self.assertEqual(fields["edges"], gp.size(1))
        self.assertAlmostEqual(fields["value"], het.mde_reference(gp, data["paper"].x), places=5)
        mod.OVERLAP_GROUP_ELEMENTS = 15  # 3 occurrences per gather: groups split across chunks (carry)
        small, _ = mod.overlap_clique_mde(mini, data["paper"].x)
        self.assertAlmostEqual(small["value"], fields["value"], places=12)

    def test_b_mlh_from_h2_record(self) -> None:
        mod = _script()

        def rows(a: float, b: float) -> list[dict]:
            return [{"metapath": "p1", "h_edge": a, "h_adj": 0.0}, {"metapath": "p2", "h_edge": b, "h_adj": 0.0}]

        rec = {"h2gb_metapaths": ["p1", "p2", "p3"], "h2b": {"empty_gp_skipped": ["p3"]},
               "per_problem": {"label": {"seeds": [0, 1],
                                         "runs": [{"seed": 0, "per_metapath": rows(0.6, 0.8)},
                                                  {"seed": 1, "per_metapath": rows(0.5, 0.7)}],
                                         "labeled_only": {"per_metapath": [{"h_edge": 0.9}, {"h_edge": 0.7}]}}}}
        out = mod.mlh_from_h2_record(rec)
        self.assertAlmostEqual(out["value"], 0.3, places=12)
        self.assertEqual([round(v, 12) for v in out["value_over_fill_seeds"]], [0.3, 0.4])
        self.assertAlmostEqual(out["labeled_only"], 0.2, places=12)
        self.assertEqual((out["paths"], out["empty_paths"]), (3, 1))


if __name__ == "__main__":
    unittest.main()
