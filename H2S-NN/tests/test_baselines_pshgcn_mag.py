"""Card 1d H1 (author decisions G-2, D1-b): PSHGCN's ogbn-mag label part with `sparse_tools` as a vendor clone.

Synthetic MAG-typed fixture with H2GB's relation names; the real data root is unreachable.
1. Each diagonal PSHGCN uses equals the dense formula its source defines: `spspmm_diag_sym_ABA(A)` =
   diag((D_row^-1 A)(D_col^-1 A^T)), `spspmm_diag_sym_AAA(A)` = diag((D^-1 A)^2) (`csrc/cpu/spspmm_diag_cpu.cpp:20-85`).
   The kernels compute in float32 (`:8`), so every degree of the fixture is a power of two: every value is exactly
   representable and the comparison with float64 is exact.
2. Label route: changing validation and test labels leaves every input of the label part unchanged.
   Card mutation: the label one-hot includes the validation rows -> this test fails.
3. No module leak: importing the registry and the adapters loads neither `sortednp` nor `sparse_tools`; after the label
   part's import block neither remains; inside it the stub raises on use.
4. One training step of their model on the fixture is finite and moves the parameters.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

REPO = Path(__file__).resolve().parents[1]
BUILT = (REPO / "vendor" / "sparse_tools" / "sparse_tools" / "_spspmm_diag_cpu.so").is_file()
SKIP_REASON = "sparse_tools not built: scripts/setup/clone_vendors.sh --only sparse_tools; scripts/setup/build_sparse_tools.sh"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def mag_fixture(seed: int = 0):
    """8 papers (128-wide x, 3 classes), 4 authors, 2 institutions, 4 fields; every degree 1, 2 or 4:
    P-P two 4-cycles (degree 2); P-F: paper i -> fields i//2 and (i//2+1)%4 (paper 2, field 4); A-P: paper i written by
    authors i%4 and (i+1)%4 (paper 2, author 4); A-I: author j -> institution j%2 (author 1, institution 2)."""
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["paper"].x = torch.randn(8, 128, generator=g)
    data["author"].x = torch.randn(4, 128, generator=g)
    data["institution"].x = torch.randn(2, 128, generator=g)
    data["field_of_study"].x = torch.randn(4, 128, generator=g)
    data["paper"].y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    for name, rows in (("train", [0, 1, 2, 3]), ("val", [4, 5]), ("test", [6, 7])):
        mask = torch.zeros(8, dtype=torch.bool)
        mask[rows] = True
        data["paper"][f"{name}_mask"] = mask
    cycle = [(a, b) for block in (0, 4) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
             for a, b in [(a + block, b + block)]]
    pp = torch.tensor(cycle).t()
    data["paper", "cites", "paper"].edge_index = torch.cat([pp, pp.flip(0)], dim=1)
    pf = torch.tensor([(i, f) for i in range(8) for f in (i // 2, (i // 2 + 1) % 4)]).t()
    data["paper", "has_topic", "field_of_study"].edge_index = pf
    ap = torch.tensor([(a, i) for i in range(8) for a in (i % 4, (i + 1) % 4)]).t()
    data["author", "writes", "paper"].edge_index = ap
    data["author", "affiliated_with", "institution"].edge_index = torch.tensor([(j, j % 2) for j in range(4)]).t()
    return data


def _dense(adj) -> torch.Tensor:
    return (adj.to_dense() != 0).double()


@unittest.skipUnless(BUILT, SKIP_REASON)
class PSHGCNMagLabelPartTests(unittest.TestCase):
    def test_diagonals_equal_their_dense_formula(self) -> None:
        from src.baselines.adapters import _pshgcn_mag as mag

        data = mag_fixture()
        relations = mag.mag_relations(data)
        with mag.pshgcn_mag_modules() as modules:
            diags = mag.self_effect_diagonals(modules, relations)
        adjs = [adj for _, adj in relations]

        def aba(m: torch.Tensor) -> torch.Tensor:
            r, c = m.sum(1), m.sum(0)
            return torch.diag((m / r[:, None]) @ (m.t() / c[:, None]))

        def aaa(m: torch.Tensor) -> torch.Tensor:
            d = m.sum(1)
            return torch.diag((m / d[:, None]) @ (m / d[:, None]))

        expected = {"PFFP": aba(_dense(adjs[3])), "PPPP": aaa(_dense(adjs[2])), "PAAP": aba(_dense(adjs[1]).t())}
        degrees = torch.cat([_dense(a).sum(d) for a in adjs for d in (0, 1)])
        self.assertTrue(set(degrees[degrees > 0].tolist()) <= {1.0, 2.0, 4.0})
        for key, value in diags.items():
            with self.subTest(key=key):
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.equal(value.double(), expected[key]), f"{key}: {value} vs {expected[key]}")

    def test_label_part_reads_training_labels_only(self) -> None:
        from src.baselines.adapters import _pshgcn_mag as mag

        outputs = []
        for change in (False, True):
            data = mag_fixture()
            labels = data["paper"].y.clone()
            if change:
                held = data["paper"].val_mask | data["paper"].test_mask
                labels[held] = (labels[held] + 1) % 3
            num_nodes = {t: int(data[t].num_nodes) for t in mag.MAG_LETTERS}
            with mag.pshgcn_mag_modules() as modules:
                outputs.append(mag.label_inputs(modules, mag.mag_relations(data), num_nodes, labels,
                                                data["paper"].train_mask, 3))
        self.assertEqual(sorted(outputs[0]), sorted(outputs[1]))
        self.assertTrue(set(mag.LABEL_KEYS) <= set(outputs[0]))
        for key in outputs[0]:
            self.assertTrue(torch.equal(outputs[0][key], outputs[1][key]), f"held-out labels reach {key}")

    def test_no_sortednp_or_sparse_tools_left_in_sys_modules(self) -> None:
        import importlib

        from src.baselines.adapters import _pshgcn_mag as mag

        for name in ("src.baselines.registry", "src.baselines.adapters.pshgcn", "src.baselines.adapters.hgb_gnn"):
            importlib.import_module(name)
        self.assertNotIn("sortednp", sys.modules)
        self.assertNotIn("sparse_tools", sys.modules)
        with mag.pshgcn_mag_modules():
            stub = sys.modules["sortednp"]
            with self.assertRaisesRegex(RuntimeError, "sortednp is not available"):
                stub.intersect  # noqa: B018 - any use raises
        self.assertNotIn("sortednp", sys.modules)
        self.assertFalse([n for n in sys.modules if n == "sparse_tools" or n.startswith("sparse_tools.")])

    def test_one_step_is_finite_and_moves(self) -> None:
        from src.baselines.adapters import _pshgcn_mag as mag

        data = mag_fixture()
        num_nodes = {t: int(data[t].num_nodes) for t in mag.MAG_LETTERS}
        relations = mag.mag_relations(data)
        params = {"emb_dim": 16, "hidden_x": 16, "hidden_l": 16, "layers_x": 2, "layers_l": 2, "dropout": 0.0,
                  "input_drop": 0.0}
        with mag.pshgcn_mag_modules() as modules:
            feats, expander, coe_num = mag.feature_inputs(modules, relations, num_nodes, data["paper"].x, seed=0)
            label_feats = mag.label_inputs(modules, relations, num_nodes, data["paper"].y, data["paper"].train_mask, 3)
            torch.manual_seed(0)
            model = mag.build_mag_model(modules, 3, coe_num, expander, params)
        idx = data["paper"].train_mask.nonzero().view(-1)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()
        optimizer.zero_grad()
        out = model({k: v[idx] for k, v in feats.items()}, {k: label_feats[k][idx] for k in mag.LABEL_KEYS})
        loss = torch.nn.functional.cross_entropy(out, data["paper"].y[idx])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, model.parameters())))


if __name__ == "__main__":
    unittest.main()
