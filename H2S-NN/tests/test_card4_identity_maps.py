"""Card 4 K1 (`docs/H2S-NN_card4_ablation_prep_card.md`; spec 1.23 X-11, pre-registration §2): the
identity-map arms S2 (`model.arm: full`, `ablation.maps: identity`) and S0 (`no_path_cells` with
`maps: identity`). Synthetic graphs only (the real data root is unreachable).

(1) S2: every transport is exactly `I` (`torch.equal`) on a synthetic ACM-shaped WISP draw, through the
    model's own map operator.
(2) The Jacobi diagonal under S2 equals the one built from dense identity maps `c_e I` (float64,
    atol 1e-12).
(3) `full` bit-identical: run separately (`v3b_golden.py compare`, `mb0_reference.py compare`; E, 1
    thread, one node), not here.
(4) S2 (and S0): one training step is finite and every parameter moves; the frames and the reflector
    MLP do not exist in the arm (so they receive no gradient); the tuning validation refuses the key.
"""

from __future__ import annotations

import unittest

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

DIAGONAL_TOL = 1e-12
SMALL = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
         "classifier_hidden_dim": 16, "propagation": "anchored"}


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _config(arm: str = "full", maps: str = "identity"):
    from src.configs.config import TrainConfig

    return TrainConfig.from_dict({
        "device": "cpu", "seed": 3,
        "data": {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 5},
        "model": {**SMALL, "arm": arm},
        "ablation": {"maps": maps}})


def _data_and_model(arm: str = "full", maps: str = "identity"):
    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model
    from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

    cfg = _config(arm, maps)
    with patched_acm_loader(acm_like_hetero(seed=9)):
        data = build_dataset(cfg)
    torch.manual_seed(0)
    return data, build_model(cfg, data)


class IdentityTransportTests(unittest.TestCase):
    def test_s2_every_transport_is_exactly_identity(self) -> None:
        from src.models.map_operators import IdentityMapOperator
        from src.models.sheaf_energy import build_cell_group

        data, model = _data_and_model()
        model.eval()
        with torch.no_grad():
            h0 = model.encode_nodes(data.x, data.node_type)
            geometry = model.geometry_inputs(h0, data.edge_index, data.edge_type)
            maps = model.node_to_edge(h=geometry, edge_index=data.edge_index,
                                      edge_type=data.edge_type,
                                      relation_embeddings=model.relation_embedding.weight).maps
        self.assertIsInstance(maps.src, IdentityMapOperator)
        self.assertIsInstance(maps.tgt, IdentityMapOperator)
        groups = [g for g in data.face_groups if g.num_faces > 0]
        self.assertTrue(groups)
        self.assertTrue(any(bool((g.face_edge_sign < 0).any()) for g in groups),
                        "the draw must contain backward hops")
        x = torch.randn(data.num_nodes, model.d, generator=torch.Generator().manual_seed(5))
        for group in groups:
            cell = build_cell_group(maps, group, data.edge_index,
                                    torch.ones(group.num_faces))
            transported = cell.transport.apply(x)                 # T_{v_i,f} x_{v_i}
            self.assertTrue(torch.equal(transported, x[group.face_node_index]))
            z = torch.randn(group.num_faces, model.d)
            prefix = cell.transport.adjoint_prefix(z)             # T_{v_i,f}^T z_f
            self.assertTrue(torch.equal(prefix, z.unsqueeze(1).expand_as(prefix)))
            self.assertTrue(torch.equal(cell.transport.apply_holonomy(z), z))


class IdentityJacobiTests(unittest.TestCase):
    def test_s2_jacobi_diagonal_equals_dense_identity_maps(self) -> None:
        from src.models.anchored_solve import AnchoredSystem
        from src.models.map_operators import DenseMapOperator, EdgeMapPair, IdentityMapOperator
        from src.models.sheaf_energy import EliminatedOperator, SheafEnergyOperator, build_cell_group
        from tests.test_sheaf_energy import Fixture

        fx = Fixture(dtype=torch.float64)
        scale = fx.maps.src.scale
        eye = torch.eye(fx.d, dtype=torch.float64)
        dense = DenseMapOperator(scale.view(-1, 1, 1) * eye, scale)
        pairs = {"identity": EdgeMapPair(IdentityMapOperator(scale, fx.d),
                                         IdentityMapOperator(scale, fx.d)),
                 "dense": EdgeMapPair(dense, dense)}
        tau = torch.tensor(1.7, dtype=torch.float64)
        jacobi = {}
        for name, pair in pairs.items():
            groups = [build_cell_group(pair, g, fx.edge_index, b)
                      for g, b in zip(fx.groups, fx.beta_by_group)]
            reduced = EliminatedOperator(SheafEnergyOperator(pair, fx.edge_index, fx.num_nodes,
                                                             groups))
            jacobi[name] = AnchoredSystem(reduced, tau).jacobi_inverse()
        self.assertEqual(jacobi["identity"].dtype, torch.float64)
        torch.testing.assert_close(jacobi["identity"], jacobi["dense"], atol=DIAGONAL_TOL, rtol=0.0)


class IdentityArmStepTests(unittest.TestCase):
    def _one_step(self, arm: str) -> None:
        from src.builders import build_optimizer

        data, model = _data_and_model(arm=arm)
        names = dict(model.named_parameters())
        self.assertFalse([n for n in names if "reflector_mlp" in n or "rotation_params" in n],
                         "the frames and the reflector MLP are not parameters of maps = I")
        self.assertIn("node_to_edge.edge_scale_mlp.0.weight", names)   # a_e = c_e^2 stays learned
        cfg = _config(arm)
        optimizer = build_optimizer(model, cfg)
        before = {n: p.detach().clone() for n, p in names.items()}
        model.train()
        logits = model(data, epoch=0).logits
        mask = data.target_train_mask
        loss = torch.nn.functional.cross_entropy(logits[mask], data.target_y[mask])
        self.assertTrue(bool(torch.isfinite(loss)))
        optimizer.zero_grad()
        loss.backward()
        # S0 = no_path_cells: cells are not in the solve, so g (path encoder and gate) gets no
        # gradient, exactly as in the learned no_path_cells arm; S2: every parameter has one.
        unused = (("path_encoder.", "path_gate.") if arm == "no_path_cells" else ())
        expected = sorted(n for n in names if n.startswith(unused)) if unused else []
        no_grad = sorted(n for n, p in names.items() if p.grad is None)
        self.assertEqual(no_grad, expected, f"parameters without gradient: {no_grad}")
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in names.values()
                            if p.grad is not None))
        optimizer.step()
        still = [n for n, p in names.items() if n not in expected
                 and torch.equal(p.detach(), before[n])]
        self.assertEqual(still, [], f"parameters that did not move: {still}")

    def test_s2_one_step_finite_and_every_parameter_moves(self) -> None:
        self._one_step("full")

    def test_s0_one_step_finite_and_every_parameter_moves(self) -> None:
        self._one_step("no_path_cells")


class IdentityConfigTests(unittest.TestCase):
    def test_key_validated_and_refused_in_tuning(self) -> None:
        from src.configs.config import reject_never_tuned_settings

        self.assertEqual(_config(maps="learned").ablation.maps, "learned")
        with self.assertRaises(ValueError):
            _config(maps="diagonal")
        with self.assertRaises(ValueError):
            _config(arm="closure_only")            # maps = I is defined for full and no_path_cells
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"ablation": {"maps": "identity"}})
        with self.assertRaises(ValueError):
            reject_never_tuned_settings(None, {"maps": {"values": ["learned", "identity"]}})
        reject_never_tuned_settings({"ablation": {"maps": "learned"}})


if __name__ == "__main__":
    unittest.main()
