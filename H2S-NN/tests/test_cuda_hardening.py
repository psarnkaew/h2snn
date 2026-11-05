from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml


def _load_orthogonal_class():
    module_path = Path(__file__).resolve().parent.parent / "src" / "models" / "orthogonal.py"
    spec = importlib.util.spec_from_file_location("h2snn_orthogonal_hardening_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Orthogonal from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Orthogonal


class CudaHardeningTests(unittest.TestCase):
    def test_matrix_exp_config_is_rejected_before_seeding(self) -> None:
        from scripts import train_h2snn

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "unsafe.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {"dataset_name": "acm"},
                        "model": {"orthogonal_map": "matrix_exp"},
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                config=config_path,
                exp_name=None,
                device=None,
                decoder_type=None,
            )
            with patch.object(train_h2snn, "parse_args", return_value=args):
                with patch.object(train_h2snn, "seed_everything") as seed_mock:
                    with self.assertRaisesRegex(ValueError, "matrix_exp.*fasthpp"):
                        train_h2snn.main()
                    seed_mock.assert_not_called()

    def test_model_memory_guard_fraction_is_validated(self) -> None:
        from src.configs.config import ModelConfig

        with self.assertRaisesRegex(ValueError, "must be in"):
            ModelConfig(node_edge_memory_guard_fraction=0.0)
        with self.assertRaisesRegex(ValueError, "must be in"):
            ModelConfig(node_edge_memory_guard_fraction=1.1)
        self.assertAlmostEqual(ModelConfig().node_edge_memory_guard_fraction, 0.20)
        self.assertEqual(ModelConfig().orthogonal_map, "fasthpp")

    def test_cpu_seeding_does_not_touch_cuda_seed_state(self) -> None:
        from src.utils.misc import seed_everything

        with patch("torch.cuda.manual_seed_all") as cuda_seed:
            seed_everything(42, "cpu")
            cuda_seed.assert_not_called()

    def test_fasthpp_has_finite_orthogonal_parameterisation(self) -> None:
        Orthogonal = _load_orthogonal_class()
        d = 4
        expected_param_dim = d * (d - 1) // 2
        target = torch.randn(3, d, d)

        transform = Orthogonal(d=d, orthogonal_map="fasthpp")
        self.assertEqual(transform.parameter_count, expected_param_dim)
        params = torch.randn(3, expected_param_dim, requires_grad=True)
        output = transform(params)
        self.assertEqual(tuple(output.shape), (3, d, d))
        self.assertTrue(torch.isfinite(output).all())
        identity = torch.eye(d).expand(3, -1, -1)
        self.assertTrue(
            torch.allclose(
                output.transpose(-2, -1) @ output,
                identity,
                atol=1e-5,
                rtol=1e-5,
            )
        )
        (output * target).sum().backward()
        self.assertIsNotNone(params.grad)
        self.assertTrue(torch.isfinite(params.grad).all())

        for rejected in ("householder", "cayley", "euler", "matrix_exp"):
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(ValueError, "fasthpp"):
                    Orthogonal(d=d, orthogonal_map=rejected)

    def test_householder_wy_matches_the_sequential_builder(self) -> None:
        """The compact-WY builder is an implementation swap, not a reparameterisation.

        `_householder_orgqr` dispatches to `_householder_wy` for square power-of-two `d`, which
        is what every sweep uses. That is only safe if it returns the same matrix *and* the same
        gradient as the reference loop, so assert both in float64.
        """
        Orthogonal = _load_orthogonal_class()

        for d in (2, 4, 8, 16, 32):
            with self.subTest(d=d):
                raw = torch.randn(16, d, d, dtype=torch.float64)
                lifted = raw.tril(diagonal=-1) + torch.eye(d, dtype=torch.float64)
                sequential_input = lifted.clone().requires_grad_(True)
                wy_input = lifted.clone().requires_grad_(True)

                sequential = Orthogonal._householder_sequential(sequential_input)
                wy = Orthogonal._householder_wy(wy_input)
                self.assertLess((sequential - wy).abs().max().item(), 1e-12)

                # A non-degenerate objective: ||Q||_F^2 is constant on the orthogonal group, so
                # differentiating it would only compare roundoff.
                weights = torch.randn(d, d, dtype=torch.float64)
                (sequential * weights).sum().backward()
                (wy * weights).sum().backward()
                self.assertLess(
                    (sequential_input.grad - wy_input.grad).abs().max().item(), 1e-10
                )

    def test_householder_wy_routes_through_fasthpp(self) -> None:
        from src.models.fasthpp import householder_product_wy

        Orthogonal = _load_orthogonal_class()
        reflectors = torch.randn(5, 8, 8, dtype=torch.float64).tril(diagonal=-1)
        reflectors = reflectors + torch.eye(8, dtype=torch.float64)
        self.assertTrue(
            torch.equal(
                Orthogonal._householder_wy(reflectors),
                householder_product_wy(reflectors),
            )
        )

    def test_householder_orgqr_dispatches_on_shape(self) -> None:
        """Non-power-of-two and non-square inputs must fall back to the sequential builder."""
        Orthogonal = _load_orthogonal_class()

        for d in (3, 5, 6, 12):
            with self.subTest(d=d):
                lifted = torch.randn(4, d, d, dtype=torch.float64).tril(diagonal=-1)
                lifted = lifted + torch.eye(d, dtype=torch.float64)
                dispatched = Orthogonal._householder_orgqr(lifted)
                expected = Orthogonal._householder_sequential(lifted)
                self.assertLess((dispatched - expected).abs().max().item(), 1e-12)
                identity = torch.eye(d, dtype=torch.float64).expand(4, -1, -1)
                self.assertTrue(
                    torch.allclose(dispatched.transpose(-2, -1) @ dispatched, identity, atol=1e-10)
                )

    def test_memory_guard_accepts_safe_and_rejects_unsafe_workloads(self) -> None:
        from src.utils.memory_guard import (
            estimate_node_edge_live_bytes,
            validate_node_edge_memory_budget,
        )

        safe_bytes = estimate_node_edge_live_bytes(
            num_edges=1_000,
            d=16,
            param_dim=120,
            use_orthogonal=True,
            element_size=4,
        )
        self.assertGreater(safe_bytes, 0)
        validate_node_edge_memory_budget(
            num_edges=1_000,
            d=16,
            param_dim=120,
            use_orthogonal=True,
            element_size=4,
            total_device_bytes=16 * 1024**3,
            guard_fraction=0.20,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"edges=547,872.*d=128.*estimated minimum live memory",
        ):
            validate_node_edge_memory_budget(
                num_edges=547_872,
                d=128,
                param_dim=128 * 127 // 2,
                use_orthogonal=True,
                element_size=4,
                total_device_bytes=80 * 1024**3,
                guard_fraction=0.20,
            )


if __name__ == "__main__":
    unittest.main()
