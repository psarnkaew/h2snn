"""v3-B card U4 (`docs/H2S-NN_v3B_card.md` §2): `model.propagation: anchored` and its arms.

T14 every arm runs forward + backward on the fixture; the loss is finite; every parameter group
    receives a nonzero gradient where it should and a zero one where it should not (`no_path_cells`
    never reads g; `hub_A` solves without fact states; the free fact->cell maps exist only in
    `free_cell_maps`); theta_tau is in the scalar optimizer group; the solver converged below
    T_max; the config keys are validated and never tuned.
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch is required for these tests: {exc}") from exc

from src.configs.config import V3B_ARMS, TrainConfig, reject_never_tuned_settings
from src.models.h2snn_model import H2SNN, ModelDims
from tests.test_factorized_maps import _random_pap_data


def _config(arm: str = "full", **model: object) -> TrainConfig:
    settings = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                "classifier_hidden_dim": 16, "propagation": "anchored", "arm": arm}
    settings.update(model)
    return TrainConfig.from_dict({
        "data": {"dataset_name": "acm", "path_source": "schemas"},
        "model": settings,
        "ablation": {"coupling": "symmetric"},
        "device": "cpu",
    })


def _group_norms(model: H2SNN) -> dict[str, float | None]:
    """Gradient norm per parameter group (None: the group does not exist in this arm)."""
    groups: dict[str, list[torch.nn.Parameter]] = {
        "node_encoder": list(model.node_encoder.parameters()),  # type: ignore[union-attr]
        "relation_embedding": list(model.relation_embedding.parameters()),
        "fact_maps": list(model.node_to_edge.parameters()),
        "g": [p for module in (model.path_encoder, model.path_gate) if module is not None
              for p in module.parameters()],
        "free_cell_maps": ([] if model.free_cell_maps is None
                           else list(model.free_cell_maps.parameters())),
        "theta_tau": [] if model.theta_tau is None else [model.theta_tau],
        "readout": list(model.readout.parameters()),  # type: ignore[union-attr]
    }
    norms: dict[str, float | None] = {}
    for name, params in groups.items():
        if not params:
            norms[name] = None
            continue
        total = sum(float(p.grad.norm()) ** 2 for p in params if p.grad is not None)
        norms[name] = math.sqrt(total)
    return norms


class ArmTests(unittest.TestCase):
    def test_T14_every_arm_trains_and_gradients_reach_the_right_groups(self) -> None:
        data = _random_pap_data()
        results: dict[str, dict[str, float | None]] = {}
        for arm in V3B_ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(0)
                config = _config(arm)
                model = H2SNN(config, ModelDims.from_prepared_data(data))
                self.assertIsNotNone(model.theta_tau)
                self.assertIsNone(model.filter_bank)
                self.assertIsNone(model.theta_eta)
                self.assertIsNone(model.listen_gate)
                self.assertIn(model.theta_tau, model.scalar_parameters())
                model.train()
                output = model(data, epoch=0)
                self.assertEqual(output.logits.shape, (data.target_indices.numel(), data.num_classes))
                loss = torch.nn.functional.cross_entropy(
                    output.logits[data.target_train_mask], data.target_y[data.target_train_mask])
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                norms = _group_norms(model)
                results[arm] = norms
                stats = model.last_solver_stats
                self.assertEqual(stats["pcg_converged"], 1.0)
                self.assertLess(stats["pcg_fwd_iters"], config.model.solver_max_iterations)
                self.assertLess(stats["pcg_bwd_iters"], config.model.solver_max_iterations)
                self.assertGreater(stats["pcg_bwd_iters"], 0)
                self.assertEqual(stats["solver_tolerance"], config.model.solver_warmup_tolerance)
                for name in ("node_encoder", "relation_embedding", "fact_maps", "theta_tau",
                             "readout"):
                    self.assertGreater(norms[name], 0.0, f"{arm}: {name} received no gradient")
                if arm == "no_path_cells":
                    # beta_f = 0 and no feature path: g is never read
                    self.assertEqual(norms["g"], 0.0)
                else:
                    self.assertGreater(norms["g"], 0.0, f"{arm}: g received no gradient")
                if arm == "free_cell_maps":
                    self.assertGreater(norms["free_cell_maps"], 0.0)
                else:
                    self.assertIsNone(norms["free_cell_maps"])
                if arm in ("hub_A", "closure_only", "no_path_cells", "instance_features"):
                    self.assertEqual(stats["num_fact_states"], 0.0, f"{arm}: fact states present")
                else:
                    self.assertGreater(stats["num_fact_states"], 0.0)
                if arm in ("closure_only", "no_path_cells", "instance_features"):
                    self.assertEqual(stats["num_cells"], 0.0)
                else:
                    self.assertGreater(stats["num_cells"], 0.0)
                self.assertEqual(output.eta, None)
                self.assertEqual(output.gamma, None)
                # evaluation: the warm start is stored and reused on the same draw
                model.eval()
                with torch.no_grad():
                    model(data, epoch=-1)
                    first = model.last_solver_stats["pcg_fwd_iters"]
                    model(data, epoch=-1)
                    second = model.last_solver_stats["pcg_fwd_iters"]
                self.assertLessEqual(second, first)
                self.assertEqual(model.last_solver_stats["solver_tolerance"],
                                 config.model.solver_tolerance)
        self.assertEqual(set(results), set(V3B_ARMS))

    def test_T14_config_keys_are_validated_and_never_tuned(self) -> None:
        base = {"data": {"dataset_name": "acm", "path_source": "schemas"}}
        TrainConfig.from_dict({**base, "model": {"propagation": "anchored"},
                               "ablation": {"coupling": "symmetric"}})
        bad = (
            {"model": {"arm": "hub_A"}},                                     # arm needs anchored
            {"model": {"propagation": "anchored"}},                          # default gates
            {"model": {"propagation": "anchored", "arm": "nope"}, "ablation": {"coupling": "symmetric"}},
            {"model": {"propagation": "anchored", "solver_max_iterations": 0},
             "ablation": {"coupling": "symmetric"}},
            {"model": {"propagation": "anchored", "solver_tolerance": 2.0},
             "ablation": {"coupling": "symmetric"}},
            {"model": {"propagation": "anchored", "tau_init": 0.0}, "ablation": {"coupling": "symmetric"}},
            {"model": {"propagation": "anchored"},
             "ablation": {"coupling": "symmetric", "holonomy_edges": True}},
            {"model": {"propagation": "anchored", "path_weight": "sigmoid_gamma"},
             "ablation": {"coupling": "symmetric"}},
        )
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                TrainConfig.from_dict({**base, **raw})
        for key in ("model.solver_tolerance", "model.solver_max_iterations", "model.tau_init",
                    "model.arm", "model.propagation", "model.solver_warmup_epochs"):
            with self.subTest(search=key), self.assertRaises(ValueError):
                reject_never_tuned_settings(base, [key])
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({**base, "model": {"arm": "covering_only"}}, [])
        # the legacy propagation is the default and builds no tau
        legacy = TrainConfig.from_dict({**base, "model": {"d": 8, "d_r": 4}})
        model = H2SNN(legacy, ModelDims.from_prepared_data(_random_pap_data()))
        self.assertEqual(legacy.model.propagation, "chebyshev")
        self.assertIsNone(model.theta_tau)
        self.assertIsNotNone(model.filter_bank)


if __name__ == "__main__":
    unittest.main()
