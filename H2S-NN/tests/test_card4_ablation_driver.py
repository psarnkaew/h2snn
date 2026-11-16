"""Card 4 K6 (`docs/H2S-NN_card4_ablation_prep_card.md`, amended 2026-11-16): the ablation driver
(`scripts/ablation/`) and the arm-tuning mode of `src/configs/config.py`. Synthetic data only.

- every emitted configuration (a) differs from `full`'s in exactly the arm's keys (one test, all arms);
- pairing: `full` and `uniform` at one seed get the same WISP draws (evaluation draw and an epoch's
  training draw, `torch.equal`); full batch has no batch order;
- (i) a main-table tuning base config with any arm key is refused (one test over all arm keys);
- (ii) arm-tuning mode with another arm's key is refused;
- (iii) arm-tuning mode builds for each of the five arms (the validation, the configuration and the
  launcher's per-trial merge).
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ablation"))
sys.path.insert(0, str(ROOT / "scripts"))
FULL = ROOT / "configs" / "experiments" / "acm_h2snn_v3b.yaml"


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class ConfigurationsATests(unittest.TestCase):
    def test_each_arm_differs_from_full_in_exactly_its_keys(self) -> None:
        import arms

        full = arms.flatten(yaml.safe_load(FULL.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as tmp:
            paths = arms.emit_configurations_a(FULL, "acm", Path(tmp))
            self.assertNotIn("literature_row", paths)
            self.assertIn("appendix_full", paths)
            self.assertIn("appendix_endpoints", paths)
            for arm, path in paths.items():
                emitted = arms.flatten(yaml.safe_load(path.read_text(encoding="utf-8")))
                differing = {k for k in set(full) | set(emitted)
                             if k != "exp_name" and full.get(k, "<absent>") != emitted.get(k, "<absent>")}
                expected = set(arms.flatten(arms.ARM_CHANGES[arm]))
                self.assertEqual(differing, expected, arm)


class PairingTests(unittest.TestCase):
    def test_full_and_uniform_get_the_same_draws_at_one_seed(self) -> None:
        from src.configs.config import TrainConfig
        from src.data.dataset import build_dataset, build_node_classification_paths
        from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

        small = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                 "classifier_hidden_dim": 16}
        draws = {}
        for arm, ablation in (("full", {}), ("uniform", {"path_weight_input": "uniform"})):
            cfg = TrainConfig.from_dict({"device": "cpu", "seed": 3,
                                         "data": {"dataset_name": "acm", "wisp_max_length": 3,
                                                  "wisp_walks_per_anchor": 5},
                                         "model": small, "ablation": ablation})
            with patched_acm_loader(acm_like_hetero(seed=9)):
                data = build_dataset(cfg)
            training = build_node_classification_paths(
                config=cfg, fact_layer=data.fact_layer, node_slices=data.node_slices,
                node_type=data.node_type, node_type_names=data.node_type_names,
                edge_index=data.edge_index, edge_type=data.edge_type,
                relation_tuples=data.relation_tuples, seed=cfg.seed + 1)   # epoch 1's draw
            draws[arm] = (data.face_groups, training)
        for which in (0, 1):
            a, b = draws["full"][which], draws["uniform"][which]
            self.assertEqual(len(a), len(b))
            for ga, gb in zip(a, b):
                for name in ("face_node_index", "face_edge_index", "face_edge_sign", "path_weight"):
                    self.assertTrue(torch.equal(getattr(ga, name), getattr(gb, name)), name)


class ArmTuningModeTests(unittest.TestCase):
    def _base(self) -> dict:
        return yaml.safe_load(FULL.read_text(encoding="utf-8"))

    def test_main_table_refuses_every_arm_key(self) -> None:
        from src.configs.config import ARM_TUNING_KEYS, V3B_ARMS, reject_never_tuned_settings

        changes = ([("model", "arm", a) for a in V3B_ARMS if a != "full"]
                   + [("ablation", "path_weight_input", v) for v in ("endpoints", "schema", "uniform")]
                   + [("ablation", "fact_maps_identity", True)])
        self.assertTrue({(a, b, str(c)) for a, b, c in ARM_TUNING_KEYS.values()}
                        <= {(a, b, str(c)) for a, b, c in changes})
        for section, key, value in changes:
            raw = self._base()
            raw[section] = {**raw.get(section, {}), key: value}
            with self.assertRaises(ValueError, msg=f"{section}.{key}: {value}"):
                reject_never_tuned_settings(raw)
        reject_never_tuned_settings(self._base())

    def test_arm_mode_refuses_another_arms_key(self) -> None:
        from src.configs.config import ARM_TUNING_KEYS, reject_never_tuned_settings

        for arm in ARM_TUNING_KEYS:
            for other, (section, key, value) in ARM_TUNING_KEYS.items():
                if other == arm:
                    continue
                raw = self._base()
                raw["tuning"] = {"arm": arm}
                raw[section] = {**raw.get(section, {}),
                                key: True if value == "True" else value}
                with self.assertRaises(ValueError, msg=f"tuning.arm {arm} with {other}'s key"):
                    reject_never_tuned_settings(raw)
            raw = self._base()                      # the mode without its own key
            raw["tuning"] = {"arm": arm}
            with self.assertRaises(ValueError):
                reject_never_tuned_settings(raw)
        raw = self._base()
        raw["tuning"] = {"arm": "schema"}           # not one of the five tuned arms
        with self.assertRaises(ValueError):
            reject_never_tuned_settings(raw)

    def test_arm_mode_builds_for_each_of_the_five_arms(self) -> None:
        import arms
        from run_h2snn_wandb_tuning import apply_sweep_overrides
        from src.configs.config import ARM_TUNING_KEYS, TrainConfig, reject_never_tuned_settings

        self.assertEqual(set(ARM_TUNING_KEYS), set(arms.TUNED_ARMS))
        for arm in arms.TUNED_ARMS:
            raw = arms.apply_change(self._base(), arm)
            raw["tuning"] = {"arm": arm}
            reject_never_tuned_settings(raw)
            cfg = TrainConfig.from_dict(copy.deepcopy(raw))
            self.assertEqual(cfg.tuning.arm, arm)
            trial = apply_sweep_overrides(copy.deepcopy(raw), {"d": 16, "lr": 1e-3})
            self.assertEqual(TrainConfig.from_dict(trial).model.d, 16)


if __name__ == "__main__":
    unittest.main()
