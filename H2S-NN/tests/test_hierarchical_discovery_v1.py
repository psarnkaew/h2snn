from __future__ import annotations

import unittest
from unittest.mock import patch


class StageIProtocolTests(unittest.TestCase):
    def test_coverage_without_task_gain_is_rejected(self) -> None:
        import torch
        from src.data.stage1_selector import make_selector_split, select_schema_features

        labels = torch.tensor([0, 0, 1, 1])
        split = make_selector_split(labels, "single_label", 0.5, 7, 1)
        result = select_schema_features(
            candidate_identities=("coverage_only",),
            feature_blocks=(torch.zeros(4, 1),),
            coverage_sets=(frozenset({0, 1, 2, 3}),),
            incidence_masses=(4.0,),
            labels=labels,
            split=split,
            task="single_label",
            max_schemas=1,
            incidence_budget=4.0,
            stop_tolerance=0.0,
            tie_tolerance=1e-8,
            probe_l2=1e-3,
            probe_max_iter=5,
            split_seed=7,
            sample_seed=11,
            candidate_identity_count=1,
            coverage_denominator=4,
            sampling_attempts=1,
            sampling_successes=1,
            sampling_unique_paths=1,
        )
        self.assertEqual(result.selected_indices, ())
        self.assertEqual(result.report.stop_reason, "nonpositive_task_gain")

    def test_task_informative_schema_is_selected_under_incidence_budget(self) -> None:
        import torch
        from src.data.stage1_selector import make_selector_split, select_schema_features

        labels = torch.tensor([0, 0, 1, 1])
        feature = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
        split = make_selector_split(labels, "single_label", 0.5, 7, 1)
        result = select_schema_features(
            candidate_identities=("task_aligned",),
            feature_blocks=(feature,),
            coverage_sets=(frozenset({0, 1}),),
            incidence_masses=(4.0,),
            labels=labels,
            split=split,
            task="single_label",
            max_schemas=1,
            incidence_budget=4.0,
            stop_tolerance=0.0,
            tie_tolerance=1e-8,
            probe_l2=1e-3,
            probe_max_iter=10,
            split_seed=7,
            sample_seed=11,
            candidate_identity_count=1,
            coverage_denominator=4,
            sampling_attempts=1,
            sampling_successes=1,
            sampling_unique_paths=1,
        )
        self.assertEqual(result.selected_indices, (0,))
        self.assertGreater(result.report.steps[0].task_gain, 0.0)


class ProtocolCompatibilityTests(unittest.TestCase):
    def test_removed_config_keys_are_rejected(self) -> None:
        from src.configs.config import TrainConfig

        removed = (
            "objective_version",
            "selector_coverage_weight",
            "schema_gate_type",
            "instance_gate_type",
            "partial_schema_probability",
            "schema_active_threshold",
            "lambda_schema_sparse",
            "lambda_instance_sparse",
            "lambda_coverage",
            "lambda_incidence_mass",
        )
        for key in removed:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "legacy"):
                    TrainConfig.from_dict({"discovery": {key: 0.0}})

    def test_removed_model_gate_flags_are_rejected(self) -> None:
        from src.configs.config import TrainConfig

        for key in (
            "ablate_uniform_gates",
            "ablate_schema_gate_only",
            "ablate_instance_gate_only",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "legacy"):
                    TrainConfig.from_dict({"model": {key: True}})

    def test_checkpoint_rejects_old_protocol_before_model_mutation(self) -> None:
        import torch
        from pathlib import Path
        from src.utils.checkpointer import Checkpointer

        model = torch.nn.Linear(2, 1)
        with patch("src.utils.checkpointer.Path.mkdir"):
            checkpointer = Checkpointer(Path("."), metric="loss", mode="min")
        with patch(
            "src.utils.checkpointer.torch.load",
            return_value={"protocol_version": "hierarchical_v4"},
        ):
            with self.assertRaisesRegex(ValueError, "protocol"):
                checkpointer.load(model)


class AMPSSAblationProtocolTests(unittest.TestCase):
    def test_five_variants_and_frozen_stage1_guard(self) -> None:
        from src.utils.ampss_ablation_protocol import (
            AMPSS_ABLATIONS,
            assert_frozen_stage1_manifests,
        )

        self.assertEqual(
            [item.name for item in AMPSS_ABLATIONS],
            [
                "no_faces",
                "manual_schemas_uniform_weights",
                "auto_pool_uniform_weights",
                "auto_pool_schema_agnostic_weights",
                "full_schema_conditioned_weights",
            ],
        )
        manifests = [
            {
                "ablation_index": index,
                "protocol_version": "schema_conditioned_v1",
                "selector_unit_hash": "units",
                "selector_support_hash": "support",
                "final_materialization_hash": "faces",
                "selected_identities": ["schema"],
            }
            for index in range(3, 6)
        ]
        assert_frozen_stage1_manifests(manifests)
        manifests[-1]["final_materialization_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "frozen Stage-I"):
            assert_frozen_stage1_manifests(manifests)


if __name__ == "__main__":
    unittest.main()
