from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class WandbAndRunReportingTests(unittest.TestCase):
    def test_wandb_disabled_preserves_local_metrics_jsonl(self) -> None:
        from src.utils.logger import ExperimentLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(Path(tmpdir) / "run")
            try:
                logger.log(
                    0,
                    {"loss": 1.0, "macro_f1": 0.1},
                    {"loss": 0.8, "macro_f1": 0.2},
                    {"loss": 0.7, "macro_f1": 0.3},
                )
            finally:
                logger.close()

            metrics_path = Path(tmpdir) / "run" / "metrics.jsonl"
            self.assertTrue(metrics_path.exists())
            record = json.loads(metrics_path.read_text(encoding="utf-8").strip())

        self.assertEqual(record["epoch"], 0)
        self.assertAlmostEqual(record["val"]["macro_f1"], 0.2)

    def test_wandb_enabled_logs_flattened_metrics_and_finishes(self) -> None:
        from src.utils.logger import ExperimentLogger

        events: dict[str, object] = {"logs": [], "finished": False}
        fake_run = SimpleNamespace(summary={})

        def fake_init(**kwargs):
            events["init"] = kwargs
            return fake_run

        def fake_log(payload, step=None):
            events["logs"].append((payload, step))  # type: ignore[union-attr]

        def fake_finish():
            events["finished"] = True

        fake_wandb = SimpleNamespace(init=fake_init, log=fake_log, finish=fake_finish)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                logger = ExperimentLogger(
                    Path(tmpdir) / "run",
                    use_wandb=True,
                    wandb_project="h2snn-test",
                    wandb_mode="offline",
                    wandb_group="group-a",
                    wandb_tags=("node_classification",),
                    wandb_name="run-a",
                    config={"seed": 42},
                )
                logger.log(
                    3,
                    {"loss": 1.0},
                    {"macro_f1": 0.5},
                    {"acc": 0.75},
                )
                logger.log_summary({"test_acc": 0.75, "seed": 42}, prefix="best")
                logger.close()

        self.assertEqual(events["init"]["project"], "h2snn-test")  # type: ignore[index]
        self.assertEqual(events["init"]["mode"], "offline")  # type: ignore[index]
        self.assertEqual(events["logs"][0][1], 3)  # type: ignore[index]
        self.assertAlmostEqual(events["logs"][0][0]["train/loss"], 1.0)  # type: ignore[index]
        self.assertAlmostEqual(events["logs"][0][0]["val/macro_f1"], 0.5)  # type: ignore[index]
        self.assertAlmostEqual(fake_run.summary["best/test_acc"], 0.75)
        self.assertTrue(events["finished"])

    def test_run_summary_uses_task_metrics_and_mean_std(self) -> None:
        from src.configs.config import TrainConfig
        from src.utils.run_reports import (
            default_monitor_for_task,
            report_metrics_for_task,
            write_summary_files,
        )

        cfg = TrainConfig.from_dict(
            {
                "data": {"dataset_name": "lastfm"},
                "output_dir": "experiments",
                "exp_name": "summary_test",
            }
        )
        records = [
            {"run_index": 0, "seed": 42, "test_auroc": 0.6, "test_aupr": 0.5, "test_mrr": 0.4},
            {"run_index": 1, "seed": 43, "test_auroc": 0.8, "test_aupr": 0.7, "test_mrr": 0.6},
            {"run_index": 2, "seed": 44, "test_auroc": 1.0, "test_aupr": 0.9, "test_mrr": 0.8},
            {"run_index": 3, "seed": 45, "test_auroc": 0.4, "test_aupr": 0.3, "test_mrr": 0.2},
            {"run_index": 4, "seed": 46, "test_auroc": 0.2, "test_aupr": 0.1, "test_mrr": 0.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = write_summary_files(
                Path(tmpdir) / "summary",
                records=records,
                metric_names=report_metrics_for_task(cfg.task_type),
                config=cfg,
            )
            summary_path = Path(tmpdir) / "summary" / "runs_summary.json"
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(default_monitor_for_task(cfg.task_type), "auroc")
        self.assertEqual(payload["summary"]["num_runs"], 5)
        self.assertAlmostEqual(loaded["summary"]["metrics"]["test_auroc"]["mean"], 0.6)
        self.assertGreater(loaded["summary"]["metrics"]["test_auroc"]["std"], 0.0)

    def test_wandb_sweep_config_and_tuning_selection(self) -> None:
        from scripts.run_h2snn_wandb_tuning import (
            apply_sweep_overrides,
            build_sweep_config,
            summarize_tuned_seed,
            validate_d_values,
            validate_trial_params,
        )

        sweep = build_sweep_config(
            task_type="graph_classification",
            metric="auroc",
            epochs=200,
            dataset_name="mutag",
        )

        # spec v1.15 §6 (Part M, M10): plain Bayes, no Hyperband
        self.assertEqual(sweep["method"], "bayes")
        self.assertEqual(sweep["metric"]["name"], "best/auroc")
        self.assertNotIn("early_terminate", sweep)
        self.assertEqual(sweep["parameters"]["d"]["values"], [16, 32, 64])
        # eta = sigmoid(theta), gamma = softplus(theta) are learned; their initial values
        # stay at the config defaults (0.5, 1.0) and are not searched.
        for legacy_parameter in (
            "eta_init",
            "gamma_init",
            "eta_0",
            "eta_1",
            "pooling_type",
            "sheafcov_feat_dim",
            "discovery_warmup_epochs",
            "discovery_operator_anneal_epochs",
            "learnable_eta",
            "use_delta0_init",
            "orthogonal_map",
            "scheduler",
            "optim_name",
            "grad_clip",
            "face_sample_k",
            "num_label_prop_steps",
            "closure_scale",
            "lambda_schema_sparse",
            "lambda_instance_sparse",
            "lambda_coverage",
            "lambda_incidence_mass",
            "partial_schema_probability",
        ):
            self.assertNotIn(legacy_parameter, sweep["parameters"])
        node_sweep = build_sweep_config(
            task_type="node_classification",
            metric="macro_f1",
            epochs=30,
            dataset_name="acm",
        )
        self.assertEqual(node_sweep["parameters"]["d"]["values"], [16, 32, 64])
        self.assertIn("encoder_dropout", node_sweep["parameters"])
        self.assertIn("classifier_dropout", node_sweep["parameters"])
        featureless_link_sweep = build_sweep_config(
            task_type="link_prediction",
            metric="auroc",
            epochs=30,
            decoder_type="dot",
            dataset_name="lastfm",
        )
        self.assertNotIn("encoder_dropout", featureless_link_sweep["parameters"])
        self.assertNotIn("classifier_dropout", featureless_link_sweep["parameters"])
        decoder_ablation = build_sweep_config(
            task_type="link_prediction",
            metric="auroc",
            epochs=30,
            decoder_type="concat_mlp",
            dataset_name="pubmed",
            include_decoder_type=True,
        )
        self.assertIn("decoder_type", decoder_ablation["parameters"])
        self.assertNotIn("classifier_dropout", decoder_ablation["parameters"])
        custom_sweep = build_sweep_config(
            task_type="node_classification",
            metric="macro_f1",
            epochs=30,
            d_values=[8, 16, 16],
        )
        self.assertEqual(custom_sweep["parameters"]["d"]["values"], [8, 16])
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_d_values([16, 0])
        with self.assertRaisesRegex(ValueError, "matrix_exp"):
            validate_trial_params({"orthogonal_map": "matrix_exp", "d": 16})
        with self.assertRaisesRegex(ValueError, "decoder_type"):
            validate_trial_params(
                {"decoder_type": "dot"},
                allowed_parameters=set(node_sweep["parameters"]),
            )
        self.assertNotIn("layer_dropout", sweep["parameters"])  # not a method component
        self.assertNotIn("K", sweep["parameters"])
        self.assertEqual(
            sweep["parameters"]["lr"],
            {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        )
        self.assertEqual(
            sweep["parameters"]["weight_decay"]["values"],
            [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.8],
        )

        raw = {
            "data": {"dataset_name": "mutag"},
            "model": {
                "d": 16,
                "d_r": 7,
                "edge_mlp_hidden_dim": 23,
                "classifier_hidden_dim": 29,
            },
            "optim": {},
        }
        tuned = apply_sweep_overrides(raw, {"d": 32, "graph_pooling": "max"})
        self.assertEqual(tuned["model"]["graph_pooling"], "max")
        with self.assertRaises(ValueError):  # layer_dropout is no longer a tuning parameter
            apply_sweep_overrides(raw, {"d": 32, "layer_dropout": 0.2})
        self.assertEqual(tuned["model"]["d_r"], 7)
        self.assertEqual(tuned["model"]["edge_mlp_hidden_dim"], 23)
        self.assertEqual(tuned["model"]["classifier_hidden_dim"], 29)

        records = [
            {"trial_name": "a", "val_auroc": 0.6, "test_auroc": 0.5},
            {"trial_name": "b", "val_auroc": 0.9, "test_auroc": 0.4},
            {"trial_name": "c", "val_auroc": 0.7, "test_auroc": 0.8},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            best = summarize_tuned_seed(
                records,
                monitor="auroc",
                out_path=Path(tmpdir) / "best_tuned_run0.json",
            )
            saved = json.loads((Path(tmpdir) / "best_tuned_run0.json").read_text(encoding="utf-8"))

        self.assertEqual(best["trial_name"], "b")
        self.assertEqual(saved["trial_name"], "b")

    def test_repeated_cv_unit_mapping_config_and_population_summary(self) -> None:
        from scripts.run_graph_classification_repeated_cv_wandb_tuning import (
            build_unit_raw_config,
            summarize_repeated_cv_records,
            unit_coordinates,
            unit_name,
            write_repeated_cv_summary,
        )
        from src.configs.config import TrainConfig

        self.assertEqual(unit_coordinates(99, 10), (9, 9))
        self.assertEqual(unit_name(2, 3), "rep02_fold03")
        self.assertEqual(
            len({unit_coordinates(index, 10) for index in range(100)}),
            100,
        )
        self.assertEqual(
            len({unit_name(*unit_coordinates(index, 10)) for index in range(100)}),
            100,
        )
        raw = build_unit_raw_config(
            {"data": {"dataset_name": "mutag"}, "seed": 42},
            unit_index=23,
            num_repetitions=10,
            num_folds=10,
            base_seed=100,
            split_seed=7,
        )
        self.assertEqual(raw["seed"], 123)
        self.assertEqual(raw["data"]["split_protocol"], "repeated_kfold")
        self.assertEqual(raw["data"]["repetition_index"], 2)
        self.assertEqual(raw["data"]["fold_index"], 3)
        self.assertEqual(raw["data"]["split_seed"], 7)

        records = []
        for unit_index, value in enumerate((1.0, 3.0, 5.0, 7.0)):
            repetition_index, fold_index = unit_coordinates(unit_index, 2)
            records.append(
                {
                    "unit_index": unit_index,
                    "repetition_index": repetition_index,
                    "fold_index": fold_index,
                    "test_auroc": value,
                    "test_macro_f1": value,
                    "test_micro_f1": value,
                    "test_acc": value,
                }
            )
        summary = summarize_repeated_cv_records(
            records,
            num_repetitions=2,
            num_folds=2,
            metric_names=("test_acc",),
        )
        metric = summary["metrics"]["test_acc"]
        self.assertAlmostEqual(metric["mean"], 4.0)
        self.assertAlmostEqual(metric["std_across_repetition_means"], 2.0)
        self.assertAlmostEqual(metric["std_across_all_folds"], 5.0 ** 0.5)
        self.assertEqual(summary["completed_units"], [0, 1, 2, 3])
        self.assertEqual(summary["missing_units"], [])
        partial = summarize_repeated_cv_records(
            records[:3],
            num_repetitions=2,
            num_folds=2,
            metric_names=("test_acc",),
        )
        self.assertEqual(partial["completed_units"], [0, 1, 2])
        self.assertEqual(partial["missing_units"], [3])
        self.assertEqual(
            partial["metrics"]["test_acc"]["num_complete_repetitions"],
            1,
        )
        tmp_parent = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmpdir:
            payload = write_repeated_cv_summary(
                Path(tmpdir),
                records=records,
                config=TrainConfig.from_dict({"data": {"dataset_name": "mutag"}}),
                num_repetitions=2,
                num_folds=2,
            )
            self.assertTrue((Path(tmpdir) / "repeated_cv_tuning_summary.json").exists())
            self.assertTrue((Path(tmpdir) / "repeated_cv_tuning_summary.csv").exists())
            self.assertAlmostEqual(payload["summary"]["metrics"]["test_acc"]["mean"], 4.0)

    def test_wandb_tuning_resume_requires_matching_fingerprints(self) -> None:
        from scripts.run_h2snn_wandb_tuning import (
            TUNING_STATE_VERSION,
            canonical_fingerprint,
            save_tuning_state,
            validate_resume_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state" / "tuning_state_run0.json"
            base_fingerprint = canonical_fingerprint({"model": {"d": 16}})
            sweep_fingerprint = canonical_fingerprint({"parameters": {"d": [16, 32]}})
            save_tuning_state(
                state_path,
                {
                    "state_version": TUNING_STATE_VERSION,
                    "base_config_fingerprint": base_fingerprint,
                    "sweep_config_fingerprint": sweep_fingerprint,
                },
            )
            state = validate_resume_state(
                json.loads(state_path.read_text(encoding="utf-8")),
                state_path=state_path,
                base_config_fingerprint=base_fingerprint,
                sweep_config_fingerprint=sweep_fingerprint,
            )
            self.assertEqual(state["state_version"], TUNING_STATE_VERSION)
            with self.assertRaisesRegex(ValueError, "base configuration changed"):
                validate_resume_state(
                    state,
                    state_path=state_path,
                    base_config_fingerprint="changed",
                    sweep_config_fingerprint=sweep_fingerprint,
                )
            with self.assertRaisesRegex(ValueError, "search space changed"):
                validate_resume_state(
                    state,
                    state_path=state_path,
                    base_config_fingerprint=base_fingerprint,
                    sweep_config_fingerprint="changed",
                )

            legacy = {"sweep_id": "old"}
            with self.assertRaisesRegex(ValueError, "legacy tuning state"):
                validate_resume_state(
                    legacy,
                    state_path=state_path,
                    base_config_fingerprint=base_fingerprint,
                    sweep_config_fingerprint=sweep_fingerprint,
                )
            with self.assertRaisesRegex(ValueError, "no tuning state"):
                validate_resume_state(
                    None,
                    state_path=state_path,
                    base_config_fingerprint=base_fingerprint,
                    sweep_config_fingerprint=sweep_fingerprint,
                )

    def test_campaign_resume_selects_compatible_continuation_or_new_state(self) -> None:
        from scripts.run_h2snn_wandb_tuning import (
            TUNING_STATE_VERSION,
            canonical_fingerprint,
            resolve_campaign_state,
            save_tuning_state,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "tuning_state_run0.json"
            base_fingerprint = canonical_fingerprint({"model": {"d": 16}})
            sweep_fingerprint = canonical_fingerprint({"parameters": {"d": [16, 32]}})
            save_tuning_state(
                state_path,
                {
                    "state_version": TUNING_STATE_VERSION,
                    "base_config_fingerprint": "old-base",
                    "sweep_config_fingerprint": sweep_fingerprint,
                    "sweep_id": "old-sweep",
                },
            )

            continuation_path, state, action = resolve_campaign_state(
                state_path,
                base_config_fingerprint=base_fingerprint,
                sweep_config_fingerprint=sweep_fingerprint,
            )
            self.assertEqual(continuation_path.name, "tuning_state_run0_continuation1.json")
            self.assertIsNone(state)
            self.assertEqual(action, "continuation")

            save_tuning_state(
                continuation_path,
                {
                    "state_version": TUNING_STATE_VERSION,
                    "base_config_fingerprint": base_fingerprint,
                    "sweep_config_fingerprint": sweep_fingerprint,
                    "sweep_id": "continuation-sweep",
                },
            )
            resumed_path, resumed_state, action = resolve_campaign_state(
                state_path,
                base_config_fingerprint=base_fingerprint,
                sweep_config_fingerprint=sweep_fingerprint,
            )
            self.assertEqual(resumed_path, continuation_path)
            self.assertEqual(resumed_state["sweep_id"], "continuation-sweep")
            self.assertEqual(action, "resume")

            missing_path, missing_state, action = resolve_campaign_state(
                Path(tmpdir) / "tuning_state_run1.json",
                base_config_fingerprint=base_fingerprint,
                sweep_config_fingerprint=sweep_fingerprint,
            )
            self.assertEqual(missing_path.name, "tuning_state_run1.json")
            self.assertIsNone(missing_state)
            self.assertEqual(action, "new")

    def test_fresh_validation_refuses_to_overwrite_existing_state(self) -> None:
        from src.configs.config import TrainConfig
        from scripts.run_h2snn_wandb_tuning import (
            save_tuning_state,
            validate_tuning_setup,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw = {
                "data": {"dataset_name": "acm"},
                "output_dir": tmpdir,
                "exp_name": "fresh_collision",
                "epochs": 5,
            }
            cfg = TrainConfig.from_dict(raw)
            state_dir = Path(tmpdir) / "state"
            save_tuning_state(state_dir / "tuning_state_run0.json", {"old": True})
            args = SimpleNamespace(
                resume=False,
                d_values=[16],
                include_pooling_type=False,
                include_decoder_type=False,
                include_discovery=False,
                hyperband_min_iter=2,
                hyperband_eta=3,
            )
            with self.assertRaisesRegex(ValueError, "existing state"):
                validate_tuning_setup(
                    raw,
                    args=args,
                    base_cfg=cfg,
                    base_exp_name=cfg.exp_name,
                    monitor="macro_f1",
                    state_dir=state_dir,
                    num_runs=1,
                )

    def test_failed_worker_does_not_prevent_next_isolated_attempt(self) -> None:
        from src.configs.config import TrainConfig
        from scripts.run_h2snn_wandb_tuning import (
            load_tuning_state,
            run_seed_sweep,
            save_tuning_state,
        )

        class FakeWandb:
            def sweep(self, **kwargs):
                return "new-safe-sweep"

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig.from_dict(
                {
                    "data": {"dataset_name": "acm"},
                    "output_dir": tmpdir,
                    "exp_name": "isolated_tuning",
                    "epochs": 5,
                }
            )
            state_path = Path(tmpdir) / "state" / "tuning_state_run0.json"
            args = SimpleNamespace(
                resume=False,
                sweep_id="",
                n_trials=2,
                d_values=[16],
                include_pooling_type=False,
                include_decoder_type=False,
                include_discovery=False,
                hyperband_min_iter=2,
                hyperband_eta=3,
                wandb_project="h2snn-test",
                wandb_entity="",
                wandb_group=None,
                wandb_tags=None,
                wandb_mode="offline",
            )
            calls: list[list[str]] = []

            def fake_worker(command, check):
                calls.append(command)
                if len(calls) == 1:
                    return SimpleNamespace(returncode=7)
                context = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                child_state_path = Path(context["state_path"])
                child_state = load_tuning_state(child_state_path)
                child_state["records"] = [
                    {
                        "trial_name": "fresh-worker",
                        "val_macro_f1": 0.7,
                        "test_macro_f1": 0.6,
                    }
                ]
                save_tuning_state(child_state_path, child_state)
                save_tuning_state(
                    Path(context["status_path"]),
                    {"ok": True, "worker_pid": 12345},
                )
                return SimpleNamespace(returncode=0)

            with patch.dict(sys.modules, {"wandb": FakeWandb()}):
                with patch(
                    # Worker spawning moved into the shared engine; run_seed_sweep now
                    # delegates there. train_trial is still patched on the script module
                    # because run_agent_worker resolves it at call time.
                    "src.utils.wandb_campaign.subprocess.run",
                    side_effect=fake_worker,
                ):
                    best = run_seed_sweep(
                        {"data": {"dataset_name": "acm"}},
                        args=args,
                        run_index=0,
                        num_runs=1,
                        seed=42,
                        base_exp_name="isolated_tuning",
                        base_cfg=cfg,
                        monitor="macro_f1",
                        state_path=state_path,
                    )

            state = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 2)
        self.assertEqual(state["attempted_trials"], 2)
        self.assertEqual(state["successful_trials"], 1)
        self.assertEqual(state["failed_trials"], 1)
        self.assertEqual(best["trial_name"], "fresh-worker")

    def test_agent_worker_runs_exactly_one_trial_and_records_pid(self) -> None:
        from scripts.run_h2snn_wandb_tuning import (
            TUNING_STATE_VERSION,
            canonical_fingerprint,
            load_tuning_state,
            run_agent_worker,
            save_tuning_state,
        )

        class FakeRun:
            id = "worker-run"
            name = "worker-trial"
            config = {"d": 16}
            summary: dict[str, object] = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeWandb:
            def __init__(self) -> None:
                self.agent_calls = 0

            def init(self, **kwargs):
                return FakeRun()

            def agent(self, sweep_id, function, count, project, entity):
                self.agent_calls += 1
                self.agent_args = (sweep_id, count, project, entity)
                function()

            def log(self, payload):
                self.logged = payload

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "tuning_state_run0.json"
            status_path = root / "worker_status.json"
            context_path = root / "worker_context.json"
            raw = {
                "data": {"dataset_name": "acm"},
                "model": {"orthogonal_map": "fasthpp", "d": 16},
                "output_dir": str(root),
                "exp_name": "worker_test",
                "epochs": 2,
            }
            base_fingerprint = canonical_fingerprint(raw)
            sweep_fingerprint = canonical_fingerprint({"safe": True})
            save_tuning_state(
                state_path,
                {
                    "state_version": TUNING_STATE_VERSION,
                    "base_config_fingerprint": base_fingerprint,
                    "sweep_config_fingerprint": sweep_fingerprint,
                    "records": [],
                },
            )
            save_tuning_state(
                context_path,
                {
                    "raw_base": raw,
                    "run_index": 0,
                    "num_runs": 1,
                    "seed": 42,
                    "base_exp_name": "worker_test",
                    "output_dir": str(root),
                    "monitor": "macro_f1",
                    "state_path": str(state_path),
                    "status_path": str(status_path),
                    "attempt_index": 0,
                    "sweep_id": "safe-sweep",
                    "wandb_project": "h2snn-test",
                    "wandb_entity": "",
                    "wandb_mode": "offline",
                    "wandb_group": "worker-group",
                    "wandb_tags": ["tuning"],
                    "allowed_parameters": ["d"],
                    "base_config_fingerprint": base_fingerprint,
                    "sweep_config_fingerprint": sweep_fingerprint,
                },
            )
            fake_wandb = FakeWandb()
            with patch.dict(sys.modules, {"wandb": fake_wandb}):
                with patch(
                    "scripts.run_h2snn_wandb_tuning.train_trial",
                    return_value={
                        "trial_name": "worker-trial",
                        "val_macro_f1": 0.8,
                        "test_macro_f1": 0.7,
                    },
                ):
                    run_agent_worker(context_path)

            state = load_tuning_state(state_path)
            status = load_tuning_state(status_path)

        self.assertEqual(fake_wandb.agent_calls, 1)
        self.assertEqual(fake_wandb.agent_args, ("safe-sweep", 1, "h2snn-test", None))
        self.assertEqual(len(state["records"]), 1)
        self.assertTrue(status["ok"])
        self.assertIsInstance(status["worker_pid"], int)

    def test_notebooks_reference_repeated_run_and_wandb_tuning_scripts(self) -> None:
        notebook_dir = Path(__file__).resolve().parent.parent / "notebooks"
        for name in (
            "h2snn_node_classification_colab.ipynb",
            "h2snn_link_prediction_colab.ipynb",
            "h2snn_graph_classification_colab.ipynb",
        ):
            payload = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in payload.get("cells", [])
            )
            self.assertTrue(
                all(cell.get("execution_count") is None for cell in payload.get("cells", []))
            )
            self.assertTrue(
                all(not cell.get("outputs") for cell in payload.get("cells", []))
            )
            self.assertIn("scripts/run_h2snn_runs.py", source)
            self.assertIn("scripts/run_h2snn_wandb_tuning.py", source)
            self.assertIn("runs_summary.json", source)
            self.assertIn("NUM_RUNS = 5", source)
        for name in (
            "h2snn_node_classification_wandb_tuning_colab.ipynb",
            "h2snn_link_prediction_wandb_tuning_colab.ipynb",
            "h2snn_graph_classification_wandb_tuning_colab.ipynb",
            "h2snn_wandb_paper_tuning_colab.ipynb",
        ):
            payload = json.loads((notebook_dir / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", [])) for cell in payload.get("cells", [])
            )
            self.assertTrue(
                all(cell.get("execution_count") is None for cell in payload.get("cells", []))
            )
            self.assertTrue(all(not cell.get("outputs") for cell in payload.get("cells", [])))
            self.assertIn("scripts/run_h2snn_runs.py", source)
            self.assertIn("scripts/run_h2snn_wandb_tuning.py", source)
            self.assertIn("runs_summary.json", source)
            self.assertIn("NUM_RUNS = 5", source)
            self.assertIn("RESUME = False", source)
            self.assertIn("--resume", source)
            self.assertIn("--validate_only", source)
            self.assertIn("BASE_SEED", source)
            self.assertIn("RUN_START", source)
            self.assertIn("RUN_COUNT", source)
            self.assertIn("--base_seed", source)
            self.assertIn("--run_start", source)
            self.assertIn("--run_count", source)
            self.assertIn("--d_values", source)
            self.assertIn("requirements.txt", source)
            self.assertIn("WANDB_API_KEY", source)
            self.assertNotIn("fearless-sweep", source)
            self.assertNotIn("cuda_illegal_access_debug", source)
            self.assertIn("HYPERBAND_MIN_ITER", source)
            for removed_switch in (
                "--include_discovery",
                "--include_decoder_type",
                "--include_pooling_type",
                "--pooling_type",
                "--sheafcov_feat_dims",
                "_ampss.yaml",
            ):
                self.assertNotIn(removed_switch, source)

    def test_repeated_cv_notebook_is_clean_and_uses_fold_campaign(self) -> None:
        notebook_dir = Path(__file__).resolve().parent.parent / "notebooks"
        repeated_cv_name = "h2snn_graph_classification_repeated_cv_wandb_tuning_colab.ipynb"
        payload = json.loads((notebook_dir / repeated_cv_name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload.get("cells", [])
        )
        self.assertTrue(
            all(cell.get("execution_count") is None for cell in payload.get("cells", []))
        )
        self.assertTrue(all(not cell.get("outputs") for cell in payload.get("cells", [])))
        self.assertIn("scripts/run_graph_classification_repeated_cv_wandb_tuning.py", source)
        # Five repetitions x ten folds = the fifty runs the manuscript reports, and
        # the reading of the TUDataset paper's own sentence that its 'fifty runs'
        # half describes. Their code uses ten.
        self.assertIn("NUM_REPETITIONS = 5", source)
        self.assertIn("NUM_FOLDS = 10", source)
        self.assertIn("UNIT_START", source)
        self.assertIn("UNIT_COUNT", source)
        self.assertIn("--validate_only", source)
        self.assertIn("--resume", source)
        self.assertIn("WANDB_API_KEY", source)
        self.assertIn("repeated_cv_tuning_summary.json", source)


if __name__ == "__main__":
    unittest.main()
