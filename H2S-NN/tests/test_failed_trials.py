"""Part N, N4: the non-finite-g stop, the non-finite-loss stop of every method, and failed trials
in the tuners (spec v1.16 K5). A trial that raises a stop condition, T_max or a non-finite loss
is recorded as failed: no score reaches the Bayesian search, the record names the condition,
epoch and value, and the summaries count failed trials per method and dataset."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

from src.models.anchored_solve import (LogWeightRangeStopError, NonFiniteLogWeightError,
                                       NonFiniteLossError, PCGConvergenceError, StopConditionError,
                                       TauStopError)
from src.utils.wandb_campaign import (TUNING_STATE_VERSION, count_failures, failure_details,
                                      record_failed_attempt, run_campaign_worker,
                                      save_tuning_state)


class NonFiniteGTests(unittest.TestCase):
    def test_non_finite_g_raises_before_exp(self) -> None:
        from src.models.h2snn_model import H2SNN, ModelDims
        from tests.test_factorized_maps import _random_pap_data
        from tests.test_v3b_model import _config

        data = _random_pap_data()
        torch.manual_seed(0)
        model = H2SNN(_config("full"), ModelDims.from_prepared_data(data))
        model(data, epoch=2)  # finite g: no stop
        gate = model.path_gate

        class Broken(torch.nn.Module):
            def forward(self, latent):
                out = gate(latent)
                return out + torch.where(torch.arange(out.size(0)).view(-1, 1) == 0,
                                         torch.tensor(float("nan")), torch.tensor(0.0))

        model.path_gate = Broken()
        with self.assertRaises(NonFiniteLogWeightError) as caught:
            model(data, epoch=2)
        self.assertIsInstance(caught.exception, StopConditionError)
        self.assertEqual((caught.exception.condition, caught.exception.epoch),
                         ("non_finite_log_weight", 2))
        self.assertTrue(math.isnan(caught.exception.value))


def _pcg_error() -> PCGConvergenceError:
    exc = PCGConvergenceError("PCG reached T_max")
    exc.condition, exc.value, exc.limit, exc.epoch = "T_max", 100.0, 100.0, 4
    return exc


CONDITIONS = {
    "tau": lambda: TauStopError(51.0, 50.0, 3),
    "log_weight_range": lambda: LogWeightRangeStopError(31.0, 30.0, 1),
    "non_finite_log_weight": lambda: NonFiniteLogWeightError(float("inf"), 0),
    "non_finite_loss": lambda: NonFiniteLossError(float("nan"), 5),
    "T_max": _pcg_error,
}


def _fake_wandb(logged: list) -> types.ModuleType:
    """W&B stand-in: `agent` runs the objective once and swallows its exception, as W&B does."""
    module = types.ModuleType("wandb")

    class Run:
        config, name, id = {}, "trial", "run-id"
        summary: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    module.init = lambda **kwargs: Run()
    module.log = lambda payload, **kwargs: logged.append(payload)

    def agent(sweep_id, function, count, project, entity):
        try:
            function()
        except Exception:  # W&B marks the run failed and returns
            pass

    module.agent = agent
    return module


class FailedTrialTests(unittest.TestCase):
    def _run_worker(self, error_factory):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            state_path, status_path = tmp / "state.json", tmp / "status.json"
            save_tuning_state(state_path, {"state_version": TUNING_STATE_VERSION,
                                           "base_config_fingerprint": "b",
                                           "sweep_config_fingerprint": "s", "records": [],
                                           "failures": []})
            context = {"state_path": str(state_path), "status_path": str(status_path),
                       "base_config_fingerprint": "b", "sweep_config_fingerprint": "s",
                       "raw_base": {}, "allowed_parameters": [], "wandb_project": "p",
                       "wandb_mode": "disabled", "wandb_group": "g", "sweep_id": "x",
                       "run_index": 0, "num_runs": 1, "seed": 0, "base_exp_name": "e",
                       "output_dir": str(tmp), "monitor": "macro_f1", "attempt_index": 0}
            context_path = tmp / "context.json"
            context_path.write_text(json.dumps(context), encoding="utf-8")
            logged: list = []

            def trial(*args, **kwargs):
                raise error_factory()

            previous = sys.modules.get("wandb")
            sys.modules["wandb"] = _fake_wandb(logged)
            try:
                with self.assertRaises(RuntimeError):
                    run_campaign_worker(context_path, train_trial=trial, known_parameters=set())
            finally:
                if previous is None:
                    sys.modules.pop("wandb")
                else:
                    sys.modules["wandb"] = previous
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            return status, state, logged

    def test_each_condition_is_a_failed_trial_without_score(self) -> None:
        for condition, factory in CONDITIONS.items():
            with self.subTest(condition=condition):
                status, state, logged = self._run_worker(factory)
                self.assertEqual(logged, [])                    # no score sent to the search
                self.assertEqual(state["records"], [])          # no successful record
                self.assertFalse(status["ok"])
                expected = factory()
                failure = status["failure"]
                self.assertEqual(failure["condition"], condition)
                self.assertEqual(failure["epoch"], expected.epoch)
                if math.isnan(expected.value):
                    self.assertTrue(math.isnan(failure["value"]))
                else:
                    self.assertEqual(failure["value"], expected.value)
                # the parent's accounting of that attempt
                parent_state = {"failures": []}
                record_failed_attempt(parent_state, {"attempt_index": 0}, status)
                self.assertEqual(parent_state["failed_trials"], 1)
                self.assertEqual(parent_state["failed_by_condition"], {condition: 1})

    def test_other_errors_are_failed_trials_named_other(self) -> None:
        self.assertIsNone(failure_details(ValueError("boom")))
        self.assertEqual(count_failures([{"error": "boom"}, {"failure": {"condition": "tau"}}]),
                         {"other": 1, "tau": 1})

    def test_nan_loss_in_a_baseline_loop_fails_the_trial(self) -> None:
        from src.baselines.training import train_full_batch_node_classification

        model = torch.nn.Linear(2, 2)
        x = torch.randn(6, 2)

        def forward():
            return model(x) * float("nan")

        with self.assertRaises(NonFiniteLossError) as caught:
            train_full_batch_node_classification(
                model=model, forward=forward, labels=torch.tensor([0, 1, 0, 1, 0, 1]),
                train_idx=torch.tensor([0, 1]), val_idx=torch.tensor([2, 3]),
                test_idx=torch.tensor([4, 5]), optimizer=torch.optim.AdamW(model.parameters()),
                criterion=torch.nn.functional.cross_entropy, is_multilabel=False, epochs=3,
                patience=3)
        self.assertEqual(caught.exception.epoch, 0)
        self.assertEqual(failure_details(caught.exception)["condition"], "non_finite_loss")

    def test_campaign_summary_counts_failed_trials_per_cell(self) -> None:
        from src.baselines.campaign import summarize_cell

        records = [{"test_macro_f1": 0.1, "failed_trials": 2,
                    "failed_by_condition": {"tau": 1, "non_finite_loss": 1}},
                   {"test_macro_f1": 0.2, "failed_trials": 1, "failed_by_condition": {"tau": 1}}]
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize_cell(records, "node_classification", Path(tmp) / "s.json")
        self.assertEqual(summary["failed_trials"], 3)
        self.assertEqual(summary["failed_by_condition"], {"non_finite_loss": 1, "tau": 2})


if __name__ == "__main__":
    unittest.main()
