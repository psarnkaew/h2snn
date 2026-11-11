"""Part M, M6: the stop conditions of spec v1.15 §6 (J10). `tau > 50` and a log-weight range of g
above 30 raise named exceptions (as T_max does), carrying the value and the epoch; trainers log
them and re-raise. Forced values trigger each; a normal forward does not."""

from __future__ import annotations

import math
import unittest

import torch

from src.configs.config import NEVER_TUNED_V3B_KEYS, TrainConfig, reject_never_tuned_settings
from src.models.anchored_solve import (LogWeightRangeStopError, StopConditionError,
                                       TauStopError)
from src.models.h2snn_model import H2SNN, ModelDims
from tests.test_factorized_maps import _random_pap_data
from tests.test_v3b_model import _config


def _model(data):
    torch.manual_seed(0)
    return H2SNN(_config("full"), ModelDims.from_prepared_data(data))


class StopConditionTests(unittest.TestCase):
    def test_normal_forward_does_not_stop(self) -> None:
        data = _random_pap_data()
        model = _model(data)
        model.train()
        model(data, epoch=0)
        self.assertLess(float(model.tau()), 50.0)
        self.assertLessEqual(model.last_path_stats["path_log_w_range"], 30.0)
        self.assertGreater(model.last_path_stats["path_log_w_range"], 0.0)

    def test_tau_above_50_stops(self) -> None:
        data = _random_pap_data()
        model = _model(data)
        with torch.no_grad():
            model.theta_tau.fill_(math.log(49.0))
        model(data, epoch=3)  # 49: no stop
        with torch.no_grad():
            model.theta_tau.fill_(math.log(51.0))
        with self.assertRaises(TauStopError) as caught:
            model(data, epoch=3)
        self.assertIsInstance(caught.exception, StopConditionError)
        self.assertAlmostEqual(caught.exception.value, 51.0, places=3)
        self.assertEqual((caught.exception.limit, caught.exception.epoch), (50.0, 3))
        self.assertIn("epoch 3", str(caught.exception))

    def test_log_weight_range_above_30_stops(self) -> None:
        """Forced log-weights: the range max - min over the cells of one forward is the value."""
        data = _random_pap_data()
        model = _model(data)
        original = model._cell_weights

        def forced(spread: float):
            def cell_weights(h0, geometry, groups):
                betas, weights, latents = original(h0, geometry, groups)
                flat = torch.cat(weights)
                log_w = torch.linspace(-spread / 2, spread / 2, flat.numel())
                parts = torch.split(log_w.exp(), [w.numel() for w in weights])
                return betas, tuple(parts), latents
            return cell_weights

        model._cell_weights = forced(29.0)
        model(data, epoch=1)
        self.assertAlmostEqual(model.last_path_stats["path_log_w_range"], 29.0, places=3)
        model._cell_weights = forced(31.0)
        model.eval()
        with self.assertRaises(LogWeightRangeStopError) as caught:
            model(data, epoch=-1)
        self.assertAlmostEqual(caught.exception.value, 31.0, places=3)
        self.assertEqual((caught.exception.limit, caught.exception.epoch), (30.0, -1))
        self.assertIn("evaluation forward", str(caught.exception))

    def test_limits_are_fixed_and_never_tuned(self) -> None:
        config = TrainConfig.from_dict({"data": {"dataset_name": "acm"}})
        self.assertEqual((config.model.tau_stop, config.model.log_weight_range_stop), (50.0, 30.0))
        for key in ("tau_stop", "log_weight_range_stop"):
            self.assertIn(key, NEVER_TUNED_V3B_KEYS)
            with self.assertRaises(ValueError):
                reject_never_tuned_settings({"data": {"dataset_name": "acm"}}, [f"model.{key}"])
            with self.assertRaises(ValueError):
                TrainConfig.from_dict({"data": {"dataset_name": "acm"}, "model": {key: 0.0}})


class TrainerLoggingTests(unittest.TestCase):
    def test_trainer_logs_and_reraises(self) -> None:
        from src.trainers.stop_conditions import logged_stop_conditions

        messages: list[str] = []

        class Logger:
            def info(self, fmt, *args):
                messages.append(fmt % args)

        with self.assertRaises(TauStopError):
            with logged_stop_conditions(Logger()):
                raise TauStopError(60.0, 50.0, 7)
        self.assertEqual(len(messages), 1)
        self.assertIn("tau: 60 > 50 at epoch 7", messages[0])


if __name__ == "__main__":
    unittest.main()
