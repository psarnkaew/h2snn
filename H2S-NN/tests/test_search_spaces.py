"""Part M, M10: search spaces (spec v1.15 §6 Tuning, J2, J3).

H2S-NN's W&B sweep is plain Bayes (no Hyperband) over the method's components only: d {16, 32,
64}; lr log-uniform [1e-4, 1e-2]; weight_decay {0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.8};
encoder/classifier dropout uniform [0, 0.9] (encoder dropout absent without features); l_max
{2, 3, 4, 5}; r {8, 16, 32, 64}; GC pooling {mean, max, add, attention}; LP decoder_type only on
request (open). K, layer_dropout and every legacy key are refused. Tuning on seed 0; the best
configuration is then trained on seeds 0-4."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from src.configs.config import NEVER_TUNED_LEGACY_KEYS, TrainConfig, reject_never_tuned_settings

LR = {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2}
WD = {"values": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.8]}
DROP = {"distribution": "uniform", "min": 0.0, "max": 0.9}
COMMON = {"d": {"values": [16, 32, 64]}, "lr": LR, "weight_decay": WD,
          "wisp_max_length": {"values": [2, 3, 4, 5]},
          "wisp_walks_per_anchor": {"values": [8, 16, 32, 64]}}


def _sweep(task: str, dataset: str, **kwargs):
    from scripts.run_h2snn_wandb_tuning import build_sweep_config

    return build_sweep_config(task_type=task, metric="val_x", epochs=50, dataset_name=dataset,
                              **kwargs)


class SweepConfigTests(unittest.TestCase):
    def test_node_classification_space_is_the_table(self) -> None:
        sweep = _sweep("node_classification", "acm")
        self.assertEqual(sweep["method"], "bayes")
        self.assertNotIn("early_terminate", sweep)  # no Hyperband
        self.assertEqual(sweep["parameters"],
                         {**COMMON, "encoder_dropout": DROP, "classifier_dropout": DROP})

    def test_link_prediction_spaces(self) -> None:
        # LastFM: no features (no encoder dropout); dot decoder has no dropout module
        self.assertEqual(_sweep("link_prediction", "lastfm", decoder_type="dot")["parameters"],
                         COMMON)
        self.assertEqual(_sweep("link_prediction", "freebase_bb",
                                decoder_type="distmult")["parameters"], COMMON)
        self.assertEqual(_sweep("link_prediction", "pubmed", decoder_type="dot")["parameters"],
                         {**COMMON, "encoder_dropout": DROP})
        self.assertEqual(_sweep("link_prediction", "acm_pa",
                                decoder_type="concat_mlp")["parameters"],
                         {**COMMON, "encoder_dropout": DROP, "classifier_dropout": DROP})
        open_decoder = _sweep("link_prediction", "pubmed", include_decoder_type=True)
        self.assertEqual(open_decoder["parameters"]["decoder_type"],
                         {"values": ["dot", "distmult", "concat_mlp"]})

    def test_graph_classification_space(self) -> None:
        for dataset in ("mutag", "proteins", "enzymes", "nci1"):
            self.assertEqual(_sweep("graph_classification", dataset)["parameters"],
                             {**COMMON, "graph_pooling":
                              {"values": ["mean", "max", "add", "attention"]}}, dataset)

    def test_shared_axes_are_the_baselines(self) -> None:
        from src.baselines import search_grids as grids

        self.assertEqual((grids.LR, grids.WEIGHT_DECAY, grids.DROPOUT), (LR, WD, DROP))

    def test_overrides_reach_the_config(self) -> None:
        from scripts.run_h2snn_wandb_tuning import apply_sweep_overrides

        raw = {"data": {"dataset_name": "mutag"}, "model": {"propagation": "anchored"},
               "ablation": {"coupling": "symmetric"}}
        params = {"d": 32, "lr": 3e-3, "weight_decay": 0.1, "wisp_max_length": 5,
                  "wisp_walks_per_anchor": 64, "graph_pooling": "attention"}
        config = TrainConfig.from_dict(apply_sweep_overrides(raw, params))
        self.assertEqual((config.model.d, config.optim.lr, config.optim.weight_decay,
                          config.data.wisp_max_length, config.data.wisp_walks_per_anchor,
                          config.model.graph_pooling), (32, 3e-3, 0.1, 5, 64, "attention"))


class NeverTunedTests(unittest.TestCase):
    def test_wisp_support_and_precision_are_tunable(self) -> None:
        reject_never_tuned_settings({"data": {"dataset_name": "acm", "wisp_max_length": 3,
                                              "wisp_walks_per_anchor": 16}},
                                    ["data.wisp_max_length", "data.wisp_walks_per_anchor"])

    def test_K_layer_dropout_and_legacy_keys_are_refused(self) -> None:
        for key in ("K", "layer_dropout"):
            with self.assertRaisesRegex(ValueError, "not a component of the method"):
                reject_never_tuned_settings(None, [key])
        for key in NEVER_TUNED_LEGACY_KEYS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                reject_never_tuned_settings(None, [f"model.{key}"])
        for key in ("encoder_dropout", "classifier_dropout", "graph_pooling", "d", "lr"):
            reject_never_tuned_settings(None, [key])  # ordinary hyperparameters


    def test_the_anchored_model_can_be_tuned(self) -> None:
        """v3-B has no gates: `coupling: symmetric` is its own setting (refused only for the
        legacy propagation, where it is an ablation row)."""
        reject_never_tuned_settings({"data": {"dataset_name": "acm"},
                                     "model": {"propagation": "anchored"},
                                     "ablation": {"coupling": "symmetric"}}, ["lr"])
        with self.assertRaises(ValueError):
            reject_never_tuned_settings({"data": {"dataset_name": "acm"},
                                         "ablation": {"coupling": "symmetric"}}, ["lr"])


class SeedProtocolTests(unittest.TestCase):
    def test_tuning_seed_and_final_seeds(self) -> None:
        import scripts.run_h2snn_wandb_tuning as tuning

        self.assertEqual((tuning.TUNING_SEED, tuning.FINAL_SEEDS), (0, (0, 1, 2, 3, 4)))

    def test_final_seeds_train_the_best_seed0_configuration(self) -> None:
        """--final_seeds reads best_tuned_run0.json and trains its sweep_params on each seed (no
        re-tuning). train_trial is replaced by a recorder (no training here)."""
        import scripts.run_h2snn_wandb_tuning as tuning

        with tempfile.TemporaryDirectory() as tmp:
            raw = {"data": {"dataset_name": "acm"}, "output_dir": tmp, "exp_name": "exp"}
            base = TrainConfig.from_dict(raw)
            summary = Path(tmp) / "exp"
            summary.mkdir(parents=True)
            best = {"sweep_params": {"d": 32, "lr": 1e-3, "wisp_max_length": 4,
                                     "wisp_walks_per_anchor": 16}}
            (summary / "best_tuned_run0.json").write_text(json.dumps(best), encoding="utf-8")
            calls = []

            def fake_trial(raw_config, *, params, seed, **kwargs):
                calls.append((seed, dict(params)))
                return {"seed": seed, "test_macro_f1": 0.5, "test_micro_f1": 0.5,
                        "test_acc": 0.5, "val_macro_f1": 0.5}

            with mock.patch.object(tuning, "train_trial", fake_trial), \
                    mock.patch.object(tuning, "write_summary_files", lambda *a, **k: None):
                tuning.run_final_seeds(raw, args=Namespace(final_seeds=[0, 1, 2, 3, 4],
                                                           monitor=None),
                                       base_cfg=base, base_exp_name="exp")
            self.assertEqual([seed for seed, _ in calls], [0, 1, 2, 3, 4])
            self.assertTrue(all(params == best["sweep_params"] for _, params in calls))


class BaselineCampaignTests(unittest.TestCase):
    def test_baseline_sweeps_are_plain_bayes(self) -> None:
        from src.baselines.campaign import Cell, build_sweep_config

        sweep = build_sweep_config(Cell("gcn", "node_classification", "acm"), monitor="macro_f1",
                                   n_trials=100, epochs=300)
        self.assertEqual(sweep["method"], "bayes")
        self.assertNotIn("early_terminate", sweep)


if __name__ == "__main__":
    unittest.main()
