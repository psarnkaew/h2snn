"""Part M3a, A1 (`docs/H2S-NN_partM3a_card.md`; spec v1.22 W2; decisions log 2026-11-13 "H2S-NN LP
decoder"): H2S-NN's LP decoder is tuned in {dot, DistMult}, the LP head shared with every baseline.

- every LP sweep holds `decoder_type: {values: [dot, distmult]}`; apart from it the LP space is the
  space of the dot decoder before W2 (the removed `concat_mlp` branch added `classifier_dropout`);
- NC and GC sweeps hold no `decoder_type`;
- `concat_mlp` is refused in a base config, in a search space, in a trial's values and on the command
  line of the tuner and of both run scripts;
- the tuned values equal the LP decoder values of every runnable baseline cell, read from the
  registry and the adapters at run time (not written down twice).
Card mutations: the tuner omits `decoder_type` on LP -> `test_lp_sweeps_tune_the_decoder` fails;
`concat_mlp` accepted by the tuner -> `test_concat_mlp_is_refused_by_the_tuner` fails.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

LR = {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2}
WD = {"values": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.8]}
DROP = {"distribution": "uniform", "min": 0.0, "max": 0.9}
# The LP space of the dot / DistMult decoder before W2 (tests/test_search_spaces.py at v3b a68cbcc).
COMMON = {"d": {"values": [16, 32, 64]}, "lr": LR, "weight_decay": WD,
          "wisp_max_length": {"values": [2, 3, 4, 5]},
          "wisp_walks_per_anchor": {"values": [8, 16, 32, 64]}}
REPO = Path(__file__).resolve().parents[1]


def _sweep(task: str, dataset: str, **kwargs):
    from scripts.run_h2snn_wandb_tuning import build_sweep_config

    return build_sweep_config(task_type=task, metric="val_x", epochs=50, dataset_name=dataset,
                              **kwargs)


def _lp_datasets() -> tuple[str, ...]:
    from src.baselines.protocol import LINK_PREDICTION
    from src.baselines.registry import DATASETS

    return DATASETS[LINK_PREDICTION]


def _argv_error(module_name: str, argv: list[str]) -> str:
    """stderr of `parse_args` of a script for `argv`; the parse must exit."""
    module = __import__(module_name, fromlist=["parse_args"])
    stderr = io.StringIO()
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
        with unittest.TestCase().assertRaises(SystemExit):
            module.parse_args()
    return stderr.getvalue()


class LinkDecoderSweepTests(unittest.TestCase):
    def test_lp_sweeps_tune_the_decoder(self) -> None:
        from scripts.run_h2snn_wandb_tuning import FEATURELESS_DATASETS

        for dataset in _lp_datasets():
            for base_decoder in (None, "dot", "distmult"):
                with self.subTest(dataset=dataset, base_decoder=base_decoder):
                    params = dict(_sweep("link_prediction", dataset,
                                         decoder_type=base_decoder)["parameters"])
                    self.assertIn("decoder_type", params)
                    self.assertEqual(params.pop("decoder_type"),
                                     {"values": ["dot", "distmult"]})
                    expected = dict(COMMON)
                    if dataset not in FEATURELESS_DATASETS:
                        expected["encoder_dropout"] = DROP
                    self.assertEqual(params, expected)  # no classifier_dropout on LP

    def test_nc_and_gc_sweeps_have_no_decoder(self) -> None:
        for task, datasets in (("node_classification", ("acm", "dblp", "imdb", "freebase")),
                               ("graph_classification", ("mutag", "proteins", "enzymes", "nci1"))):
            for dataset in datasets:
                with self.subTest(task=task, dataset=dataset):
                    self.assertNotIn("decoder_type", _sweep(task, dataset)["parameters"])

    def test_values_equal_the_baselines_shared_head(self) -> None:
        from scripts.run_h2snn_wandb_tuning import SPACE_DECODER_TYPE
        from src.baselines.protocol import LINK_PREDICTION
        from src.baselines.registry import get_adapter, iter_cells

        seen: dict[tuple[str, ...], list[str]] = {}
        for entry in iter_cells():
            if entry["task"] != LINK_PREDICTION or not entry["runnable"]:
                continue
            space = get_adapter(entry["model"]).search_space(LINK_PREDICTION, entry["dataset"])
            self.assertIn("decoder_type", space, f"{entry['model']} {entry['dataset']}")
            values = tuple(space["decoder_type"]["values"])
            seen.setdefault(values, []).append(f"{entry['model']}/{entry['dataset']}")
        self.assertTrue(seen, "no runnable baseline LP cell")
        self.assertEqual(list(seen), [tuple(SPACE_DECODER_TYPE["values"])],
                         f"baseline LP decoder values {seen}")


class ConcatMlpRefusalTests(unittest.TestCase):
    def test_concat_mlp_is_refused_by_the_tuner(self) -> None:
        from scripts.run_h2snn_wandb_tuning import load_raw_config, validate_trial_params
        from src.configs.config import reject_never_tuned_settings

        # base config: the sweep builder, the never-tuned guard and a YAML loaded by the tuner
        with self.assertRaisesRegex(ValueError, "concat_mlp"):
            _sweep("link_prediction", "pubmed", decoder_type="concat_mlp")
        with self.assertRaisesRegex(ValueError, "concat_mlp is removed"):
            reject_never_tuned_settings({"data": {"dataset_name": "pubmed",
                                                  "decoder_type": "concat_mlp"}})
        with tempfile.TemporaryDirectory(dir=REPO) as tmp:
            path = Path(tmp) / "base.yaml"
            path.write_text("data:\n  dataset_name: pubmed\n  decoder_type: concat_mlp\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "concat_mlp"):
                load_raw_config(path)
        # search space and a trial's values
        for space in ({"decoder_type": {"values": ["dot", "distmult", "concat_mlp"]}},
                      {"data.decoder_type": {"value": "concat_mlp"}},
                      {"decoder_type": "concat_mlp"}):
            with self.subTest(space=space), self.assertRaisesRegex(ValueError,
                                                                   "concat_mlp is removed"):
                reject_never_tuned_settings(None, space)
        with self.assertRaisesRegex(ValueError, "decoder_type"):
            validate_trial_params({"decoder_type": "concat_mlp"})
        # the two allowed decoders pass the same checks
        reject_never_tuned_settings({"data": {"dataset_name": "pubmed", "decoder_type": "distmult"}},
                                    {"decoder_type": {"values": ["dot", "distmult"]}})

    def test_concat_mlp_is_refused_on_the_command_lines(self) -> None:
        for module, argv in (
            ("scripts.run_h2snn_wandb_tuning", ["tune", "--config", "x.yaml"]),
            ("scripts.run_h2snn_runs", ["runs", "--config", "x.yaml"]),
            ("scripts.train_h2snn", ["train", "--config", "x.yaml"]),
        ):
            with self.subTest(module=module):
                error = _argv_error(module, argv + ["--decoder_type", "concat_mlp"])
                self.assertIn("invalid choice: 'concat_mlp'", error)

    def test_include_decoder_type_flag_is_gone(self) -> None:
        for module, argv in (
            ("scripts.run_h2snn_wandb_tuning", ["tune", "--config", "x.yaml"]),
            ("scripts.run_graph_classification_repeated_cv_wandb_tuning",
             ["gc", "--config", "x.yaml"]),
        ):
            with self.subTest(module=module):
                error = _argv_error(module, argv + ["--include_decoder_type"])
                self.assertIn("unrecognized arguments: --include_decoder_type", error)


if __name__ == "__main__":
    unittest.main()
