"""Part M3a-fix, F1 (`docs/H2S-NN_partM3a_fix_card.md`; spec v1.22 W2; decisions log 2026-11-15, Part M3a
review, decision 2): `concat_mlp` is refused by config validation and removed from the model.

- a config with `data.decoder_type: concat_mlp` on an LP dataset raises `ValueError` naming W2: through
  `TrainConfig.from_dict`, a YAML through the config loader, and the first two steps of
  `train_h2snn.py --config` / `run_h2snn_runs.py --config` (`parse_args`, then the script's own
  `load_config`; nothing is trained);
- `LinkPredictionReadout(decoder_type="concat_mlp")` raises (the branch is gone); dot and distmult build;
- the refusal of `reject_never_tuned_settings` (M3a) stays.
Card mutation: the config accepts `concat_mlp` again -> `test_config_refuses_concat_mlp` fails.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

REPO = Path(__file__).resolve().parents[1]
RAW = {"data": {"dataset_name": "pubmed", "decoder_type": "concat_mlp"}}


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class ConcatMlpConfigTests(unittest.TestCase):
    def test_config_refuses_concat_mlp(self) -> None:
        from src.configs.config import TrainConfig

        with self.assertRaisesRegex(ValueError, r"data\.decoder_type: concat_mlp is removed .*W2"):
            TrainConfig.from_dict(RAW)
        for decoder in ("dot", "distmult"):  # the two decoders still resolve
            cfg = TrainConfig.from_dict({"data": {"dataset_name": "pubmed", "decoder_type": decoder}})
            self.assertEqual(cfg.data.decoder_type, decoder)

    def test_yaml_and_run_scripts_refuse_concat_mlp(self) -> None:
        from src.configs.config import TrainConfig

        with tempfile.TemporaryDirectory(dir=REPO) as tmp:
            path = Path(tmp) / "lp_concat_mlp.yaml"
            path.write_text(yaml.safe_dump(RAW), encoding="utf-8")
            with self.subTest(loader="yaml + TrainConfig.from_dict"), \
                    self.assertRaisesRegex(ValueError, "W2"):
                TrainConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
            for module_name, prog in (("scripts.train_h2snn", "train_h2snn.py"),
                                      ("scripts.run_h2snn_runs", "run_h2snn_runs.py")):
                module = __import__(module_name, fromlist=["parse_args", "load_config"])
                with self.subTest(script=prog), \
                        mock.patch.object(sys, "argv", [prog, "--config", str(path)]):
                    args = module.parse_args()
                    with self.assertRaisesRegex(ValueError, "W2"):
                        module.load_config(args.config)

    def test_readout_has_no_concat_mlp(self) -> None:
        from src.models.link_readout import LinkPredictionReadout

        with self.assertRaisesRegex(ValueError, "Unsupported decoder_type 'concat_mlp'"):
            LinkPredictionReadout(d=4, decoder_type="concat_mlp")
        for decoder in ("dot", "distmult"):
            readout = LinkPredictionReadout(d=4, decoder_type=decoder)
            self.assertFalse(hasattr(readout, "mlp"))

    def test_tuning_refusal_stays(self) -> None:
        from src.configs.config import reject_never_tuned_settings

        with self.assertRaisesRegex(ValueError, "concat_mlp is removed"):
            reject_never_tuned_settings(RAW)


if __name__ == "__main__":
    unittest.main()
