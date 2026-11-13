"""Part P-fix, F3 (`docs/H2S-NN_partPfix_card.md`; spec v1.20 U2, decision D-O7): the read-out
script reports Kendall tau-b over length-2 sub-path schemas only.

On two synthetic 1-epoch runs: the dataset table holds exactly one tau block (length-2), the
dataset JSON exactly one tau entry (`kendall_tau_length2`), and its values equal
`kendall_matrix(records, span=2)` recomputed from the runs' JSONs; no value is reported.
Card mutation: the all-spans tau in the script -> this test fails.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

PERMUTATIONS = 50


class LengthTwoTauTests(unittest.TestCase):
    def test_one_tau_block_over_length_two_schemas(self) -> None:
        import scripts.schema_readout as readout
        from src.analysis.schema_readout import kendall_matrix
        from src.builders import build_experiment_components
        from src.configs.config import TrainConfig
        from src.utils.misc import save_config

        with tempfile.TemporaryDirectory() as tmp, patched_acm_loader(acm_like_hetero(seed=8)):
            run_dirs = []
            for seed in (0, 1):
                cfg = TrainConfig.from_dict({
                    "device": "cpu", "seed": seed, "epochs": 1, "exp_name": f"run{seed}",
                    "output_dir": tmp, "checkpoint_metric": "macro_f1", "checkpoint_mode": "max",
                    "data": {"dataset_name": "acm", "wisp_max_length": 3,
                             "wisp_walks_per_anchor": 4},
                    "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16,
                              "face_mlp_hidden_dim": 16, "classifier_hidden_dim": 16,
                              "propagation": "anchored", "arm": "full"},
                    "ablation": {"coupling": "symmetric"}})
                cfg.exp_dir.mkdir(parents=True)
                save_config(cfg, cfg.exp_dir / "config.yaml")
                data, _, _, _, logger, _, trainer = build_experiment_components(cfg)
                try:
                    trainer.fit(data)
                finally:
                    logger.close()
                run_dirs.append(cfg.exp_dir)
            out = Path(tmp) / "readout" / "acm"
            readout.main(["--runs", *map(str, run_dirs), "--out_dir", str(out.parent),
                          "--permutations", str(PERMUTATIONS)])
            table = (out / "schema_readout.md").read_text("utf-8")
            summary = json.loads((out / "schema_readout.json").read_text("utf-8"))
            records = [json.loads((out / f"{d.name}.json").read_text("utf-8")) for d in run_dirs]

        headings = [line for line in table.splitlines() if "Kendall tau" in line]
        self.assertEqual(headings, ["## Kendall tau of the I rankings between seeds "
                                    "(length-2 sub-path schemas)"])
        self.assertEqual([k for k in summary if "tau" in k], ["kendall_tau_length2"])
        expected = kendall_matrix(records, span=2, permutations=PERMUTATIONS)
        self.assertEqual(summary["kendall_tau_length2"], expected)
        all_spans = kendall_matrix(records, span=None, permutations=PERMUTATIONS)
        self.assertNotEqual(expected, all_spans, "the fixture separates the two definitions")


if __name__ == "__main__":
    unittest.main()
