"""Card 5 T4 / T9 reference capture. Run from the H2S-NN/ folder of a checkout (the base commit 894fd74 for the
reference; the card5 tests call `records()` from the current code and compare):

    ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 python <this file> <out.json>

Synthetic data only (`tests.partO_fixtures.acm_like_hetero`, the real data root never read). Records:
- `hgb_full_batch`: the full-batch NC trainer on an ACM-shaped graph, 3 epochs: `best_test_metrics`;
- `acm_minibatch_flag`: the mini-batch trainer on the same graph with ACM's analysis flag (not a B1a dataset), 3
  epochs: `best_test_metrics`;
- `baselines_1_cell_list`: `make_cell_list.py --phase baselines-1` text with `<out>` = `/OUT`.
One thread; values stored as float.hex for an exact compare.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))


def _hex(record: dict) -> dict:
    return {k: (float(v).hex() if isinstance(v, float) else v) for k, v in sorted(record.items())}


def _fit(raw: dict) -> dict:
    from src.builders import build_experiment_components
    from src.configs.config import TrainConfig
    from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

    with tempfile.TemporaryDirectory() as tmp, patched_acm_loader(
            acm_like_hetero(num_papers=30, num_authors=16, seed=13)):
        cfg = TrainConfig.from_dict({**raw, "output_dir": tmp})
        torch.manual_seed(0)
        data, _, _, _, logger, _, trainer = build_experiment_components(cfg)
        try:
            trainer.fit(data)
        finally:
            logger.close()
        return _hex(trainer._best_test_metrics)


def records() -> dict:
    torch.set_num_threads(1)
    model = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
             "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"}
    base = {"device": "cpu", "seed": 3, "epochs": 3, "exp_name": "t4", "early_stopping_patience": 100,
            "checkpoint_metric": "macro_f1", "checkpoint_mode": "max", "model": model,
            "ablation": {"coupling": "symmetric"}}
    full_batch = _fit({**base, "data": {"dataset_name": "acm", "wisp_max_length": 3,
                                         "wisp_walks_per_anchor": 3}})
    minibatch = _fit({**base, "data": {"dataset_name": "acm", "training_mode": "minibatch", "batch_size": 8,
                                        "epoch_targets": 16, "wisp_max_length": 3, "wisp_walks_per_anchor": 3}})
    from scripts.tsubame.make_cell_list import tuning_list

    return {"hgb_full_batch": full_batch, "acm_minibatch_flag": minibatch,
            "baselines_1_cell_list": "\n".join(tuning_list("baselines-1", Path("/OUT"))) + "\n"}


if __name__ == "__main__":
    from tests.partMB_fixtures import unreachable_real_data

    unreachable_real_data()
    out = Path(sys.argv[1])
    captured = records()
    out.write_text(json.dumps(captured, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, str) else f"{len(v)} chars") for k, v in captured.items()},
                     indent=1))
