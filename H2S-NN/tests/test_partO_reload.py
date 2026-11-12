"""Part O, O1 (`docs/H2S-NN_partO_card.md`): what the server runs keep, and the exact reload.

O1.1 the main-table seed runs (`scripts/run_h2snn_wandb_tuning.py --final_seeds`, the real
     `run_final_seeds` / `train_trial`) leave, per seed, `config.yaml` (the resolved configuration:
     seed = base seed of the evaluation draw, l_max, r, path source) and `checkpoints/best.pt`;
     a later seed deletes or overwrites nothing of an earlier one.
O1.2 train 2 epochs on a tiny ACM-shaped graph (anchored v3-B model, WISP), save; reload from
     `best.pt` + `config.yaml` in a fresh process; the rebuilt evaluation cells (node, fact and sign
     tensors, relation and signature ids, m_f) and w_f = exp(g(xi_f)) equal those of the end of
     training exactly (`torch.equal`, same dtype and device). Negative control (the card's
     mutation): the same reload with a different base seed in `config.yaml` is detected.

Synthetic graphs only (no real data, no metric is read).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

try:
    import torch
    import yaml
except ImportError as exc:  # pragma: no cover - environment specific
    raise unittest.SkipTest(f"torch and yaml are required for these tests: {exc}") from exc

from tests.partO_fixtures import acm_like_hetero, patched_acm_loader, save_hetero

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _raw_config(output_dir: Path, exp_name: str, seed: int) -> dict:
    """A v3-B main-model configuration (anchored, arm full, WISP), small sizes, CPU."""
    return {
        "device": "cpu", "seed": seed, "epochs": 2, "early_stopping_patience": 50,
        "exp_name": exp_name, "output_dir": str(output_dir),
        "checkpoint_metric": "macro_f1", "checkpoint_mode": "max",
        "data": {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 4},
        "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
                  "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
        "ablation": {"coupling": "symmetric"},
        "optim": {"name": "adamw", "lr": 0.01, "weight_decay": 0.0},
    }


class _FakeWandb:
    """`train_trial`'s epoch callback logs to W&B; without a run the real module raises."""

    @staticmethod
    def log(*args, **kwargs) -> None:
        return None


def _cells(groups) -> list[dict[str, torch.Tensor]]:
    from tests.partO_reload_child import cell_tensors

    return cell_tensors(groups)


def _first_difference(expected: dict, actual: dict) -> str | None:
    """Name of the first tensor that differs (value, shape, dtype or device), or None."""
    if len(expected["cells"]) != len(actual["cells"]):
        return f"number of groups {len(expected['cells'])} != {len(actual['cells'])}"
    for g, (a, b) in enumerate(zip(expected["cells"], actual["cells"])):
        for key in a:
            x, y = a[key], b[key]
            if x is None or y is None:
                if x is not y:
                    return f"group {g} {key}: one side is None"
                continue
            if x.dtype != y.dtype or x.device != y.device or not torch.equal(x, y):
                return f"group {g} {key}"
    if len(expected["weights"]) != len(actual["weights"]):
        return "number of w_f tensors"
    for g, (x, y) in enumerate(zip(expected["weights"], actual["weights"])):
        if x.dtype != y.dtype or x.device != y.device or not torch.equal(x, y):
            return f"group {g} w_f"
    return None


def _reload_in_fresh_process(run_dir: Path, graph: Path, out: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "tests.partO_reload_child", str(run_dir), str(graph), str(out)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise AssertionError(f"reload process failed:\n{result.stdout}\n{result.stderr}")
    return torch.load(out, weights_only=False)


class RetentionTests(unittest.TestCase):
    """O1.1 through the campaign's own final-seed path."""

    def test_final_seed_runs_keep_config_and_best_checkpoint(self) -> None:
        import scripts.run_h2snn_wandb_tuning as tuning

        hetero = acm_like_hetero(seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            raw = _raw_config(Path(tmp), "exp", seed=0)
            base = tuning.TrainConfig.from_dict(raw)
            summary = Path(tmp) / "exp"
            summary.mkdir(parents=True)
            params = {"wisp_max_length": 3, "wisp_walks_per_anchor": 4, "lr": 0.01}
            (summary / "best_tuned_run0.json").write_text(json.dumps({"sweep_params": params}),
                                                          encoding="utf-8")
            with patched_acm_loader(hetero), mock.patch.dict(sys.modules, {"wandb": _FakeWandb}):
                tuning.run_final_seeds(raw, args=Namespace(final_seeds=[0, 1], monitor=None),
                                       base_cfg=base, base_exp_name="exp")
            for seed in (0, 1):
                with self.subTest(seed=seed):
                    run_dirs = sorted((summary / "final_seeds").glob(f"*final_seed{seed}*"))
                    self.assertEqual(len(run_dirs), 1, run_dirs)
                    run_dir = run_dirs[0]
                    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
                    self.assertEqual(saved["seed"], seed)  # base seed of the evaluation draw
                    self.assertEqual(saved["data"]["path_source"], "wisp")  # resolved
                    self.assertEqual(saved["data"]["wisp_max_length"], 3)
                    self.assertEqual(saved["data"]["wisp_walks_per_anchor"], 4)
                    payload = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=False)
                    self.assertIn("state_dict", payload)
                    self.assertIn(payload["epoch"], (0, 1))


class ReloadTests(unittest.TestCase):
    """O1.2: exact reload of the evaluation draw and w_f in a fresh process."""

    def test_reload_rebuilds_cells_and_weights_exactly(self) -> None:
        from src.analysis.model_loading import evaluation_cell_weights, nonempty_groups
        from src.builders import build_experiment_components
        from src.configs.config import TrainConfig
        from src.utils.misc import save_config, seed_everything

        hetero = acm_like_hetero(seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            graph = Path(tmp) / "graph.pt"
            save_hetero(hetero, graph)
            # what `train_trial` does for one seed (scripts/run_h2snn_wandb_tuning.py:384-473),
            # with an epoch callback that records w_f on the evaluation draw after each epoch
            cfg = TrainConfig.from_dict(_raw_config(Path(tmp), "reload", seed=5))
            cfg.exp_dir.mkdir(parents=True, exist_ok=True)
            save_config(cfg, cfg.exp_dir / "config.yaml")
            seed_everything(cfg.seed, cfg.device)
            with patched_acm_loader(hetero):
                data, model, _, _, logger, _, trainer = build_experiment_components(cfg)
            snapshots: dict[int, tuple[torch.Tensor, ...]] = {}

            def snapshot(epoch, *_metrics) -> None:  # runs after the epoch's checkpoint save
                snapshots[epoch] = tuple(w.clone() for w in evaluation_cell_weights(model, data))

            try:
                trainer.fit(data, epoch_callback=snapshot)
            finally:
                logger.close()
            best_epoch = int(torch.load(cfg.exp_dir / "checkpoints" / "best.pt",
                                        weights_only=False)["epoch"])
            expected = {"cells": _cells(nonempty_groups(data.face_groups)),
                        "weights": list(snapshots[best_epoch])}
            self.assertTrue(expected["weights"], "the evaluation draw has cells")

            actual = _reload_in_fresh_process(cfg.exp_dir, graph, Path(tmp) / "reload.pt")
            self.assertEqual(actual["seed"], cfg.seed)
            difference = _first_difference(expected, actual)
            self.assertIsNone(difference, f"first differing tensor: {difference}")

            # negative control: the same run with another base seed in its configuration
            mutated = Path(tmp) / "reload_other_seed"
            shutil.copytree(cfg.exp_dir, mutated)
            saved = yaml.safe_load((mutated / "config.yaml").read_text(encoding="utf-8"))
            saved["seed"] = cfg.seed + 1
            (mutated / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False),
                                                 encoding="utf-8")
            other = _reload_in_fresh_process(mutated, graph, Path(tmp) / "other.pt")
            self.assertIsNotNone(_first_difference(expected, other),
                                 "a reload with another base seed must not match")


if __name__ == "__main__":
    unittest.main()
