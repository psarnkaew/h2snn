"""Card 4b (`docs/H2S-NN_card4b_prefreeze_card.md`; spec 1.24 Y4). Synthetic data only.

L2 arm-sweep key exclusion (`ARM_INERT_SEARCH_KEYS`, `arm_search_space`, the tuning launcher):
  (1) the `no_path_cells` sweep space lacks `wisp_max_length` and `wisp_walks_per_anchor`, every other
      arm's space equals the main-table space minus its (empty) entry;
  (2) a main-table sweep keeps every key and refuses the exclusion;
  (3) the table check: every entry is shown unread (the arm's evaluation logits are bit-identical for two
      values of the key, shared parameters copied), and a mutated entry for a key that is read in that
      arm (`wisp_max_length` for `full`) is refused by the same check.
"""

from __future__ import annotations

import argparse
import copy
import sys
import unittest
from pathlib import Path

import torch
import yaml

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FULL = ROOT / "configs" / "experiments" / "acm_h2snn_v3b.yaml"
SMALL = {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
         "classifier_hidden_dim": 16}
# Two values of each search key used by the table check (both inside the tuned ranges).
PROBE_VALUES = {"wisp_max_length": (2, 4), "wisp_walks_per_anchor": (3, 8)}


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def _arm_raw(arm: str) -> dict:
    """The arm's tuning base config (`full`'s file + the arm's key + tuning.arm)."""
    from src.configs.config import ARM_TUNING_KEYS

    raw = yaml.safe_load(FULL.read_text(encoding="utf-8"))
    if arm:
        section, key, value = ARM_TUNING_KEYS[arm]
        raw[section] = {**raw.get(section, {}), key: True if value == "True" else value}
        raw["tuning"] = {"arm": arm}
    return raw


def _sweep_space(raw: dict) -> dict:
    from run_h2snn_wandb_tuning import _build_sweep_for_run
    from src.configs.config import TrainConfig

    cfg = TrainConfig.from_dict(copy.deepcopy(raw))
    args = argparse.Namespace(d_values=None)
    return _build_sweep_for_run(args=args, base_cfg=cfg, base_exp_name="x", monitor="macro_f1",
                                run_index=0)["parameters"]


def _eval_logits(model_overrides: dict, ablation: dict, key: str, value: int,
                 state: dict | None):
    from src.configs.config import TrainConfig
    from src.data.dataset import build_dataset
    from src.models.h2snn_model import build_model
    from tests.partO_fixtures import acm_like_hetero, patched_acm_loader

    data_cfg = {"dataset_name": "acm", "wisp_max_length": 3, "wisp_walks_per_anchor": 5, key: value}
    cfg = TrainConfig.from_dict({"device": "cpu", "seed": 3, "data": data_cfg,
                                 "model": {**SMALL, **model_overrides}, "ablation": ablation})
    with patched_acm_loader(acm_like_hetero(seed=9)):
        data = build_dataset(cfg)
    torch.manual_seed(0)
    model = build_model(cfg, data)
    if state is not None:   # shared parameters from the other model (extra ones, e.g. g's
        own = model.state_dict()   # projection for a longer l_max, stay as built)
        model.load_state_dict({k: v for k, v in state.items()
                               if k in own and own[k].shape == v.shape}, strict=False)
    model.eval()
    with torch.no_grad():
        return model(data).logits.clone(), model.state_dict()


def key_is_unread(arm: str, key: str) -> bool:
    """True iff the arm's evaluation logits are bit-identical for the two probe values of `key`."""
    from src.configs.config import ARM_TUNING_KEYS

    model_overrides, ablation = {}, {}
    if arm != "full":
        section, name, value = ARM_TUNING_KEYS[arm]
        (model_overrides if section == "model" else ablation)[name] = (
            True if value == "True" else value)
    first, second = PROBE_VALUES[key]
    logits_a, state = _eval_logits(model_overrides, ablation, key, first, None)
    logits_b, _ = _eval_logits(model_overrides, ablation, key, second, state)
    return bool(torch.equal(logits_a, logits_b))


def check_inert_table(table: dict[str, tuple[str, ...]]) -> None:
    """The table check of card 4b L2: every entry must be shown unread, else ValueError."""
    for arm, keys in table.items():
        for key in keys:
            if not key_is_unread(arm, key):
                raise ValueError(f"{key} is read in arm {arm}: not an inert search key")


class ArmSweepKeyExclusionTests(unittest.TestCase):
    def test_arm_spaces_drop_exactly_their_inert_keys(self) -> None:
        from src.configs.config import ARM_INERT_SEARCH_KEYS

        main = _sweep_space(_arm_raw(""))
        for arm, inert in ARM_INERT_SEARCH_KEYS.items():
            space = _sweep_space(_arm_raw(arm))
            self.assertEqual(space, {k: v for k, v in main.items() if k not in inert}, arm)
        self.assertNotIn("wisp_max_length", _sweep_space(_arm_raw("no_path_cells")))
        self.assertNotIn("wisp_walks_per_anchor", _sweep_space(_arm_raw("no_path_cells")))

    def test_main_table_sweep_keeps_every_key_and_refuses_the_exclusion(self) -> None:
        from run_h2snn_wandb_tuning import build_sweep_config
        from src.configs.config import arm_search_space

        main = _sweep_space(_arm_raw(""))
        self.assertIn("wisp_max_length", main)
        self.assertIn("wisp_walks_per_anchor", main)
        reference = build_sweep_config(task_type="node_classification", metric="macro_f1",
                                       epochs=300, dataset_name="acm")["parameters"]
        self.assertEqual(main, reference)
        with self.assertRaises(ValueError):
            arm_search_space(main, "")

    def test_table_entries_are_unread_and_a_read_key_is_refused(self) -> None:
        from src.configs.config import ARM_INERT_SEARCH_KEYS

        check_inert_table(ARM_INERT_SEARCH_KEYS)                      # every entry unread
        with self.assertRaises(ValueError):                           # mutation: read in `full`
            check_inert_table({"full": ("wisp_max_length",)})


if __name__ == "__main__":
    unittest.main()
