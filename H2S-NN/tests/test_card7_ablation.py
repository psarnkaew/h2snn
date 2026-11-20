"""Card 7 T1, updated by card 9 W9 (pre-registration v3): the ablation launchers write exactly the 8 rows (a), no
(b) row and no `tune` row, and no manifest row names an out-of-scope arm. `sweeps.py` stays in the tree (uncalled by
any manifest) and still writes its 4 plan-v2 arms. Synthetic configurations only (the ACM experiment file as a
placeholder `full`); the real data root is unreachable.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from tests.partMB_fixtures import restore_real_data, unreachable_real_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ablation"))
sys.path.insert(0, str(ROOT / "scripts"))
FULL = ROOT / "configs" / "experiments" / "acm_h2snn_v3b.yaml"
FREEBASE = ROOT / "configs" / "experiments" / "freebase_h2snn_v3b.yaml"

ROWS = {"full", "no_path_cells", "S2", "S0", "uniform", "endpoints", "instance_features", "literature"}
TUNED = {"endpoints", "no_path_cells", "instance_features", "S2"}
# Card 9 "Out of scope" (v3 brings `uniform` back as a row): arms that must not appear in any list or manifest.
OUT_OF_SCOPE = ("closure_only", "hub_A", "covering_only", "schema", "direction_blind_maps",
                "free_cell_maps", "ego_head", "h0_only")


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def out_of_scope_in(text: str) -> list[str]:
    """Out-of-scope arm names in `text` as whole tokens (`schema` is also part of `path_source: schemas` and of
    the literature row's file names, so tokens are split on non-identifier characters)."""
    import re

    tokens = {part for token in re.split(r"[^A-Za-z0-9_]+", text) for part in token.split("__") if part}
    return [arm for arm in OUT_OF_SCOPE if any(t == arm or t.endswith(f"_{arm}") for t in tokens)]


class LauncherTests(unittest.TestCase):
    def test_arms_writes_exactly_the_eight_rows(self) -> None:
        import arms

        self.assertEqual(set(arms.ARM_CHANGES), ROWS)
        self.assertEqual(set(arms.TUNED_ARMS), TUNED)          # kept for sweeps.py only
        self.assertEqual(arms.SEEDS, (1, 2, 3, 4, 5))
        with tempfile.TemporaryDirectory() as tmp:
            paths = arms.emit_configurations_a(FULL, "acm", Path(tmp))
            self.assertEqual(set(paths), ROWS)
            self.assertEqual({p.name for p in (Path(tmp) / "acm").iterdir()}, {f"{r}.yaml" for r in ROWS})
            literature = yaml.safe_load(paths["literature"].read_text(encoding="utf-8"))
            self.assertEqual(literature["data"]["path_source"], "schemas")

    def test_sweeps_writes_exactly_the_four_b_arms(self) -> None:
        import sweeps

        with tempfile.TemporaryDirectory() as tmp:
            records = sweeps.emit_configurations_b(FULL, Path(tmp))
            self.assertEqual(set(records), TUNED)
            for arm, record in records.items():
                self.assertEqual((record["n_trials"], record["seed"]), (100, 0), arm)
                self.assertTrue(record["launcher_space_matches"], arm)
                base = yaml.safe_load(Path(record["base_config"]).read_text(encoding="utf-8"))
                self.assertEqual(base["tuning"], {"arm": arm})
            written = {p.name for p in (Path(tmp) / "acm").iterdir()}
            self.assertEqual(written, {f"tune_{a}.yaml" for a in TUNED} | {f"sweep_{a}.json" for a in TUNED})

    def test_manifest_holds_the_plan_rows_and_no_out_of_scope_arm(self) -> None:
        import seed_runs

        with tempfile.TemporaryDirectory() as tmp:
            rows = seed_runs.build_manifest("acm", FULL, Path(tmp) / "main", Path(tmp), None)
            seed_runs.write_manifest(rows, Path(tmp) / "acm" / "manifest.tsv")
            kinds = [(r["kind"], r["configuration"], r["arm"]) for r in rows]
            self.assertEqual(sorted(kinds), sorted([("runs", "a", a) for a in ROWS]))
            for row in rows:
                first = 5 if row["arm"] == "full" else 1
                self.assertEqual((row["base_seed"], row["base_seed"] + row["num_runs"] - 1), (first, 5), row)
            text = (Path(tmp) / "acm" / "manifest.tsv").read_text(encoding="utf-8")
            self.assertEqual(out_of_scope_in(text), [], text)

    def test_freebase_mode_is_an_input_written_into_every_configuration(self) -> None:
        import seed_runs

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                seed_runs.build_manifest("freebase", FREEBASE, Path(tmp) / "main", Path(tmp), None)
            rows = seed_runs.build_manifest("freebase", FREEBASE, Path(tmp) / "main", Path(tmp), "minibatch")
            self.assertEqual(len(rows), len(ROWS))
            for row in rows:
                path = Path(row["config"])
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(raw["data"]["training_mode"], "minibatch", path.name)

    def test_the_out_of_scope_check_catches_an_arm(self) -> None:
        self.assertEqual(out_of_scope_in("runs\tacm\tclosure_only\ta"), ["closure_only"])
        self.assertEqual(out_of_scope_in("acm__ablation_a_ego_head"), ["ego_head"])
        self.assertEqual(out_of_scope_in("tune_schema.yaml"), ["schema"])
        self.assertEqual(out_of_scope_in("path_source: schemas literature.yaml"), [])


if __name__ == "__main__":
    unittest.main()
