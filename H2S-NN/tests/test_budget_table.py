"""Part M3b B8: one budget table per (task, dataset class), read by H2S-NN's trainers, the registry and the adapters.

* the table (`src/configs/config.py`: `PROTOCOL_BUDGETS` + `BATCH_BUDGETS`, listed by `budget_table()`) has one row
  per class with the values of spec §6 / MB2 / MB4;
* H2S-NN reads it: mini-batch NC resolves B and the epoch from it, the large-LP loop takes its 8,192 from it;
* a baseline's budget follows the dataset's source protocol (`resolve_budget(task, params, dataset)`);
* runtime check over the registry: every cell on a loaded H2GB NC dataset or a large LP task is either runnable by an
  adapter that reads `batch_plan` (declares `READS_BATCH_PLAN`) or carries a non-empty pending reason.
AdamW and no scheduler are checked by the M9 tests (`tests/test_shared_protocol.py`), unchanged.
"""

from __future__ import annotations

import importlib
import unittest


class BudgetTableTests(unittest.TestCase):
    def test_one_row_per_task_and_dataset_class(self) -> None:
        from src.configs.config import budget_table

        rows = budget_table()
        keys = [(r["task"], r["class"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        by = {r["class"]: r for r in rows}
        self.assertEqual(set(by), {"hgb_nc", "h2gb_nc", "local_lp", "large_lp", "gc"})
        self.assertEqual((by["h2gb_nc"]["epochs"], by["h2gb_nc"]["patience"], by["h2gb_nc"]["batch_sizes"]),
                         (500, 40, (32, 256, 1024)))
        self.assertEqual((by["hgb_nc"]["epochs"], by["hgb_nc"]["patience"], by["hgb_nc"]["batch_sizes"]),
                         (500, 40, None))
        self.assertEqual((by["large_lp"]["epochs"], by["large_lp"]["patience"], by["large_lp"]["positives_per_step"],
                          by["large_lp"]["mode"]), (500, 40, 8192, "minibatch"))
        self.assertEqual(by["local_lp"]["positives_per_step"], 8192)

    def test_dataset_classes(self) -> None:
        from src.configs.config import batch_budget

        self.assertEqual(batch_budget("node_classification", "ogbn-mag").dataset_class, "h2gb_nc")
        self.assertEqual(batch_budget("node_classification", "acm").dataset_class, "hgb_nc")
        self.assertEqual(batch_budget("link_prediction", "h-pokec_uu").dataset_class, "large_lp")
        self.assertEqual(batch_budget("link_prediction", "acm_pa").dataset_class, "local_lp")
        self.assertEqual(batch_budget("graph_classification", "mutag").dataset_class, "gc")


class ReadersTests(unittest.TestCase):
    def test_h2snn_reads_the_table(self) -> None:
        from src.configs.config import TrainConfig, batch_budget
        from src.trainers.minibatch_lp import LP_POSITIVES_PER_STEP

        plan = batch_budget("node_classification", "ogbn-mag")
        config = TrainConfig.from_dict({"data": {"dataset_name": "ogbn-mag", "batch_size": 256,
                                                 "wisp_walks_per_anchor": 4}})
        self.assertIn(config.data.batch_size, plan.batch_sizes)
        self.assertEqual(config.data.epoch_targets, plan.epoch_targets)
        self.assertEqual(LP_POSITIVES_PER_STEP, batch_budget("link_prediction", "ogbn-mag_pp").positives_per_step)

    def test_baseline_budget_follows_the_source_protocol(self) -> None:
        from src.baselines.budget import batch_plan, resolve_budget
        from src.baselines.protocol import BaselineParams

        params = BaselineParams(values={}, source="tuned")
        self.assertEqual((resolve_budget("node_classification", params, "oag-cs").epochs,
                          resolve_budget("node_classification", params, "oag-cs").patience), (500, 40))
        self.assertEqual((resolve_budget("node_classification", params, "acm").epochs,
                          resolve_budget("node_classification", params).patience), (500, 40))
        self.assertEqual(batch_plan("node_classification", "h-pokec").epoch_targets, 8192)

    def test_registry_cells_on_h2gb_and_large_lp_read_the_table_or_say_why_not(self) -> None:
        from src.baselines.registry import (DATASETS, H2GB_NC, LARGE_LP, MODELS, model_spec,
                                            unsupported_reason)

        checked = 0
        for task, datasets in (("node_classification", H2GB_NC), ("link_prediction", LARGE_LP)):
            for dataset in datasets:
                self.assertIn(dataset, DATASETS[task])
                for spec in MODELS:
                    reason = unsupported_reason(spec.name, task, dataset)
                    with self.subTest(model=spec.name, task=task, dataset=dataset):
                        if reason is None:
                            module = importlib.import_module(model_spec(spec.name).module)
                            self.assertTrue(getattr(module, "READS_BATCH_PLAN", False),
                                            f"{spec.name} runs on {dataset} without reading batch_plan")
                        else:
                            self.assertTrue(reason.strip())
                    checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
