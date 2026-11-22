"""Card 8 C4: registry and cell statuses.

* R-GCN: NC 12 and LP 8 runnable; MAGNN: NC Freebase + the 8 H2GB datasets runnable (ACM, DBLP, IMDB carry their
  reason) and LP 8 runnable; neither has a GC cell.
* GC option (a): every baseline GC cell is unsupported with the option-(a) reason (the adapters keep their code).
* The cell statuses equal card 8 C0 after amendments 1, 3, 4, 5 (`docs/card8_evidence/c0_inventory.md` §3, §7):
  NC 49 filled, 33 published, 89 tune, 8 probe, 1 N/A; LP 0 filled, 5 published, 88 tune, 2 probe, 1 N/A.
* No tuned search space of the new adapters carries a source-schedule key.
"""

from __future__ import annotations

import unittest
from collections import Counter

from src.baselines.protocol import GRAPH_CLASSIFICATION, LINK_PREDICTION, NODE_CLASSIFICATION


class RegistryTests(unittest.TestCase):
    def test_rgcn_and_magnn_cells(self) -> None:
        from src.baselines.registry import DATASETS, MAGNN_HGB_NC_NOT_WIRED, unsupported_reason

        for model, nc_runnable in (("rgcn", set(DATASETS[NODE_CLASSIFICATION])),
                                   ("magnn", set(DATASETS[NODE_CLASSIFICATION]) - {"acm", "dblp", "imdb"})):
            with self.subTest(model=model):
                runnable = {d for d in DATASETS[NODE_CLASSIFICATION] if unsupported_reason(model, NODE_CLASSIFICATION, d)
                            is None}
                self.assertEqual(runnable, nc_runnable)
                self.assertEqual(len(runnable), 12 if model == "rgcn" else 9)
                self.assertTrue(all(unsupported_reason(model, LINK_PREDICTION, d) is None
                                    for d in DATASETS[LINK_PREDICTION]))
        for d in ("acm", "dblp", "imdb"):
            self.assertEqual(unsupported_reason("magnn", NODE_CLASSIFICATION, d), MAGNN_HGB_NC_NOT_WIRED)

    def test_gc_option_a_removes_every_baseline_gc_cell(self) -> None:
        from src.baselines.registry import DATASETS, GC_OPTION_A_REASON, MODELS, unsupported_reason

        for spec in MODELS:
            for d in DATASETS[GRAPH_CLASSIFICATION]:
                self.assertEqual(unsupported_reason(spec.name, GRAPH_CLASSIFICATION, d), GC_OPTION_A_REASON)

    def test_cell_statuses_equal_card8_c0(self) -> None:
        from src.baselines.cell_status import cell_status
        from src.baselines.registry import DATASETS, MODELS

        nc_methods = ("gcn", "gat", "chebnetii", "nsd", "csnn", "rgcn", "han", "magnn", "sehgnn", "hinormer",
                      "pshgcn", "h2gformer", "hetero2net", "h2sgnn", "hetsheaf")
        lp_methods = ("gcn", "gat", "nsd", "csnn", "rgcn", "han", "magnn", "pshgcn", "h2gformer", "hetero2net",
                      "h2sgnn", "hetsheaf")
        self.assertEqual(set(nc_methods), {s.name for s in MODELS})
        nc = Counter(cell_status(m, NODE_CLASSIFICATION, d) for m in nc_methods for d in DATASETS[NODE_CLASSIFICATION])
        lp = Counter(cell_status(m, LINK_PREDICTION, d) for m in lp_methods for d in DATASETS[LINK_PREDICTION])
        self.assertEqual(dict(nc), {"filled": 44, "published": 38, "tune": 89, "probe": 8, "N/A": 1})  # card 17: 5 IEEE-CIS-G H2GB cells filled -> published
        self.assertEqual(dict(lp), {"published": 5, "tune": 88, "probe": 2, "N/A": 1})
        gc = Counter(cell_status(m, GRAPH_CLASSIFICATION, d) for m in nc_methods for d in DATASETS[GRAPH_CLASSIFICATION])
        self.assertEqual(gc["filled"], 24)
        self.assertNotIn("tune", gc)
        self.assertNotIn("published", gc)

    def test_new_tuned_spaces_carry_no_source_schedule_key(self) -> None:
        from src.baselines.published import source_schedule_keys
        from src.baselines.registry import DATASETS, get_adapter, unsupported_reason

        for model in ("rgcn", "magnn"):
            adapter = get_adapter(model)
            for task in (NODE_CLASSIFICATION, LINK_PREDICTION):
                for d in DATASETS[task]:
                    if unsupported_reason(model, task, d) is None:
                        with self.subTest(model=model, task=task, dataset=d):
                            self.assertEqual(source_schedule_keys(adapter.search_space(task, d)), [])


if __name__ == "__main__":
    unittest.main()


class VendorModuleLeakTests(unittest.TestCase):
    """Card 8 C4 fix: the R-GCN and MAGNN blocks leave no HGB `model` / `utils` module in `sys.modules` (a leaked
    MAGNN `utils` made HAN's `import utils` fail in one process)."""

    def test_no_hgb_method_module_survives_a_block(self) -> None:
        import sys

        import numpy as np
        import torch

        from src.baselines.adapters import magnn, rgcn

        rgcn.entity_classify_model([4], 8, 3, 2, -1, 3, 0.0)
        rgcn.lp_rgcn_model([4], 8, 3, 4, 2, 0.0)
        magnn.build_nc_model(1, 2, [[0, 1]], [4], 8, 3, 2, 8, "RotatE0", 0.0)
        magnn.build_lp_model([1, 1], 2, [[[0, 1]], [[1, 0]]], [4], 8, 2, 8, "RotatE0", 0.0)
        magnn.nc_parse([{0: "0 1", 1: "1 0"}], [{0: np.array([[1, 2, 0]]), 1: np.array([[0, 2, 1]])}],
                       np.array([0, 1]), torch.device("cpu"), None)
        leaked = sorted(name for name, module in list(sys.modules.items())
                        if "/vendor/HGB/" in str(getattr(module, "__file__", "") or "")
                        and ("/methods/MAGNN/" in str(module.__file__) or "/methods/RGCN/" in str(module.__file__)))
        self.assertEqual(leaked, [])
