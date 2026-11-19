"""Card 7 T8 (`docs/H2S-NN_card7_card.md` V7): the timing script and the cost-vs-accuracy figure, on crafted inputs
(no real data; the real IMDB smoke is a separate lab-GPU job, `docs/card7_evidence/v7_smoke_job.sh`).

- time per epoch = the median of epochs 2-4 (None with fewer epochs); every field is written;
- the baseline path: per-epoch training time from the previous report to the first `.eval()`, the parameter count
  of the first optimizer, a stubbed `fit_cell` (no data, no metric);
- the cost-vs-accuracy combination: one point per method with both inputs, the others listed as missing, the
  central-point flag carried, JSON and PDF written.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "timing"))

import cost_vs_accuracy as cva  # noqa: E402
import time_methods as tm  # noqa: E402


class EpochTests(unittest.TestCase):
    def test_median_of_epochs_two_to_four(self) -> None:
        per_epoch = [{"train_seconds": v} for v in (9.0, 1.0, 3.0, 2.0)]
        self.assertEqual(tm.time_per_epoch(per_epoch), 2.0)               # epoch 1 (warm-up) excluded
        self.assertIsNone(tm.time_per_epoch(per_epoch[:3]))
        self.assertIsNone(tm.time_per_epoch(per_epoch[:1]))               # the 1-epoch smoke

    def test_finish_writes_every_field_and_refuses_a_missing_one(self) -> None:
        record = {f: None for f in tm.FIELDS}
        record["per_epoch"] = [{"epoch": 1, "train_seconds": 1.0, "epoch_seconds": 1.5}]
        out = tm.finish(dict(record))
        self.assertEqual(list(out)[:len(tm.FIELDS)], list(tm.FIELDS))
        self.assertEqual((out["epochs_run"], out["time_per_epoch_seconds"]), (1, None))
        del record["parameters"]
        with self.assertRaises(KeyError):
            tm.finish(record)


class BaselinePathTests(unittest.TestCase):
    def test_baseline_timing_with_a_stubbed_fit(self) -> None:
        import torch

        def fake_fit(raw_base, params, *, source, device, report):
            model = torch.nn.Linear(4, 3)                                  # 15 parameters
            torch.optim.Adam(model.parameters(), lr=1e-3)
            torch.optim.Adam(torch.nn.Linear(2, 2).parameters())          # a second optimizer: not counted
            for epoch in range(4):
                model.train()
                time.sleep(0.01 * (epoch + 1))                             # training
                model.eval()
                time.sleep(0.005)                                          # validation
                report(epoch=epoch, monitor="val_macro_f1", running_best=None, metrics={})
            return {}

        with mock.patch("src.baselines.campaign.fit_cell", fake_fit), \
                mock.patch.object(tm, "baseline_params", return_value=({"lr": 1e-3}, "central", {"rule": {}})):
            record = tm.time_baseline("gcn", "imdb", 4, "cpu", None)
        self.assertEqual(record["epochs_run"], 4)
        self.assertEqual(record["parameters"], 15)
        self.assertTrue(record["flagged"])
        for e in record["per_epoch"][1:]:
            self.assertGreater(e["epoch_seconds"], e["train_seconds"])     # validation excluded from training
        expected = sorted(e["train_seconds"] for e in record["per_epoch"][1:])[1]
        self.assertEqual(record["time_per_epoch_seconds"], expected)
        self.assertGreater(record["per_epoch"][3]["train_seconds"], record["per_epoch"][1]["train_seconds"])
        self.assertEqual(torch.nn.Module.eval.__name__, "eval")             # patches restored
        self.assertEqual(torch.optim.Optimizer.__init__.__name__, "__init__")
        self.assertEqual(set(tm.FIELDS) - set(record), set())


class CostVsAccuracyTests(unittest.TestCase):
    def test_combination_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            timing = []
            for method, seconds, flagged in (("h2snn", 2.0, False), ("gcn", 0.5, True), ("han", None, False),
                                             ("sehgnn", 1.0, False)):
                path = tmp / f"t_{method}.json"
                path.write_text(json.dumps({"method": method, "dataset": "ogbn-mag", "time_per_epoch_seconds": seconds,
                                            "peak_gpu_memory_mb": 1000.0, "parameters": 10,
                                            "configuration": "central" if flagged else "tuned",
                                            "flagged": flagged}), encoding="utf-8")
                timing.append(path)
            other = tmp / "t_imdb.json"
            other.write_text(json.dumps({"method": "h2snn", "dataset": "imdb"}), encoding="utf-8")
            timing.append(other)
            metrics = {}
            for method, mean in (("h2snn", 0.4), ("gcn", 0.3), ("han", 0.35), ("hetsheaf", 0.2)):
                path = tmp / f"m_{method}.json"
                path.write_text(json.dumps({"summary": {"metrics": {"test_macro_f1": {"mean": mean, "std": 0.01}}}}),
                                encoding="utf-8")
                metrics[method] = path
            result = cva.combine(timing, metrics)
            self.assertEqual([p["method"] for p in result["points"]], ["gcn", "h2snn"])
            gcn = result["points"][0]
            self.assertEqual((gcn["test_macro_f1_mean"], gcn["time_per_epoch_seconds"], gcn["flagged"]),
                             (0.3, 0.5, True))
            self.assertEqual(sorted(m["method"] for m in result["missing"]), ["han", "hetsheaf", "sehgnn"])
            cva.figure(result, tmp / "fig.pdf")
            self.assertGreater((tmp / "fig.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
