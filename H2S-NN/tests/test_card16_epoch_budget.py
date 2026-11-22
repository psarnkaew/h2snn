"""Card 16 part 2 (author 2026-11-22): H2GB NC loops check early stopping at their evaluations and patience counts
epochs: eval_period 4, patience 40 -> the loop stops after 10 evaluations without improvement (OAG's eval_period 2:
20). A published H2GB cell (patience None) keeps its source's rule and runs every epoch. The epoch the run stopped at
is what `StoppedEpoch` (the record's `stopped_epoch`, `campaign.fit_cell`) keeps."""

from __future__ import annotations

import unittest

import torch


class H2GBPatienceTests(unittest.TestCase):
    def _run(self, eval_period: int, patience: int | None) -> tuple[dict, list[int], int | None]:
        from src.baselines.adapters.hetero2net import train_h2gb_epochs
        from src.baselines.minibatch import MinibatchPlan
        from src.baselines.progress import StoppedEpoch

        plan = MinibatchPlan("pdns", 4, 8, 2, 500, eval_period, patience)
        current = {"epoch": -1}
        evaluated: list[int] = []

        def train_epoch(epoch: int) -> None:
            current["epoch"] = epoch

        def evaluate(loader: str) -> dict[str, float]:
            if loader != "subset":
                return {"macro_f1": 0.0}
            evaluated.append(current["epoch"])
            # improves at the evaluations of epochs 0 and 3, never after
            return {"macro_f1": 0.1 * current["epoch"] if current["epoch"] <= 3 else 0.2}

        stopped = StoppedEpoch(None)
        record = train_h2gb_epochs(model=torch.nn.Linear(1, 1), plan=plan, train_epoch=train_epoch, evaluate=evaluate,
                                   subset_loader="subset", subset_ids=torch.arange(5), val_loader="val",
                                   test_loader="test", report=stopped)
        return record, evaluated, stopped.epoch

    def test_patience_counts_epochs_over_evaluation_points(self) -> None:
        from src.baselines.minibatch import minibatch_plan

        self.assertEqual(minibatch_plan("pdns", 256).patience, 40)  # from the budget table
        record, evaluated, stopped = self._run(eval_period=4, patience=40)
        self.assertEqual(record["best_epoch"], 3.0)
        self.assertEqual(len([e for e in evaluated if e > 3]), 10)  # 7, 11, ..., 43
        self.assertEqual((evaluated[-1], stopped), (43, 43))
        _, evaluated, _ = self._run(eval_period=2, patience=40)  # OAG
        self.assertEqual(len([e for e in evaluated if e > 3]), 20)  # 5, 7, ..., 43
        _, evaluated, stopped = self._run(eval_period=4, patience=None)  # a published H2GB cell
        self.assertEqual((evaluated[-1], stopped), (499, 499))


if __name__ == "__main__":
    unittest.main()
