"""Part 2 E3 (slim mode): class-C adapters (NSD, CSNN) on H2GB's `hgt` batches (D1 second branch, S1), synthetic graph
only: one training step on a sampled batch, rebound to the batch's HGB reduction, is finite and moves the parameters.
H2SGNN is a stop item (part 2 E0, S4b). The real data root is unreachable.
"""

from __future__ import annotations

import unittest

import torch

from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.test_baselines_h2gb_classa import _first_batch, _one_step_moves, _Params
from tests.test_baselines_minibatch import synthetic_hetero


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


class ClassCStepTests(unittest.TestCase):
    def _step(self, module_name: str, params: dict) -> None:
        import importlib

        from src.baselines.minibatch import batch_type_inputs, h2gb_input_dims, hgb_homogeneous_batch

        module = importlib.import_module(f"src.baselines.adapters.{module_name}")
        data = synthetic_hetero()
        node_types, counts, dims = h2gb_input_dims(data)
        torch.manual_seed(0)
        model = module.build_h2gb_model(dims, 3, _Params(params), torch.device("cpu"))
        batch = _first_batch(data)
        size = int(batch["a"].batch_size)

        def forward():
            inputs = batch_type_inputs(batch, node_types, counts)
            _, edge_index, offsets = hgb_homogeneous_batch(batch, node_types)
            out = module.h2gb_forward(model, inputs, edge_index, sum(x.size(0) for x in inputs))
            return out[offsets["a"][0]:offsets["a"][0] + size]

        _one_step_moves(self, model, forward, batch["a"].y[:size])

    def test_nsd_step(self) -> None:
        self._step("nsd", {"hidden_channels": 4, "d": 2, "layers": 2, "dropout": 0.0, "sheaf_model": "DiagSheaf"})

    def test_csnn_step(self) -> None:
        self._step("csnn", {"hidden_channels": 4, "d": 2, "layers": 2, "dropout": 0.0})


if __name__ == "__main__":
    unittest.main()
