"""Card 2 D4b mutation (in memory, nothing on disk changes): the trainer's scopes are built on the FULL graph (H2GB's
graph of the task, held-out pairs kept) instead of the bundle's message graph. The held-out route test
(tests/test_large_lp_h2snn.py: HeldOutRouteTests) must fail.   python ../docs/dryrun_evidence/d4b_mutation.py  (H2S-NN/)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path.cwd()))

import src.trainers.large_lp_trainer as large  # noqa: E402
import tests.test_large_lp_h2snn as t  # noqa: E402
from src.data.large_lp_h2snn import prepare_large_link_data  # noqa: E402
from src.trainers.minibatch_lp import LinkScopes  # noqa: E402

t.setUpModule()
data, files = t.synthetic_task()


def full_graph_scopes(model, lp, cfg):
    from types import SimpleNamespace

    import torch

    rel = ("user", "has_friend", "user")
    empty = torch.zeros(2, 0, dtype=torch.long)
    full = SimpleNamespace(task=t.TASK, graph=data, node_type="user", train_pos=data[rel].edge_index,
                           val_pos=empty, val_neg=empty, test_pos=empty, test_2hop=empty, test_random=empty)
    graph = prepare_large_link_data(full, cfg.data.fact_layer).graph  # the full graph in H2S-NN's layout
    return LinkScopes(model, graph, cfg.data.wisp_walks_per_anchor, cfg.data.wisp_max_length)


with mock.patch.object(large, "link_scopes", full_graph_scopes):
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(t.HeldOutRouteTests))
t.tearDownModule()
caught = bool(result.failures or result.errors)
print("MUTATION scopes on the full graph:", "caught (test fails)" if caught else "NOT caught")
for _, text in result.failures + result.errors:
    print(text.strip().splitlines()[-1][:300])
sys.exit(0 if caught else 1)
