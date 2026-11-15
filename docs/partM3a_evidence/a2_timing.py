"""M3a-A2 cost (informational, not a gate): time per target of the reference builder (O(E) per target)
and of the CSR builder on one synthetic graph with 10^6 facts, CPU. Run from H2S-NN/:

    python ../docs/partM3a_evidence/a2_timing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))

from src.data.scopes import build_incidence_index, scope_members  # noqa: E402
from src.data.structures import PreparedH2SData  # noqa: E402
from tests.scope_reference import reference_scope_members  # noqa: E402

NUM_NODES, NUM_FACTS = 100_000, 1_000_000
g = torch.Generator().manual_seed(0)
edge_index = torch.randint(0, NUM_NODES, (2, NUM_FACTS), generator=g)
y = torch.zeros(NUM_NODES, dtype=torch.long)
mask = torch.zeros(NUM_NODES, dtype=torch.bool)
graph = PreparedH2SData(
    x=torch.zeros(NUM_NODES, 1), y=y, node_type=torch.zeros(NUM_NODES, dtype=torch.long),
    node_type_names=("n",), node_slices={"n": (0, NUM_NODES)}, feature_dims={"n": 1},
    edge_index=edge_index, edge_type=torch.zeros(NUM_FACTS, dtype=torch.long),
    relation_names=("n__r__n",), relation_tuples=(("n", "r", "n"),),
    train_mask=mask, val_mask=mask, test_mask=mask, target_node_type="n",
    target_indices=torch.arange(NUM_NODES), target_y=y, target_train_mask=mask,
    target_val_mask=mask, target_test_mask=mask, num_classes=2, is_multilabel=False,
    face_groups=(), metrics=torch.zeros(0))
targets = torch.randint(0, NUM_NODES, (200,), generator=g).tolist()
print(f"graph: {NUM_NODES} nodes, {NUM_FACTS} facts; {len(targets)} targets, no cells; "
      f"torch {torch.__version__}, threads {torch.get_num_threads()}")

start = time.perf_counter()
index = build_incidence_index(graph.edge_index, graph.num_nodes)
build = time.perf_counter() - start
print(f"CSR index build (once per graph): {build * 1e3:.1f} ms")

start = time.perf_counter()
for t in targets:
    new = scope_members(graph, t, (), index)
csr = (time.perf_counter() - start) / len(targets)

reference_targets = targets[:50]
start = time.perf_counter()
for t in reference_targets:
    old = reference_scope_members(graph, t, ())
reference = (time.perf_counter() - start) / len(reference_targets)

same = all(torch.equal(scope_members(graph, t, (), index).facts, reference_scope_members(graph, t, ()).facts)
           for t in reference_targets[:10])
print(f"time per target: reference (O(E)) {reference * 1e3:.3f} ms over {len(reference_targets)} targets; "
      f"CSR {csr * 1e3:.3f} ms over {len(targets)} targets; ratio {reference / csr:.0f}x; "
      f"facts equal on 10 targets: {same}")
