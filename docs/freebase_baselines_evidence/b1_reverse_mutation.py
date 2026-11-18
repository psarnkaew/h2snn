"""Mutation for the reverse completion (author decision 2026-11-18): `plan_relations` without the transpose (parallel
relations still merged); the toy-graph test must fail, then pass again with the real function."""
import sys
import unittest

import torch

sys.path.insert(0, ".")
import src.baselines.adapters._hgb_generic as g  # noqa: E402

REAL = g.plan_relations


def no_transpose(relations):
    grouped = {}
    for source, destination, edge_index in relations:
        grouped.setdefault((source, destination), []).append(edge_index.long())
    return {pair: g._unique_edges(torch.cat(parts, dim=1)) for pair, parts in grouped.items()}


NAME = "tests.test_freebase_baselines.RelationPlanTests.test_one_direction_toy_graph_gets_both_directions_and_a_symmetric_same_type_key"
for label, fn, want_ok in [("mutant (transpose dropped)", no_transpose, False), ("real", REAL, True)]:
    g.plan_relations = fn
    result = unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromName(NAME))
    print(f"{label}: {'pass' if result.wasSuccessful() else 'FAIL'} (expected {'pass' if want_ok else 'FAIL'})")
    assert result.wasSuccessful() == want_ok
print("mutation check OK")
