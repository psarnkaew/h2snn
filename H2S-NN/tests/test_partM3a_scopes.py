"""Part M3a, A2 (`docs/H2S-NN_partM3a_card.md`; decisions log 2026-11-13, Part MB review): the scope
builder reads a CSR incidence index built once per graph; its output is identical to the former
O(E)-per-target builder (`tests/scope_reference.py`, `v3b` a68cbcc verbatim).

- on 24 random synthetic graphs (multi-edges, self-loops, facts stored in both orientations,
  isolated nodes, a featureless type) and random target sets with their mini-batch cells, every field
  of `ScopeMembers` (every target) and of `ScopeBatch` (incl. the batch graph and its groups) is
  `torch.equal` to the reference;
- the incidence index is the per-node ascending fact list (self-loops once), built from the graph
  given: a training graph and an evaluation graph each get their own (N6), and it is built once per
  graph (cached);
Card mutation: the index built without self-loop facts -> `test_equal_to_the_reference_builder` fails.
"""

from __future__ import annotations

import dataclasses
import unittest
from collections.abc import Mapping

import torch

from src.data.scopes import (build_incidence_index, build_scope_batch, incidence_index,
                             scope_members)
from src.data.structures import PreparedH2SData
from src.data.wisp_sampler import build_wisp_anchor_cells
from tests.partMB_fixtures import restore_real_data, unreachable_real_data
from tests.scope_reference import reference_build_scope_batch, reference_scope_members

NUM_GRAPHS = 24


def setUpModule() -> None:
    unreachable_real_data()


def tearDownModule() -> None:
    restore_real_data()


def random_graph(seed: int) -> PreparedH2SData:
    """Papers (type 0, the targets), authors (type 1, featureless), fields (type 2); relations
    writes (paper -> author), has (paper -> field), cites (paper -> paper). Cites holds pairs stored
    in both orientations and self-loops; every relation holds repeated facts (multi-edges); some
    authors and fields have no fact (isolated nodes); at least one paper has a self-loop."""
    g = torch.Generator().manual_seed(seed)

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (1,), generator=g))

    papers, authors, fields = randint(4, 9), randint(3, 7), randint(2, 4)
    num_nodes = papers + authors + fields
    paper_ids = list(range(papers))
    author_ids = list(range(papers, papers + authors))
    field_ids = list(range(papers + authors, num_nodes))

    def pick(ids: list[int]) -> int:
        return ids[randint(0, len(ids) - 1)]

    edges: list[tuple[int, int, int]] = []
    for _ in range(randint(papers, 2 * papers)):
        edges.append((pick(paper_ids), pick(author_ids[:-1]), 0))  # last author: isolated
    for _ in range(randint(2, papers)):
        edges.append((pick(paper_ids), pick(field_ids[:-1] or field_ids), 1))
    for _ in range(randint(papers, 2 * papers)):
        u, v = pick(paper_ids), pick(paper_ids)
        edges.append((u, v, 2))
        if u != v and randint(0, 1):
            edges.append((v, u, 2))  # the same pair in the other orientation
    loop = pick(paper_ids)
    edges.append((loop, loop, 2))  # a self-loop on a target
    for _ in range(randint(1, 4)):  # multi-edges: repeat stored facts
        edges.append(edges[randint(0, len(edges) - 1)])
    order = torch.randperm(len(edges), generator=g).tolist()
    edges = [edges[i] for i in order]

    x = torch.randn(num_nodes, 3, generator=g)
    x[papers:papers + authors] = 0.0  # featureless type
    node_type = torch.tensor([0] * papers + [1] * authors + [2] * fields)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[:papers] = torch.randint(0, 2, (papers,), generator=g)
    train = torch.zeros(papers, dtype=torch.bool)
    train[: max(1, papers // 2)] = True
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[: max(1, papers // 2)] = True
    return PreparedH2SData(
        x=x, y=y, node_type=node_type, node_type_names=("paper", "author", "field"),
        node_slices={"paper": (0, papers), "author": (papers, papers + authors),
                     "field": (papers + authors, num_nodes)},
        feature_dims={"paper": 3, "author": 0, "field": 3},
        edge_index=torch.tensor([[u for u, _, _ in edges], [v for _, v, _ in edges]]),
        edge_type=torch.tensor([r for _, _, r in edges]),
        relation_names=("paper__writes__author", "paper__has__field", "paper__cites__paper"),
        relation_tuples=(("paper", "writes", "author"), ("paper", "has", "field"),
                         ("paper", "cites", "paper")),
        train_mask=node_mask, val_mask=~node_mask & (y >= 0), test_mask=~node_mask & (y >= 0),
        target_node_type="paper", target_indices=torch.arange(papers), target_y=y[:papers],
        target_train_mask=train, target_val_mask=~train, target_test_mask=~train,
        num_classes=2, is_multilabel=False, face_groups=(), metrics=torch.zeros(0))


def assert_same(case: unittest.TestCase, a, b, path: str) -> None:
    """Every field equal: tensors by dtype, shape and `torch.equal`; dataclasses field by field."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        case.assertTrue(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), path)
        case.assertEqual((a.dtype, tuple(a.shape)), (b.dtype, tuple(b.shape)), path)
        case.assertTrue(torch.equal(a, b), path)
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        case.assertIs(type(a), type(b), path)
        for field in dataclasses.fields(a):
            assert_same(case, getattr(a, field.name), getattr(b, field.name),
                        f"{path}.{field.name}")
    elif isinstance(a, (tuple, list)):
        case.assertEqual(len(a), len(b), path)
        for i, (u, v) in enumerate(zip(a, b)):
            assert_same(case, u, v, f"{path}[{i}]")
    elif isinstance(a, Mapping):
        case.assertEqual(list(a), list(b), path)
        for key in a:
            assert_same(case, a[key], b[key], f"{path}[{key!r}]")
    else:
        case.assertEqual(a, b, path)


def draw(graph: PreparedH2SData, targets: torch.Tensor, seed: int):
    return build_wisp_anchor_cells(graph.edge_index, graph.edge_type, graph.node_type, (),
                                   targets, 4, 3, seed=seed, chunk_size=3)


class IncidenceIndexTests(unittest.TestCase):
    def test_index_lists_each_nodes_facts_ascending(self) -> None:
        for seed in range(NUM_GRAPHS):
            graph = random_graph(seed)
            index = build_incidence_index(graph.edge_index, graph.num_nodes)
            src, dst = graph.edge_index
            self.assertEqual(index.rowptr.numel(), graph.num_nodes + 1)
            for v in range(graph.num_nodes):
                expected = ((src == v) | (dst == v)).nonzero(as_tuple=True)[0]
                self.assertTrue(torch.equal(index.facts_of(v), expected), (seed, v))

    def test_each_graph_gets_its_own_index_built_once(self) -> None:
        train, evaluation = random_graph(101), random_graph(202)  # N6: two graphs
        self.assertFalse(torch.equal(train.edge_index, evaluation.edge_index))
        first, second = incidence_index(train), incidence_index(evaluation)
        for graph, index in ((train, first), (evaluation, second)):
            assert_same(self, index, build_incidence_index(graph.edge_index, graph.num_nodes),
                        "index")
        self.assertIs(incidence_index(train), first)        # built once per graph
        self.assertIs(incidence_index(evaluation), second)
        targets = evaluation.target_indices[:3]
        cells = draw(evaluation, targets, 7)
        own = [cells.of(int(t)) for t in targets.tolist()]
        assert_same(self, build_scope_batch(evaluation, targets, own),
                    reference_build_scope_batch(evaluation, targets, own), "eval batch")
        evaluation.edge_index.add_(0)  # an in-place change of the facts invalidates the cache
        self.assertIsNot(incidence_index(evaluation), second)


class ReferenceEqualityTests(unittest.TestCase):
    def test_equal_to_the_reference_builder(self) -> None:
        loops_on_targets = 0
        for seed in range(NUM_GRAPHS):
            graph = random_graph(seed)
            src, dst = graph.edge_index
            loops_on_targets += int((src == dst).sum())
            g = torch.Generator().manual_seed(1000 + seed)
            targets_all = graph.target_indices
            cells = draw(graph, targets_all, seed)
            # every target's members with its own cells
            for t in targets_all.tolist():
                assert_same(self, scope_members(graph, t, cells.of(t)),
                            reference_scope_members(graph, t, cells.of(t)),
                            f"graph {seed} members of {t}")
            # random target sets (sizes 1 .. all) with their mini-batch cells
            for draw_index in range(3):
                size = int(torch.randint(1, targets_all.numel() + 1, (1,), generator=g))
                targets = targets_all[torch.randperm(targets_all.numel(), generator=g)[:size]]
                batch_cells = draw(graph, targets, 100 * seed + draw_index)
                own = [batch_cells.of(int(t)) for t in targets.tolist()]
                assert_same(self, build_scope_batch(graph, targets, own),
                            reference_build_scope_batch(graph, targets, own),
                            f"graph {seed} batch {draw_index}")
        self.assertGreater(loops_on_targets, 0)


if __name__ == "__main__":
    unittest.main()
