"""Fixtures of the Part MB tests (card `docs/H2S-NN_partMB_card.md`). Synthetic graphs only.

`unreachable_real_data()` patches the project's data root to an empty directory for a test module's
lifetime (`setUpModule`/`tearDownModule`), so no test of the card can read real data (Part P-fix
finding: a module-level bound root read the real Freebase files).

`example_graph()` is the §2-style hand graph of `minibatch_design_v3B.md`: target paper P1 (node 0)
with authors, a field, a cited and a citing paper; `example_cells()` are two hand cells of P1.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import torch

from src.data.structures import FaceGroup, PreparedH2SData

_PATCH: list = []


def unreachable_real_data() -> None:
    import src.data.data_layout as layout

    empty = Path(tempfile.mkdtemp(prefix="mb_no_real_data_"))
    patcher = mock.patch.object(layout, "REPO_DATA_ROOT", empty)
    patcher.start()
    _PATCH.append(patcher)


def restore_real_data() -> None:
    while _PATCH:
        _PATCH.pop().stop()


# papers 0-5 (P1 = 0), authors 6-9 (featureless), fields 10-11
WRITES = [(0, 6), (0, 7), (0, 8), (1, 6), (2, 9), (3, 7), (4, 9)]   # relation 0, paper -> author
FIELD = [(0, 10), (1, 11), (3, 10), (5, 11)]                         # relation 1, paper -> field
CITES = [(0, 1), (2, 0), (1, 3), (4, 5)]                             # relation 2, paper -> paper
EDGES = [(u, v, 0) for u, v in WRITES] + [(u, v, 1) for u, v in FIELD] + \
    [(u, v, 2) for u, v in CITES]


def fact(u: int, v: int, r: int, edges=EDGES) -> int:
    return edges.index((u, v, r))


def example_graph(edges=EDGES, seed: int = 0, feature_dim: int = 4) -> PreparedH2SData:
    generator = torch.Generator().manual_seed(seed)
    num_nodes = 12
    node_type = torch.tensor([0] * 6 + [1] * 4 + [2] * 2)
    x = torch.randn(num_nodes, feature_dim, generator=generator)
    x[6:10] = 0.0  # authors: no attributes (feature dim 0)
    edge_index = torch.tensor([[u for u, _, _ in edges], [v for _, v, _ in edges]])
    edge_type = torch.tensor([r for _, _, r in edges])
    targets = torch.arange(6)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[:6] = torch.tensor([0, 1, 0, 1, 0, 1])
    train = torch.zeros(6, dtype=torch.bool)
    train[:4] = True
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[:4] = True
    return PreparedH2SData(
        x=x, y=y, node_type=node_type, node_type_names=("paper", "author", "field"),
        node_slices={"paper": (0, 6), "author": (6, 10), "field": (10, 12)},
        feature_dims={"paper": feature_dim, "author": 0, "field": feature_dim},
        edge_index=edge_index, edge_type=edge_type,
        relation_names=("paper__writes__author", "paper__has__field", "paper__cites__paper"),
        relation_tuples=(("paper", "writes", "author"), ("paper", "has", "field"),
                         ("paper", "cites", "paper")),
        train_mask=node_mask, val_mask=~node_mask & (y >= 0), test_mask=~node_mask & (y >= 0),
        target_node_type="paper", target_indices=targets, target_y=y[:6],
        target_train_mask=train, target_val_mask=~train, target_test_mask=~train,
        num_classes=2, is_multilabel=False, face_groups=(), metrics=torch.zeros(0))


def _group(nodes, facts, signs, relations, m) -> FaceGroup:
    return FaceGroup(template_name="hand", node_type_sequence=(),
                     face_node_index=torch.tensor(nodes), face_edge_index=torch.tensor(facts),
                     relation_index=torch.tensor(relations),
                     face_edge_sign=torch.tensor(signs, dtype=torch.float32),
                     path_weight=torch.tensor(m, dtype=torch.float32))


def example_cells() -> tuple[FaceGroup, ...]:
    """P1's hand cells: P1 -writes-> A6 <-writes- P(1); P1 -field-> F10 <-field- P(3) -writes-> A7."""
    two = _group([[0, 6, 1]], [[fact(0, 6, 0), fact(1, 6, 0)]], [[1.0, -1.0]], [[0, 0]], [0.5])
    three = _group([[0, 10, 3, 7]], [[fact(0, 10, 1), fact(3, 10, 1), fact(3, 7, 0)]],
                   [[1.0, -1.0, 1.0]], [[1, 1, 0]], [0.25])
    return two, three


def cells_of_paper_1() -> tuple[FaceGroup, ...]:
    """P(1) -writes-> A6 <-writes- P1: the same instance as P1's first cell, from the other end."""
    return (_group([[1, 6, 0]], [[fact(1, 6, 0), fact(0, 6, 0)]], [[1.0, -1.0]], [[0, 0]], [0.5]),)


__all__ = ["CITES", "EDGES", "FIELD", "WRITES", "cells_of_paper_1", "example_cells",
           "example_graph", "fact", "restore_real_data", "unreachable_real_data"]
