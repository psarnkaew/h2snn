"""The typed view of a TU minibatch, shared by the heterogeneous graph-classification cells.

HAN, PSHGCN, H2SGNN, Hetero2Net and H2G-former all need the graph *typed*. On the
node-classification and link-prediction datasets that means node types and relation types
both carry information. On TUDataset it does not, and getting this wrong in either
direction produces a meaningless cell.

**Node types are the features.** TU heterogeneity is synthesised by reading node labels as
node types and edge labels as relation types. But the node features *are* the one-hot node
labels -- verified directly: every row of ``batch.x`` sums to exactly 1.0, and
``batch.node_type`` reproduces its argmax. So a model that consumes ``x`` already has the
node-type information, and there is nothing left for a separate type channel to add.

**Keying relations by node-type pair does not work here.** That is what PSHGCN and H2SGNN
do natively, and what :mod:`._polynomial_lp` does on the link-prediction graphs. NCI1 has
37 node types, so type-pair keys would give up to 1,369 relation operators and a polynomial
of order K over 1,369 keys. Relations are therefore keyed by **edge type**, which is both
tractable and the unit that actually carries semantics on molecules (bond order).

So a TU graph is presented as a **single-node-type multiplex graph**: one node type whose
features are the node labels, and one relation per edge type. That is exactly the shape
YouTube and Amazon take in link prediction, and the keys follow the same rule --
palindromic two-character labels with the type-chaining test dropped, since with one node
type every composition is valid (see :mod:`._polynomial_lp`).

**What this costs, and where.** Only MUTAG has more than one edge type:

=========  ===========  ===========
dataset    node types   edge types
=========  ===========  ===========
MUTAG      7            4
PROTEINS   3            1
ENZYMES    3            1
NCI1       37           1
=========  ===========  ===========

On PROTEINS, ENZYMES and NCI1 these models see a **single relation**, so their
relation-heterogeneity mechanism contributes nothing and they reduce to their homogeneous
limit. Adapters record ``num_relation_keys`` so that is visible in the results rather than
inferred from a surprising number. It is a property of the datasets, not of the adapters:
a homogeneous model loses nothing structural on those three either, because the only
heterogeneity present is the node labels it already reads as features.
"""

from __future__ import annotations

from typing import Any

import torch

#: Two-character key for one relation. Palindromic on purpose: these operators are
#: symmetrised, hence their own transpose, and PSHGCN's ``lis_t`` reverses a key in place.
#: See :func:`._polynomial_lp._relation_label`.
def relation_key(index: int) -> str:
    if index >= 26:
        raise ValueError(
            f"relation index {index} exceeds the two-character key space the polynomial "
            "models require (adjs[j[:2]] in PSHGCN/hgb/model.py:70)"
        )
    letter = chr(ord("A") + index)
    return letter + letter


def batch_relation_operators(
    batch: Any, num_relations: int, device: torch.device
) -> dict[str, Any]:
    """Row-normalised ``D^-1 A`` per edge type over one batch's node set.

    Each relation is symmetrised and given self-loops before normalising, matching
    PSHGCN's and H2SGNN's own treatment of a same-type relation
    (PSHGCN/hgb-lp/processing.py:50-64, H2SGNN/processing.py:65-79).

    A relation absent from this batch still gets an entry -- an all-zero operator -- so the
    key set is identical across batches. Without that, a model sized by ``len(adjs)`` would
    silently change shape between minibatches.
    """

    from torch_sparse import SparseTensor, mul
    from torch_sparse import sum as sparse_sum

    num_nodes = int(batch.num_nodes)
    edge_type = getattr(batch, "edge_type", None)
    if edge_type is None:
        edge_type = torch.zeros(
            batch.edge_index.size(1), dtype=torch.long, device=batch.edge_index.device
        )

    operators: dict[str, Any] = {}
    for relation_id in range(num_relations):
        mask = edge_type == relation_id
        row, column = batch.edge_index[:, mask]
        # A + A^T, then self-loops.
        rows = torch.cat([row, column])
        columns = torch.cat([column, row])
        adjacency = SparseTensor(
            row=rows.cpu(),
            col=columns.cpu(),
            sparse_sizes=(num_nodes, num_nodes),
        ).coalesce().set_diag()

        adjacency = adjacency.fill_value(1.0)
        degree = sparse_sum(adjacency, dim=1)
        inverse = degree.pow_(-1.0)
        inverse.masked_fill_(inverse == float("inf"), 0.0)
        operators[relation_key(relation_id)] = mul(adjacency, inverse.view(-1, 1)).to(device)

    return operators


def batch_metapath_operators(
    batch: Any, num_relations: int, device: torch.device
) -> tuple[list[Any], list[str]]:
    """Length-2 relation products over one batch, the ``adjs_list`` H2SGNN and HAN consume.

    Every ordered pair of relations, with no type-chaining test: with a single node type
    the key characters are relation labels rather than types, so every composition is
    valid. The whole batch is the target block, so nothing is cropped.
    """

    from torch_sparse import matmul as torch_sparse_matmul

    operators = batch_relation_operators(batch, num_relations, torch.device("cpu"))
    products: list[Any] = []
    names: list[str] = []
    for first, first_operator in operators.items():
        for second, second_operator in operators.items():
            products.append(
                torch_sparse_matmul(first_operator, second_operator).to(device)
            )
            names.append(first + second)
    return products, names


def batch_hetero_data(batch: Any, num_relations: int) -> Any:
    """One batch as a ``HeteroData`` with a single node type and one relation per edge type.

    For the encoders that consume ``x_dict``/``edge_index_dict`` rather than operators.
    ``batch.batch`` is carried through under the node type so the shared head can still
    pool per graph.
    """

    from torch_geometric.data import HeteroData

    node_type = "node"
    edge_type = getattr(batch, "edge_type", None)
    if edge_type is None:
        edge_type = torch.zeros(
            batch.edge_index.size(1), dtype=torch.long, device=batch.edge_index.device
        )

    hetero = HeteroData()
    hetero[node_type].x = batch.x.float()
    hetero[node_type].num_nodes = int(batch.num_nodes)
    for relation_id in range(num_relations):
        mask = edge_type == relation_id
        edge_index = batch.edge_index[:, mask]
        relation = (node_type, relation_key(relation_id), node_type)
        # An empty relation is kept rather than dropped: the encoder's per-relation
        # modules are built once, and a batch that happens to omit a bond type must not
        # change the graph's metadata.
        hetero[relation].edge_index = edge_index
    return hetero


__all__ = [
    "batch_hetero_data",
    "batch_metapath_operators",
    "batch_relation_operators",
    "relation_key",
]
