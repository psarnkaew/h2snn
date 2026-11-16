"""ChebNetII on the H2GB node-classification datasets: its own large-graph scheme (`vendor/ChebNetII/ogb`, card
baselines-h2gb C4, class B).

What their ogb code does (VERIFIED-CODE, ChebNetII `ded6c18`): the graph made undirected (`ogb/processing.py:60-61`,
`to_undirected`), the symmetric-normalised adjacency without self-loops, zero-degree rows given degree 1
(`processing.py:17-24`, `sys_normalized_adjacency`), L-hat = -A_norm (`:68-69`), the Chebyshev bases
T_0 = X, T_1 = L-hat X, T_k = 2 L-hat T_{k-1} - T_{k-2} computed once before training (`:73-105`); the model
(`ogb/model.py:8-56`) mixes the bases of a batch's rows with learned coefficients, then three linear layers;
training and evaluation slice contiguous rows (`ogb/large_train.py:75-83`).

On H2GB's heterogeneous graph: every relation's edges join one homogeneous graph (nodes concatenated in H2GB's type
order) and X is the stack of the types' features, which needs one feature width for every type (MAG 128, OAG 768);
a dataset with featureless types or several widths is refused (an author decision, C4 report).
`processing.py` runs its argparse at import, so `sys_normalized_adjacency` is restated here line for line.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch


def homogeneous_inputs(data: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, int]]]:
    """(X [N, F], undirected edge_index, node slices) of H2GB's graph; refused unless every type has `x` of one
    width."""
    from torch_geometric.utils import to_undirected

    widths = {t: (None if data[t].get("x") is None else int(data[t].x.size(1))) for t in data.node_types}
    if None in widths.values() or len(set(widths.values())) != 1:
        raise ValueError(f"ChebNetII's ogb scheme needs one feature matrix; node feature widths are {widths}.")
    slices, start = {}, 0
    for node_type in data.node_types:
        count = int(data[node_type].num_nodes)
        slices[node_type] = (start, start + count)
        start += count
    edges = [torch.stack([data[e].edge_index[0] + slices[e[0]][0], data[e].edge_index[1] + slices[e[2]][0]])
             for e in data.edge_types]
    edge_index = to_undirected(torch.cat(edges, dim=1), num_nodes=start)
    x = torch.cat([data[t].x for t in data.node_types], dim=0)
    return x, edge_index, slices


def sys_normalized_adjacency(adj: sp.spmatrix) -> sp.coo_matrix:
    """`ChebNetII/ogb/processing.py:17-24`, restated (that file runs argparse at import)."""
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def chebyshev_bases(x: torch.Tensor, edge_index: torch.Tensor, K: int, rows: torch.Tensor) -> list[torch.Tensor]:
    """T_0..T_K (`processing.py:60-105`) restricted to `rows` (their per-split slices), in x's dtype."""
    n = x.size(0)
    row, col = edge_index.numpy()
    adj = sys_normalized_adjacency(sp.csr_matrix((np.ones(row.shape[0]), (row, col)), shape=(n, n)))
    adj = -1.0 * adj
    coo = adj.tocoo()
    mat = torch.sparse_coo_tensor(np.vstack((coo.row, coo.col)), torch.from_numpy(coo.data).to(x.dtype), (n, n))
    t0 = x
    t1 = torch.sparse.mm(mat, t0)
    out = [t0[rows], t1[rows]]
    for _ in range(1, K):
        t2 = 2 * torch.sparse.mm(mat, t1) - t0
        t0, t1 = t1, t2
        out.append(t2[rows])
    return out


def build_model(num_features: int, num_classes: int, params: Any) -> torch.nn.Module:
    """Their `ChebNetII` (`ogb/model.py`), defaults of `ogb/large_train.py:25-31` unless searched."""
    from src.baselines.runtime import pushd_sys_path

    args = SimpleNamespace(K=int(params.get("K", 10)), dropout=float(params.get("dropout", 0.5)))
    with pushd_sys_path("chebnetii_ogb"):
        from model import ChebNetII  # type: ignore[import-not-found]

        return ChebNetII(num_features, int(params.get("hidden", 2048)), num_classes, args)



def block_inputs(data: Any) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, tuple[int, int]]]:
    """(per-type `x` in H2GB's type order, undirected edge_index over the concatenated nodes, node slices) for the
    per-type projection of decision C4-c; refused when a type has no `x` (see `block_bases`)."""
    from torch_geometric.utils import to_undirected

    missing = [t for t in data.node_types if data[t].get("x") is None]
    if missing:
        raise ValueError(f"featureless node types {missing}: their identity block makes T_k(L)[:, block] X a dense "
                         "N x N_type matrix per k (part 2 E2 stop item)")
    slices, start = {}, 0
    for node_type in data.node_types:
        count = int(data[node_type].num_nodes)
        slices[node_type] = (start, start + count)
        start += count
    edges = [torch.stack([data[e].edge_index[0] + slices[e[0]][0], data[e].edge_index[1] + slices[e[2]][0]])
             for e in data.edge_types]
    return [data[t].x for t in data.node_types], to_undirected(torch.cat(edges, dim=1), num_nodes=start), slices


def block_bases(features: list[torch.Tensor], edge_index: torch.Tensor, slices: dict[str, tuple[int, int]], K: int,
                rows: torch.Tensor) -> list[list[torch.Tensor]]:
    """Decision C4-c: `out[t][k]` = rows of T_k(L) [X_t | 1] placed in type t's block (zeros elsewhere), so that
    T_k(L) (X_t W_t^T + 1 b_t^T) = out[t][k][:, :-1] W_t^T + out[t][k][:, -1:] b_t^T (the projection is linear)."""
    n = sum(f.size(0) for f in features)
    out = []
    for feature, (start, end) in zip(features, slices.values()):
        block = torch.zeros(n, feature.size(1) + 1, dtype=feature.dtype)
        block[start:end, :-1] = feature
        block[start:end, -1] = 1.0
        out.append(chebyshev_bases(block, edge_index, K, rows))
    return out


class ProjectedBases(torch.nn.Module):
    """HGB's per-type input projection (`TypeProjectedInput`, the ChebNetII HGB adapter's) applied after the bases:
    the batch's T_k(L) X_proj rows from the per-block bases."""

    def __init__(self, in_dims: list[int], hidden_dim: int) -> None:
        super().__init__()
        from src.baselines.encoders import TypeProjectedInput

        self.projection = TypeProjectedInput(in_dims, hidden_dim)

    def forward(self, blocks: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        K1 = len(blocks[0])
        out = []
        for k in range(K1):
            total = None
            for linear, per_k in zip(self.projection.projections, blocks):
                value = per_k[k][:, :-1] @ linear.weight.t() + per_k[k][:, -1:] * linear.bias
                total = value if total is None else total + value
            out.append(total)
        return out



def forward_inputs(data: Any) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, tuple[int, int]]]:
    """Card 1c F5 (decision G-6): per-type inputs with a sparse identity block for each featureless type (HGB's identity
    encoding; projected, it is an embedding table), the undirected edge_index and the node slices."""
    from torch_geometric.utils import to_undirected

    slices, start, features = {}, 0, []
    for node_type in data.node_types:
        count = int(data[node_type].num_nodes)
        slices[node_type] = (start, start + count)
        start += count
        x = data[node_type].get("x")
        if x is None:
            index = torch.arange(count)
            x = torch.sparse_coo_tensor(torch.stack([index, index]), torch.ones(count), (count, count)).coalesce()
        features.append(x)
    edges = [torch.stack([data[e].edge_index[0] + slices[e[0]][0], data[e].edge_index[1] + slices[e[2]][0]])
             for e in data.edge_types]
    return features, to_undirected(torch.cat(edges, dim=1), num_nodes=start), slices


class ForwardBases(torch.nn.Module):
    """Decision G-6: HGB's per-type projection first, then their Chebyshev bases (`processing.py:60-105`) of the
    projected features computed in every forward over the whole graph; returns the bases' rows of `rows`. Equal to
    the per-block precomputation by linearity (`ChebNetIIProjectionTests`)."""

    def __init__(self, in_dims: list[int], hidden_dim: int, edge_index: torch.Tensor, num_nodes: int, K: int) -> None:
        super().__init__()
        from src.baselines.encoders import TypeProjectedInput

        self.projection = TypeProjectedInput(in_dims, hidden_dim)
        coo = (-1.0 * sys_normalized_adjacency(sp.csr_matrix(
            (np.ones(edge_index.size(1)), (edge_index[0].numpy(), edge_index[1].numpy())),
            shape=(num_nodes, num_nodes)))).tocoo()
        self.register_buffer("laplacian", torch.sparse_coo_tensor(
            np.vstack((coo.row, coo.col)), torch.from_numpy(coo.data).float(), (num_nodes, num_nodes)).coalesce())
        self.K = int(K)

    def forward(self, features: list[torch.Tensor], rows: torch.Tensor) -> list[torch.Tensor]:
        t0 = self.projection(features)
        t1 = torch.sparse.mm(self.laplacian, t0)
        out = [t0[rows], t1[rows]]
        for _ in range(1, self.K):
            t2 = 2 * torch.sparse.mm(self.laplacian, t1) - t0
            t0, t1 = t1, t2
            out.append(t2[rows])
        return out


__all__ = ["ForwardBases", "ProjectedBases", "forward_inputs", "block_bases", "block_inputs", "build_model", "chebyshev_bases",
           "homogeneous_inputs", "sys_normalized_adjacency"]
