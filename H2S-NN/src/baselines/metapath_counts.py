"""Part M3b B7: the meta-path (type-sequence) sets SeHGNN and PSHGCN build on a schema, by their own code, without
propagating features (decisions log 2026-11-13, "Meta-paths on H2GB", M-path 3, 4).

* SeHGNN (`vendor/SeHGNN/hgb/utils.py`, commit `e92bd37`): the feature keys are made by
  `hg_propagate_feat_dgl(g, tgt, num_hops, num_hops + 1, [])` (`utils.py:52-86`; `main.py` sets each type's
  own key to its features, `utils.py:327-330`), the label keys by `hg_propagate_sparse_pyg(adjs, tgt,
  num_label_hops, num_label_hops + 1, [], prop_feats=False)` (`utils.py:89-`). Both are called here on the
  *schema graph*: one node per node type, one edge per relation, 1-dimensional features; the keys they leave on
  the target are exactly the keys they build on the full graph (a key exists iff the type sequence is a walk of
  the schema), and nothing of the real data is propagated.
* PSHGCN (`vendor/PSHGCN/hgb/processing.py:109-129`, commit `f87406c`): the polynomial terms `lis[k]` (chains of
  2-letter relation keys with matching types, k = 1..K) and their transposes `lis_t[k]`. That block is not a
  function in their code; it is read from their file (AST) and executed on the relation keys, unchanged.
Node types are given as single letters, as both codes name them.
"""

from __future__ import annotations

import ast
import contextlib
import io
from argparse import Namespace

from src.baselines.runtime.vendor_paths import pushd_sys_path, vendor_path


def sehgnn_feature_keys(types: str, relations: list[tuple[str, str]], target: str, num_hops: int) -> list[str]:
    """SeHGNN's feature keys on `target` for `num_hops` (types: one letter each; relations: (src, dst))."""
    import dgl
    import torch

    graph = dgl.heterograph({(s, s + d, d): ([0], [0]) for s, d in relations},
                            num_nodes_dict={t: 1 for t in types})
    for node_type in graph.ntypes:
        graph.nodes[node_type].data[node_type] = torch.ones(1, 1)
    with pushd_sys_path("sehgnn", "sehgnn_data"), contextlib.redirect_stdout(io.StringIO()):
        from utils import hg_propagate_feat_dgl  # type: ignore[import-not-found]

        graph = hg_propagate_feat_dgl(graph, target, num_hops, num_hops + 1, [], echo=False)
    return sorted(graph.nodes[target].data.keys())


def sehgnn_label_keys(types: str, relations: list[tuple[str, str]], target: str, num_label_hops: int) -> list[str]:
    """SeHGNN's label-propagation keys (meta-path adjacencies ending at `target`)."""
    import torch
    from torch_sparse import SparseTensor

    adjs = {s + d: SparseTensor(row=torch.tensor([0]), col=torch.tensor([0]), value=torch.ones(1),
                                sparse_sizes=(1, 1)) for s, d in relations}
    # SeHGNN keys an adjacency `dst+src` (the row type first): the relation s -> d is stored as `ds`
    adjs = {key[1] + key[0]: adj for key, adj in adjs.items()}
    with pushd_sys_path("sehgnn", "sehgnn_data"), contextlib.redirect_stdout(io.StringIO()):
        from utils import hg_propagate_sparse_pyg  # type: ignore[import-not-found]

        out = hg_propagate_sparse_pyg(adjs, target, num_label_hops, num_label_hops + 1, [], prop_feats=False,
                                      echo=False)
    return sorted(out.keys())


def _pshgcn_block() -> ast.Module:
    source = (vendor_path("pshgcn") / "processing.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_dataset")
    body = function.body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and getattr(node.targets[0], "id", None) == "lis")
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.Return))
    return ast.Module(body=body[start:end], type_ignores=[])


def pshgcn_terms(relation_keys: list[str], K: int) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """PSHGCN's `lis` and `lis_t` for adjacency keys `relation_keys` (2 letters each) and order `K`."""
    namespace = {"adjs": {key: None for key in relation_keys}, "args": Namespace(K=K)}
    exec(compile(_pshgcn_block(), "<PSHGCN processing.py:109-129>", "exec"), namespace)  # noqa: S102
    return namespace["lis"], namespace["lis_t"]


__all__ = ["pshgcn_terms", "sehgnn_feature_keys", "sehgnn_label_keys"]
