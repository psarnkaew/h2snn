"""The generic branch of decision F-1 for PSHGCN and H2SGNN (card 3 Part B; decisions F-1, G-2, G-3; B0 record
`docs/freebase_baselines_evidence/b0_preconditions.md`).

Their HGB code builds the relation dictionary per dataset by name (`PSHGCN/hgb/processing.py:80-107`,
`H2SGNN/processing.py:85-123`); it has no Freebase and no H2GB branch. The ACM / IMDB / AMiner pattern, from any schema:

* **one operator per (source type, destination type)**, keyed by the two type letters (`'PA'`: rows paper, columns
  author), the source type's keys first (their ACM order: `PP, PA, PC, PK, AP, CP, KP`, `processing.py:87-95`);
* **parallel relations between the same ordered type pair are merged** (union of the edges; C4-b), as ACM's `PP` is
  the union of `cite` and `ref` (`processing.py:61-67`);
* **missing reverses are completed** (author decision 2026-11-18, B0): a cross-type relation whose reverse type pair is
  not stored gets its transpose as the reverse key; a same-type key that is not symmetric gets its transpose added.
  A no-op where every reverse is stored (ACM, IMDB, H2GB's `ToUndirected` graphs); on HGB Freebase (36 relations, each
  stored in one direction) it is SeHGNN's Freebase construction (`SeHGNN/hgb/utils.py:446, 461-467`);
* **same-type keys get self-loops** (`set_diag`, as ACM's `PP`), then every operator is row-normalised `D^-1 A` over
  the flattened node space (`processing.py:69-77`).

PSHGCN takes `lis` / `lis_t` from the keys (their rule, `processing.py:109-128`, = `_polynomial_lp.build_term_lists`
with type chaining); H2SGNN takes the target-X-target products cropped to the target block (`processing.py:145-158`).

The decoupled part (PSHGCN on H2GB NC and large LP; class B, card Part B "decoupled: precompute per node"): their
ogbn-mag scheme (`PSHGCN/ogbn-mag/processing.py:14-64`, `utils.py:107-132`, `model.py:48-120`) on the dataset's
relations: target named `P` (their propagation reads `P`, `utils.py:119, 124`), every relation and its completed reverse
as a DGL edge type, their `propagate` (mean = row-normalised), `expander` over the dataset's keys (their function with
the MAG key list `processing.py:15` as an argument), their model with each type's input width (S3 (b), `_pshgcn_mag`)
and **no label part** (G-2): their forward without the label lines (`model.py:105-117`). Stated deviations.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any

import numpy as np
import torch

Relation = tuple[str, str, torch.Tensor]  # (source type, destination type, type-local edge_index [2, E])


# ------------------------------------------------------------------------------------------- relation plan
def _unique_edges(edge_index: torch.Tensor) -> torch.Tensor:
    return torch.unique(edge_index, dim=1) if edge_index.numel() else edge_index.reshape(2, 0)


def plan_relations(relations: list[Relation]) -> dict[tuple[str, str], torch.Tensor]:
    """One edge set per ordered type pair: parallel relations merged, missing reverses completed (module docstring).
    Order: stored pairs in order of first appearance, completed reverses after them."""
    grouped: dict[tuple[str, str], list[torch.Tensor]] = {}
    for source, destination, edge_index in relations:
        grouped.setdefault((source, destination), []).append(edge_index.long())
    plan = {pair: _unique_edges(torch.cat(parts, dim=1)) for pair, parts in grouped.items()}
    stored = list(plan.items())
    for (source, destination), edges in stored:
        if source == destination:
            symmetric = _unique_edges(torch.cat([edges, edges.flip(0)], dim=1))
            if symmetric.size(1) != edges.size(1):
                plan[(source, destination)] = symmetric
        elif (destination, source) not in plan:
            plan[(destination, source)] = _unique_edges(edges.flip(0))
    return plan


def type_letters(node_types: list[str], target: str | None = None, target_letter: str | None = None
                 ) -> dict[str, str]:
    """One distinct capital per node type: `target_letter` for the target if given, else the first letter of the name
    (else the next unused capital)."""
    letters: dict[str, str] = {}
    used: set[str] = set()
    if target is not None and target_letter is not None:
        letters[target], used = target_letter, {target_letter}
    for node_type in node_types:
        if node_type in letters:
            continue
        first = str(node_type)[:1].upper()
        letter = first if first in string.ascii_uppercase and first not in used else next(
            c for c in string.ascii_uppercase if c not in used)
        letters[node_type] = letter
        used.add(letter)
    return letters


def ordered_types(node_types: list[str], target: str) -> list[str]:
    """The target type first (their target block starts at row 0, `H2SGNN/processing.py:145-158`), the rest in order."""
    return [target] + [t for t in node_types if t != target]


# ------------------------------------------------------------------------------------------- HGB-style operators
def hgb_operators(plan: dict[tuple[str, str], torch.Tensor], types: list[str], counts: dict[str, int],
                  letters: dict[str, str], target: str) -> dict[str, Any]:
    """Row-normalised `SparseTensor` per key over the flattened node space (`types` order), as their
    `processing.py:61-77` builds them; the target's keys first (module docstring)."""
    from torch_sparse import SparseTensor, mul
    from torch_sparse import sum as sparsesum

    offsets, total = {}, 0
    for node_type in types:
        offsets[node_type] = total
        total += int(counts[node_type])
    order = [p for p in plan if p[0] == target] + [p for p in plan if p[0] != target]
    adjs: dict[str, Any] = {}
    for source, destination in order:
        edges = plan[(source, destination)]
        adj = SparseTensor(row=edges[0] + offsets[source], col=edges[1] + offsets[destination],
                           sparse_sizes=(total, total))
        if source == destination:  # as their ACM PP: coalesce, set_diag (processing.py:61-67)
            adj = adj.coalesce().set_diag()
        adj = adj.fill_value(1.)
        deg = sparsesum(adj, dim=1)
        deg_inv = deg.pow_(-1.0)
        deg_inv.masked_fill_(deg_inv == float("inf"), 0.)
        adjs[letters[source] + letters[destination]] = mul(adj, deg_inv.view(-1, 1))
    return adjs


def h2sgnn_metapaths(adjs: dict[str, Any], target_letter: str, target_count: int) -> tuple[list[str], list[Any]]:
    """Their generic branch (`H2SGNN/processing.py:145-158`): every key pair `(i, j)` with `i[-1] == j[0]`,
    `i[0] == target`, `j[1] == target`, the product cropped to the target block (rows and columns <= target_max)."""
    from torch_sparse import SparseTensor
    from torch_sparse import matmul as torch_sparse_matmul

    target_max = int(target_count) - 1
    names, operators = [], []
    for i in adjs.keys():
        for j in adjs.keys():
            if i[-1] == j[0] and i[0] == target_letter and j[1] == target_letter:
                temp = torch_sparse_matmul(adjs[i], adjs[j])
                row, col, value = temp.coo()
                mask = (row <= target_max) & (col <= target_max)
                operators.append(SparseTensor(row=row[mask], col=col[mask], value=value[mask],
                                              sparse_sizes=(target_max + 1, target_max + 1)))
                names.append(i + j)
    return names, operators


def sparse_identity(n: int) -> torch.Tensor:
    """Their `torch.eye(count)` for a featureless type (`processing.py:17-21`) as a sparse COO matrix: the same
    product with a `Linear`, without the dense n x n matrix (Freebase: 180,098 nodes, ~37 GB dense)."""
    index = torch.arange(int(n)).repeat(2, 1)
    return torch.sparse_coo_tensor(index, torch.ones(int(n)), (int(n), int(n))).coalesce()


# ------------------------------------------------------------------------------------------- HGB raw (Freebase)
def read_hgb_raw(vendor: str, raw_dir: Path) -> dict[str, Any]:
    """Their `data_loader` on an HGB directory (no dense features): node types (HGB type ids as strings, target 0
    first), counts, the stored relations in their link order (type-local), labels as their `load_dataset` builds them
    (`processing.py:24-28`), HGB's test rows, and the attribute matrices (None if featureless)."""
    from src.baselines.runtime import pushd_sys_path

    with pushd_sys_path(vendor):
        from dataset_loader import data_loader  # type: ignore[import-not-found]

        dl = data_loader(str(raw_dir))
    types = [str(t) for t in sorted(dl.nodes["count"])]
    counts = {str(t): int(dl.nodes["count"][t]) for t in dl.nodes["count"]}
    shift = {str(t): int(dl.nodes["shift"][t]) for t in dl.nodes["shift"]}
    relations: list[Relation] = []
    for link_type, matrix in dl.links["data"].items():
        source, destination = (str(t) for t in dl.links["meta"][link_type])
        coo = matrix.tocoo()
        edges = torch.stack([torch.as_tensor(coo.row, dtype=torch.long) - shift[source],
                             torch.as_tensor(coo.col, dtype=torch.long) - shift[destination]])
        relations.append((source, destination, edges))
    labels = torch.LongTensor(dl.labels_train["data"] + dl.labels_test["data"]).argmax(dim=1)
    return {"types": types, "counts": counts, "relations": relations, "labels": labels,
            "num_classes": int(dl.labels_test["num_classes"]),
            "test_idx": np.nonzero(dl.labels_test["mask"])[0],
            "attr": {str(t): dl.nodes["attr"][t] for t in dl.nodes["count"]}}


def hgb_target_rows(raw_dir: Path, target_type: int = 0) -> torch.Tensor:
    """`rows[k]` = their index of the bundle's target row `k`. Our bundle numbers each type's nodes in `node.dat` file
    order (PyG `HGBDataset`, `hgb_dataset.py:141-143`); their `data_loader` by HGB id minus the type's shift. HGB
    Freebase's `node.dat` is not sorted (61,021 inversions), so the two orders differ there."""
    ids = []
    with (Path(raw_dir) / "node.dat").open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if int(parts[2]) == target_type:
                ids.append(int(parts[0]))
    rows = torch.tensor(ids, dtype=torch.long)
    return rows - rows.min()


def project_bundle_split(bundle: Any, raw: dict[str, Any], rows: torch.Tensor, *, vendor: str
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The bundle's train / validation / test rows in their index space (`rows`), after checking that their labels
    equal the bundle's on every labelled row and that their test rows are the bundle's (as `_hgb_vendor.verify_labels`
    and `canonical_split`; HGB Freebase leaves most books unlabelled, where their label array is all-zero)."""
    data = bundle.data
    n = int(data.target_y.shape[0])
    if rows.numel() != n or torch.unique(rows).numel() != n:
        raise RuntimeError(f"{vendor}: {rows.numel()} target rows in node.dat, the bundle has {n}")
    labelled = (data.target_train_mask | data.target_val_mask | data.target_test_mask).nonzero().view(-1)
    theirs = raw["labels"].detach().cpu()
    if not torch.equal(theirs[rows[labelled]].long(), data.target_y[labelled].long().cpu()):
        raise RuntimeError(f"{vendor}'s labels for {bundle.dataset} differ from the project bundle's on labelled rows")
    split = [rows[m.nonzero().view(-1).cpu()] for m in (data.target_train_mask, data.target_val_mask,
                                                         data.target_test_mask)]
    if not torch.equal(torch.as_tensor(raw["test_idx"], dtype=torch.long).sort().values, split[2].sort().values):
        raise RuntimeError(f"{vendor} and the bundle disagree on the {bundle.dataset} test rows")
    return split[0], split[1], split[2]


def hgb_features(raw: dict[str, Any]) -> list[torch.Tensor]:
    """Per type in `raw["types"]` order: the attributes, or the sparse identity for a featureless type."""
    out = []
    for node_type in raw["types"]:
        attr = raw["attr"][node_type]
        out.append(sparse_identity(raw["counts"][node_type]) if attr is None else torch.FloatTensor(attr))
    return out


def hgb_generic_inputs(raw: dict[str, Any], letters: dict[str, str] | None = None) -> dict[str, Any]:
    """The generic branch on an HGB release whose target is type 0: operators, PSHGCN-free pieces, H2SGNN's
    meta-paths. Letters default to the HGB type ids (`'0'`..`'7'`, as SeHGNN's Freebase keys)."""
    types = raw["types"]
    letters = letters or {t: t for t in types}
    plan = plan_relations(raw["relations"])
    adjs = hgb_operators(plan, types, raw["counts"], letters, types[0])
    names, operators = h2sgnn_metapaths(adjs, letters[types[0]], raw["counts"][types[0]])
    return {"adjs": adjs, "metapath_names": names, "metapath_operators": operators}


# ------------------------------------------------------------------------------------------- H2GB / batches
def hetero_relations(data: Any) -> list[Relation]:
    """(source, destination, edge_index) of every edge type of a PyG `HeteroData` (incl. empty ones)."""
    return [(s, d, data[s, r, d].edge_index) for s, r, d in data.edge_types]


def hetero_counts(data: Any) -> dict[str, int]:
    return {t: int(data[t].num_nodes) for t in data.node_types}


def h2sgnn_hetero_operators(data: Any, target: str) -> tuple[list[str], list[Any]]:
    """H2SGNN's meta-path operators on a PyG `HeteroData` (a sampled batch or a message graph), target block = every
    target-type row of `data`, in `data`'s row order."""
    types = ordered_types(list(data.node_types), target)
    letters = type_letters(types)
    adjs = hgb_operators(plan_relations(hetero_relations(data)), types, hetero_counts(data), letters, target)
    return h2sgnn_metapaths(adjs, letters[target], int(data[target].num_nodes))


def h2sgnn_metapath_names(node_types: list[str], edge_types: list[tuple[str, str, str]], target: str) -> list[str]:
    """The meta-path names `h2sgnn_hetero_operators` gives on any graph of this schema (they depend on the type pairs
    only: `plan_relations` keys every stored pair and completes a missing cross-type reverse; a same-type pair stays one
    key), without building an operator."""
    types = ordered_types(list(node_types), target)
    letters = type_letters(types)
    pairs: list[tuple[str, str]] = []
    for source, _, destination in edge_types:
        if (source, destination) not in pairs:
            pairs.append((source, destination))
    for source, destination in list(pairs):
        if source != destination and (destination, source) not in pairs:
            pairs.append((destination, source))
    order = [p for p in pairs if p[0] == target] + [p for p in pairs if p[0] != target]
    keys = [letters[s] + letters[d] for s, d in order]
    t = letters[target]
    return [i + j for i in keys for j in keys if i[-1] == j[0] and i[0] == t and j[1] == t]


# ------------------------------------------------------------------------------------------- PSHGCN decoupled
def expander(keys: list[str], K: int) -> tuple[list[str], int]:
    """Their `expander` (`PSHGCN/ogbn-mag/processing.py:14-64`) with the relation keys as an argument instead of
    MAG's list (`:15`); the target is `P`."""
    lis: dict[int, list[str]] = {}
    for k in range(1, K + 1):
        lis[k] = []
        if k == 1:
            for i in keys:
                lis[k].append(i)
        else:
            for i in lis[k - 1]:
                for j in keys:
                    if i[-1] == j[0]:
                        lis[k].append(i + j)
    lis_t: dict[int, list[str]] = {}
    for k, v in lis.items():
        lis_t[k] = []
        for i in v:
            temp = i[-2:][::-1]
            i = i[:-2]
            while len(i):
                temp += i[-2:][::-1]
                i = i[:-2]
            lis_t[k].append(temp)
    gi = ["10"]
    index = 11
    for v in lis.values():
        for i in v:
            gi.append(str(index) + i)
            index += 1
    gi_t = ["10"]
    index = 11
    for v in lis_t.values():
        for i in v:
            gi_t.append(str(index) + i)
            index += 1
    res = []
    for i in gi_t:
        for j in gi:
            if len(i) == 2:
                if len(j) == 2 or j[2] == "P":
                    res.append(i + j)
            else:
                coe = i[:2] + j[:2]
                if len(j) == 2 and i[2] == "P":
                    res.append(coe + i[2:])
                elif len(j) > 2 and i[-1] == j[2] and i[2] == "P":
                    res.append(coe + i[2:] + j[2:])
    return res, len(gi)


#: Their ogbn-mag default order (`main.py:198`: `--K 2`); `expander(K // 2)` (`processing.py:159`).
DECOUPLED_K = 2
#: Their random-input width for a type without features (`ogbn-mag/processing.py:193-205`, inputs rule B).
RANDOM_INPUT_WIDTH = 256


def decoupled_inputs(data: Any, target: str, seed: int, K: int = DECOUPLED_K) -> dict[str, Any]:
    """Their feature part on `data` (an H2GB graph or a large-LP message graph): letters (target `P`), the DGL graph
    of the planned relations, each type's `x` (or their random-uniform(-0.5, 0.5) 256-wide input if featureless),
    their `propagate(g, K)`, the target's features per key, the input widths, `expander(K // 2)`."""
    import dgl

    from src.baselines.adapters._pshgcn_mag import pshgcn_mag_modules

    types = ordered_types(list(data.node_types), target)
    letters = type_letters(types, target, "P")
    plan = plan_relations(hetero_relations(data))
    edges = {(letters[s], f"{letters[s]}-{letters[d]}", letters[d]): (e[0], e[1]) for (s, d), e in plan.items()}
    g = dgl.heterograph(edges, {letters[t]: int(data[t].num_nodes) for t in types})
    generator = torch.Generator().manual_seed(int(seed))
    widths = {}
    for node_type in types:
        x = data[node_type].get("x")
        if x is None:
            x = torch.rand(int(data[node_type].num_nodes), RANDOM_INPUT_WIDTH, generator=generator) - 0.5
        g.nodes[letters[node_type]].data[letters[node_type]] = x.float()
        widths[letters[node_type]] = int(x.size(1))
    keys = [f"{letters[s]}{letters[d]}" for s, d in plan]
    with pshgcn_mag_modules() as modules:
        g = modules.utils.propagate(g, K)
        feats = {k: g.nodes["P"].data.pop(k) for k in list(g.nodes["P"].data.keys())}
        modules.utils.propagate_clean(g)
    expanded, coe_num = expander(keys, int(K / 2))
    return {"feats": feats, "widths": widths, "expander": expanded, "coe_num": coe_num, "letters": letters,
            "plan": plan}


def build_decoupled_model(inputs: dict[str, Any], out_dim: int, params: Any) -> torch.nn.Module:
    """Their ogbn-mag `PSHGCN` with each type's input width (`_pshgcn_mag.build_model`, S3 (b)) and no label keys."""
    from src.baselines.adapters._pshgcn_mag import build_model, pshgcn_mag_modules

    with pshgcn_mag_modules():
        widths = dict(inputs["widths"])
        widths = {"P": widths.pop("P"), **widths}
        return build_model(widths, int(params.get("emb_dim", 512)), int(params.get("hidden_x", 1024)), 512, out_dim,
                           [], int(params.get("layers_x", 3)), 2, inputs["coe_num"], inputs["expander"],
                           float(params.get("dropout", 0.5)), float(params.get("input_drop", 0.1)), False)


def decoupled_forward(model: torch.nn.Module, feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Their forward (`ogbn-mag/model.py:96-118`) without the label part (G-2): the label-key lines and
    `x + lin_l(lx)` are left out."""
    init_x = model.input_dropout(feats["P"])
    feats = {k: model.embedings_x[k[-1]](model.input_dropout(v)) for k, v in feats.items()}
    x = model.W_x[0] * model.W_x[0] * feats["P"]
    for i in range(1, len(model.expander)):
        w1 = int(model.expander[i][:2]) - 10
        w2 = int(model.expander[i][2:4]) - 10
        x = x + model.W_x[w1] * model.W_x[w2] * feats[model.expander[i][4:]]
    x = x + model.lin_residue(init_x)
    x = model.dropout(model.prelu(x))
    return model.lin_x(x)


__all__ = ["DECOUPLED_K", "RANDOM_INPUT_WIDTH", "build_decoupled_model", "decoupled_forward", "decoupled_inputs",
           "expander", "h2sgnn_hetero_operators", "h2sgnn_metapath_names", "h2sgnn_metapaths", "hetero_counts", "hetero_relations",
           "hgb_features", "hgb_generic_inputs", "hgb_operators", "ordered_types", "plan_relations", "read_hgb_raw",
           "hgb_target_rows", "project_bundle_split", "sparse_identity", "type_letters"]
