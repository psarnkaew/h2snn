"""PSHGCN's ogbn-mag model with the input width of each node type from the dataset (card 1c F2; author decision S3 (b)).

Their `PSHGCN` (`vendor/PSHGCN/ogbn-mag/model.py:48-120`) hard-codes MAG's widths: the target type `P` 128, every other
type 256 (`embedings_x`, `:57-62`) and `lin_residue` 128 (`:68`). `build_model` constructs their class unchanged and
rebuilds only those width-bearing modules with the same constructor (`MLP(in, emb_dim, emb_dim, 2, dropout, bias)`,
`nn.Linear(in, emb_dim, bias=False)` with their xavier init); propagation, coefficients, label part, loss and optimiser
are theirs. The forward reads the target as letter `P` and each type by its letter (`:96-107`), so the adapter names
the target type `P`.

Card 1d H1 (author decision D1-b): the label part runs with `sparse_tools` as a vendor clone (`vendor/sparse_tools`,
commit `0596f00`, CPU extensions built in place by `scripts/setup/build_sparse_tools.sh`; nothing installed in
`h2snn-b`). Its `__init__.py:24` imports `utils.py`, which imports `sortednp` (numpy < 2 only); PSHGCN never uses that
module (`SparseAdjList`), so `pshgcn_mag_modules` puts a stub `sortednp` in `sys.modules` for the time the vendor modules
are in use and removes it, and the vendor modules, afterwards.
"""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace
from typing import Any, Iterator

import torch
from torch import nn


def build_model(widths: dict[str, int], emb_dim: int, hidden_x: int, hidden_l: int, nclass: int,
                label_keys: list[str], layers_x: int, layers_l: int, coe_num: int, expander: list[str],
                dropout: float, input_drop: float, bias: bool = False) -> nn.Module:
    """Their `PSHGCN` with `widths` (letter -> input width; the target is `P`)."""
    from src.baselines.runtime import pushd_sys_path

    with pushd_sys_path("pshgcn_mag"):
        from model import MLP, PSHGCN  # type: ignore[import-not-found]

        model = PSHGCN(emb_dim, hidden_x, hidden_l, nclass, list(widths), label_keys, layers_x, layers_l, coe_num,
                       expander, dropout, input_drop, bias)
        for node, width in widths.items():
            model.embedings_x[node] = MLP(width, emb_dim, emb_dim, model.layers_emb, dropout, bias=bias)
        model.lin_residue = nn.Linear(widths["P"], emb_dim, bias=False)
        nn.init.xavier_uniform_(model.lin_residue.weight, gain=nn.init.calculate_gain("relu"))
    return model


SORTEDNP_MESSAGE = "sortednp is not available: SparseAdjList is not used by PSHGCN"
#: Their ogbn-mag letters (`processing.py:113-125`), on H2GB's MAG type names.
MAG_LETTERS = {"paper": "P", "author": "A", "institution": "I", "field_of_study": "F"}
#: Their label keys (`main.py:126`); they exist for K = 2, their default (`main.py:198`).
LABEL_KEYS = ["PPPP", "PAAP", "PP", "PFFP"]
MAG_K = 2


class _UnavailableModule(types.ModuleType):
    """Any use raises. Dunder lookups (`__file__`, `__path__`, ...) are answered as by a module without them
    (`AttributeError`), so Python's and `pushd_sys_path`'s module introspection (`getattr(module, "__file__", None)`,
    `vendor_paths.py:91`) still works; every other name -- any real use of sortednp -- raises `RuntimeError`."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError(SORTEDNP_MESSAGE)


@contextlib.contextmanager
def pshgcn_mag_modules() -> Iterator[SimpleNamespace]:
    """Their ogbn-mag `processing`, `utils`, `model` and `sparse_tools` (vendor clones), importable only inside this
    block, with a stub `sortednp` that raises on any use. On exit the stub and every module of the two clones leave
    `sys.modules` (`pushd_sys_path` evicts what it loaded)."""
    from src.baselines.runtime import pushd_sys_path

    if "sortednp" in sys.modules:
        raise RuntimeError("a sortednp module is already loaded; PSHGCN's label part expects none")
    sys.modules["sortednp"] = _UnavailableModule("sortednp")
    try:
        with pushd_sys_path("pshgcn_mag", "sparse_tools"):
            import sparse_tools  # type: ignore[import-not-found]
            import processing  # type: ignore[import-not-found]
            import utils as pshgcn_utils  # type: ignore[import-not-found]
            import model as pshgcn_model  # type: ignore[import-not-found]

            yield SimpleNamespace(sparse_tools=sparse_tools, processing=processing, utils=pshgcn_utils,
                                  model=pshgcn_model)
    finally:
        if isinstance(sys.modules.get("sortednp"), _UnavailableModule):
            del sys.modules["sortednp"]
        for name in [n for n in sys.modules if n == "sparse_tools" or n.startswith("sparse_tools.")]:
            del sys.modules[name]


def _adjacency(edge_index: torch.Tensor, rows: int, cols: int) -> Any:
    from torch_sparse import SparseTensor

    return SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(rows, cols))


def mag_relations(data: Any) -> list[tuple[tuple[str, str, str], Any]]:
    """Their four MAG relations in their order (`processing.py:113-125`: A-I, A-P, P-P (symmetrised), P-F) as
    `SparseTensor(row=src, col=dst)` from H2GB's MAG graph (sizes from the node counts)."""
    n = {t: int(data[t].num_nodes) for t in MAG_LETTERS}
    out = [(("A", "A-I", "I"), _adjacency(data["author", "affiliated_with", "institution"].edge_index,
                                         n["author"], n["institution"])),
           (("A", "A-P", "P"), _adjacency(data["author", "writes", "paper"].edge_index, n["author"], n["paper"])),
           (("P", "P-P", "P"), _adjacency(data["paper", "cites", "paper"].edge_index, n["paper"],
                                          n["paper"]).to_symmetric()),
           (("P", "P-F", "F"), _adjacency(data["paper", "has_topic", "field_of_study"].edge_index, n["paper"],
                                          n["field_of_study"]))]
    if not bool((out[2][1].get_diag() == 0).all()):  # their assert, processing.py:113
        raise ValueError("P-P has self-loops")
    return out


def mag_graph(relations: list, num_nodes: dict[str, int]) -> Any:
    """Their DGL heterograph (`processing.py:114-125`): each relation, and its reverse unless same-type."""
    import dgl

    edges = {}
    for (stype, rtype, dtype), adj in relations:
        src, dst, _ = adj.coo()
        edges[(stype, rtype, dtype)] = (src, dst)
        if stype != dtype:
            edges[(dtype, rtype[::-1], stype)] = (dst, src)
    return dgl.heterograph(edges, {MAG_LETTERS[t]: c for t, c in num_nodes.items()})


def self_effect_diagonals(modules: SimpleNamespace, relations: list) -> dict[str, torch.Tensor]:
    """Their diagonals (`processing.py:134-150`): PFFP = ABA(P-F), PPPP = AAA(P-P), PAAP = ABA((A-P)^T)."""
    st = modules.sparse_tools
    adjs = [adj for _, adj in relations]
    return {"PFFP": st.spspmm_diag_sym_ABA(adjs[3]), "PPPP": st.spspmm_diag_sym_AAA(adjs[2]),
            "PAAP": st.spspmm_diag_sym_ABA(adjs[1].t())}


def label_inputs(modules: SimpleNamespace, relations: list, num_nodes: dict[str, int], labels: torch.Tensor,
                 train_mask: torch.Tensor, num_classes: int, K: int = MAG_K) -> dict[str, torch.Tensor]:
    """Their label part at stage 0 (`main.py:96-115`): one-hot labels of the **training rows only**, propagated with
    their `propagate`, the self-effect `diag * label_onehot` removed for PPPP, PAAP, PFFP."""
    import torch.nn.functional as F

    if K != MAG_K:
        raise ValueError(f"their label keys {LABEL_KEYS} exist for K = {MAG_K} only (main.py:126)")
    g = mag_graph(relations, num_nodes)
    label_onehot = torch.zeros((num_nodes["paper"], num_classes))
    train = train_mask.nonzero().view(-1)
    label_onehot[train] = F.one_hot(labels[train].long(), num_classes).float()
    g.nodes["P"].data["P"] = label_onehot
    g = modules.utils.propagate(g, K)
    label_feats = {k: g.nodes["P"].data.pop(k) for k in list(g.nodes["P"].data.keys()) if k != "P"}
    modules.utils.propagate_clean(g)
    for key, diag in self_effect_diagonals(modules, relations).items():
        if key in label_feats:
            label_feats[key] = label_feats[key] - diag.unsqueeze(-1) * label_onehot
            if not bool((label_feats[key] > -1e-6).all()):  # their assert, main.py:114
                raise ValueError(f"{key}: self-effect removal left a negative entry")
    return label_feats


def feature_inputs(modules: SimpleNamespace, relations: list, num_nodes: dict[str, int], paper_x: torch.Tensor,
                   seed: int, K: int = MAG_K) -> tuple[dict[str, torch.Tensor], list[str], int]:
    """Their feature part (`processing.py:193-205, 153-160`): paper features, random-uniform(-0.5, 0.5) 256-wide inputs
    for author / field / institution (their default without extra embeddings; inputs rule B), `propagate(g, K)`, and
    `expander(K // 2)`. Returns (features of the papers per key, expander, coefficient count)."""
    g = mag_graph(relations, num_nodes)
    generator = torch.Generator().manual_seed(int(seed))
    g.nodes["P"].data["P"] = paper_x.float()
    for node_type in ("author", "field_of_study", "institution"):
        letter = MAG_LETTERS[node_type]
        g.nodes[letter].data[letter] = torch.rand(num_nodes[node_type], 256, generator=generator) - 0.5
    g = modules.utils.propagate(g, K)
    feats = {k: g.nodes["P"].data.pop(k) for k in list(g.nodes["P"].data.keys())}
    modules.utils.propagate_clean(g)
    expander, coe_num = modules.processing.expander(int(K / 2))
    return feats, expander, coe_num


def build_mag_model(modules: SimpleNamespace, num_classes: int, coe_num: int, expander: list[str],
                    params: Any) -> nn.Module:
    """Their `PSHGCN` (`ogbn-mag/model.py`) with their CLI defaults (`main.py:186-200`) unless searched; node types and
    label keys as `main.py:125-126`. Their widths (P 128, others 256) equal the inputs built here."""
    return modules.model.PSHGCN(int(params.get("emb_dim", 512)), int(params.get("hidden_x", 1024)),
                                int(params.get("hidden_l", 512)), num_classes, ["P", "A", "I", "F"], LABEL_KEYS,
                                int(params.get("layers_x", 3)), int(params.get("layers_l", 3)), coe_num, expander,
                                float(params.get("dropout", 0.5)), float(params.get("input_drop", 0.1)), False)


def fit(bundle: Any, params: Any, *, seed: int, device: torch.device) -> dict[str, float]:
    """Card 1d H1: PSHGCN on ogbn-mag / mag-year with their ogbn-mag code in full, batches of target rows in the shared
    H2GB NC loop (stage 0; the enhance stage is off in their default `--stage 1`, `main.py:200`)."""
    from src.baselines.minibatch import h2gb_graph, minibatch_plan, train_index_minibatch
    from src.utils.misc import seed_everything

    seed_everything(seed, device)
    data = h2gb_graph(bundle)
    if int(data["paper"].x.size(1)) != 128:
        raise ValueError("their ogbn-mag model reads 128-wide paper features (model.py:59)")
    num_nodes = {t: int(data[t].num_nodes) for t in MAG_LETTERS}
    relations = mag_relations(data)
    labels = data["paper"].y.long()
    num_classes = int(labels.max()) + 1
    with pshgcn_mag_modules() as modules:
        feats, expander, coe_num = feature_inputs(modules, relations, num_nodes, data["paper"].x, seed)
        label_feats = label_inputs(modules, relations, num_nodes, labels, data["paper"].train_mask, num_classes)
        torch.manual_seed(seed)
        model = build_mag_model(modules, num_classes, coe_num, expander, params).to(device)
    plan = minibatch_plan(bundle.dataset, int(params.get("batch_size", 256)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 1e-3)),
                                  weight_decay=float(params.get("weight_decay", 5e-5)))

    def forward_rows(index: torch.Tensor) -> torch.Tensor:
        rows = index.cpu()
        return model({k: v[rows].to(device) for k, v in feats.items()},
                     {k: label_feats[k][rows].to(device) for k in LABEL_KEYS})

    record = train_index_minibatch(
        model=model, forward_rows=forward_rows, labels=labels.to(device),
        train_idx=data["paper"].train_mask.nonzero().view(-1), val_idx=data["paper"].val_mask.nonzero().view(-1),
        test_idx=data["paper"].test_mask.nonzero().view(-1), optimizer=optimizer,
        criterion=torch.nn.functional.cross_entropy, plan=plan, base_seed=seed, report=getattr(params, "report", None))
    record["params_source"] = params.source
    record["batch_size"] = float(plan.batch_size)
    return record


__all__ = ["LABEL_KEYS", "MAG_K", "MAG_LETTERS", "SORTEDNP_MESSAGE", "build_mag_model", "build_model", "fit", "feature_inputs", "label_inputs",
           "mag_graph", "mag_relations", "pshgcn_mag_modules", "self_effect_diagonals"]
