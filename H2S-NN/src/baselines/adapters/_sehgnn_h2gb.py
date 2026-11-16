"""SeHGNN on the H2GB node-classification datasets: its own large-graph scheme (`vendor/SeHGNN/ogbn`, card
baselines-h2gb C4, class B).

What their ogbn code does (VERIFIED-CODE, SeHGNN `e92bd37`): features are propagated once before any model exists by
`hg_propagate` (`ogbn/utils.py:49-80`: per hop, a mean over each relation into the key `f'{dst}{key}'`, every node type
named by one letter, `ogbn/utils.py:378-383`), then `SeHGNN_mag` (`ogbn/model.py:126-273`) is trained on batches of
target rows of those stored features (`ogbn/main.py:170-174`, `ogbn/utils.py:138-139`); evaluation slices contiguous
rows (`main.py:296-306`). Featureless node types get fixed random-uniform(-0.5, 0.5) vectors of `embed_size`
(`ogbn/utils.py:352-354`, their default without `--extra-embedding`).

Adapter choices (stated in the C4 report; the H2GB graph is H2GB's `dataset[0]`):
* node types are given distinct letters (first letter of the name, else the next unused capital) because their keys
  are built from one letter per type;
* **parallel relations** between the same (source type, destination type) are merged into one relation (union of the
  edges): their naming would otherwise write every parallel relation into the same key and keep the last one (OAG:
  paper-field x6, author-paper x3; PDNS: domain-domain x2). MAG and IEEE-CIS have none;
* stage 0 of their multi-stage training only: no label features (`main.py:290-291` drops them at stage 0) and no
  enhance stage; their self-effect removal for label features reads MAG-specific diagonal files (`main.py:270-276`).
"""

from __future__ import annotations

import contextlib
import string
import sys
import types
from typing import Any, Iterator

import torch


@contextlib.contextmanager
def _without_sparse_tools() -> Iterator[None]:
    """Their `ogbn/utils.py:28` imports `sparse_tools` (a separate C++ package, `SeHGNN/Readme.md:24`, not installed),
    used only for the label self-effect diagonals (`ogbn/utils.py:306-317`), never by `hg_propagate`. While the module
    is imported, a stub that raises on any use stands in for it (no vendor edit)."""
    if "sparse_tools" in sys.modules:
        yield
        return
    stub = types.ModuleType("sparse_tools")

    def _refuse(name: str) -> Any:
        raise NotImplementedError(f"sparse_tools.{name} is not available (card baselines-h2gb C4: stage 0 only).")

    stub.__getattr__ = _refuse  # type: ignore[attr-defined]
    sys.modules["sparse_tools"] = stub
    try:
        yield
    finally:
        sys.modules.pop("sparse_tools", None)


def type_letters(node_types: list[str] | tuple[str, ...]) -> dict[str, str]:
    """A distinct capital letter per node type, in order: the type's first letter if free, else the next free one."""
    letters: dict[str, str] = {}
    used: set[str] = set()
    for node_type in node_types:
        first = node_type[:1].upper()
        letter = first if first in string.ascii_uppercase and first not in used else next(
            c for c in string.ascii_uppercase if c not in used)
        letters[node_type] = letter
        used.add(letter)
    if len(letters) != len(set(letters.values())):
        raise ValueError("more than 26 node types")
    return letters


def merged_relations(data: Any, letters: dict[str, str]) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]:
    """One relation per (source letter, destination letter): the union of `data`'s relations between those types,
    named as their MAG relations (`'A-P'`, `ogbn/utils.py:378-383`)."""
    grouped: dict[tuple[str, str], list[torch.Tensor]] = {}
    for src, _, dst in data.edge_types:
        grouped.setdefault((src, dst), []).append(data[src, _, dst].edge_index)
    out = {}
    for (src, dst), parts in grouped.items():
        edges = torch.unique(torch.cat(parts, dim=1), dim=1)
        ls, ld = letters[src], letters[dst]
        out[(ls, f"{ls}-{ld}", ld)] = (edges[0], edges[1])
    return out


#: Card 1c F3 (decision G-4): on ogbn-mag / mag-year their README's "without extra embeddings" command feeds author,
#: field and institution with random-uniform(-0.5, 0.5) vectors of `embed_size` (`ogbn/utils.py:352-354`), not with
#: the graph's (H2GB's averaged) features; stated.
MAG_RANDOM_TYPES: dict[str, frozenset[str]] = {
    "ogbn-mag": frozenset({"author", "field_of_study", "institution"}),
    "mag-year": frozenset({"author", "field_of_study", "institution"}),
}


def node_inputs(data: Any, embed_size: int, seed: int, random_types: frozenset[str] = frozenset()
                ) -> dict[str, torch.Tensor]:
    """Each type's `x`, or their fixed random-uniform(-0.5, 0.5) vectors for a featureless type or a type in
    `random_types` (their MAG branch)."""
    generator = torch.Generator().manual_seed(int(seed))
    inputs = {}
    for node_type in data.node_types:
        x = data[node_type].get("x")
        if x is None or node_type in random_types:
            x = torch.rand(int(data[node_type].num_nodes), embed_size, generator=generator) - 0.5
        inputs[node_type] = x
    return inputs


def propagate(data: Any, target_type: str, num_hops: int, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Their `hg_propagate` on H2GB's graph: the stored features of every target node, one tensor per meta-path key."""
    import dgl

    from src.baselines.runtime import pushd_sys_path

    letters = type_letters(list(data.node_types))
    graph = dgl.heterograph(merged_relations(data, letters),
                            {letters[t]: int(data[t].num_nodes) for t in data.node_types})
    for node_type, x in inputs.items():
        graph.nodes[letters[node_type]].data[letters[node_type]] = x
    target = letters[target_type]
    with pushd_sys_path("sehgnn_ogbn"), _without_sparse_tools():
        from utils import hg_propagate  # type: ignore[import-not-found]

        graph = hg_propagate(graph, target, num_hops, num_hops + 1, [], echo=False)
    return {key: graph.nodes[target].data.pop(key) for key in list(graph.nodes[target].data.keys())}


def build_model(feats: dict[str, torch.Tensor], target_letter: str, num_classes: int, params: Any) -> torch.nn.Module:
    """`SeHGNN_mag` with their CLI defaults (`ogbn/main.py:404-460`) unless searched; no label features (stage 0)."""
    from src.baselines.runtime import pushd_sys_path

    data_size = {key: int(value.size(-1)) for key, value in feats.items()}
    with pushd_sys_path("sehgnn_ogbn"):
        from model import SeHGNN_mag  # type: ignore[import-not-found]

        return SeHGNN_mag("h2gb", data_size, int(params.get("embed_size", 256)), int(params.get("hidden", 512)),
                          num_classes, len(feats), 0, target_letter,
                          dropout=float(params.get("dropout", 0.5)), input_drop=float(params.get("input_drop", 0.1)),
                          att_drop=float(params.get("att_drop", 0.0)), label_drop=0.0,
                          n_layers_1=int(params.get("n_layers_1", 2)), n_layers_2=int(params.get("n_layers_2", 2)),
                          n_layers_3=int(params.get("n_layers_3", 4)), act=str(params.get("act", "relu")),
                          residual=bool(params.get("residual", False)), bns=bool(params.get("bns", False)))


__all__ = ["build_model", "merged_relations", "node_inputs", "propagate", "type_letters"]
