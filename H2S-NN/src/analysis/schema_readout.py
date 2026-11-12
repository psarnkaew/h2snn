"""Schema read-out and coverage of a run's evaluation draw (spec v1.19 §6 "Schema read-out and
coverage", §3.2 read-outs; card Part O, O2). Evaluation only: imports no training code; the model
comes from `src.analysis.model_loading`.

Per sub-path schema sigma (canonical signature of positions i..j of a cell, `wisp_readouts`):

    I_sigma      share of the pair weight w_f m_f / (2 |f|) over the pairs of sigma (Prop. K)
    c_sigma      the same with every w_f = 1 (coverage: how much of the draw's mass is sigma)
    rho_sigma    I_sigma / c_sigma (1 for every sigma when g is constant)
    mean_g       mean g(xi_f) = log w_f over the distinct cells that contain sigma

`Delta L_sigma` is not computed (its implementation uses the v2 operator). Literature meta-paths
(`LITERATURE_METAPATHS`) are mapped to signatures by relation and type names; a name that does not
resolve raises.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from src.analysis.wisp_readouts import kendall_tau, pair_index
from src.configs.config import LITERATURE_METAPATHS, MetaPathConfig
from src.data.fact_layer import FactLayer
from src.data.schemas import resolve_schema
from src.data.structures import FaceGroup

Signature = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SchemaStats:
    signature: Signature
    span: int            # number of hops of the sub-path
    coverage: float      # c_sigma
    share: float         # I_sigma
    ratio: float | None  # rho_sigma (None when c_sigma = 0)
    mean_g: float        # mean log w_f over the cells containing sigma
    num_cells: int       # distinct cells containing sigma


def schema_statistics(groups: tuple[FaceGroup, ...], weights: tuple[torch.Tensor, ...],
                      node_type: torch.Tensor, self_inverse: tuple[int, ...] = ()
                      ) -> dict[Signature, SchemaStats]:
    """c_sigma, I_sigma, rho_sigma, mean g per sub-path schema of the cells `groups` (non-empty
    groups, aligned with their `weights` = w_f). Float64 throughout."""
    if len(groups) != len(weights):
        raise ValueError(f"{len(weights)} weight tensors for {len(groups)} groups.")
    learned: dict[Signature, float] = defaultdict(float)
    uniform: dict[Signature, float] = defaultdict(float)
    cells: dict[Signature, set[tuple[int, int]]] = defaultdict(set)
    log_w_of: dict[tuple[int, int], float] = {}
    for g, (group, weight) in enumerate(zip(groups, weights)):
        if group.num_faces == 0:
            raise ValueError("schema_statistics takes the non-empty groups only.")
        if group.path_weight is None:
            raise ValueError("a cell group without m_f (path_weight) has no coverage.")
        w = weight.detach().double().cpu()
        if w.numel() != group.num_faces:
            raise ValueError(f"group {g}: {w.numel()} weights for {group.num_faces} cells.")
        m = group.path_weight.detach().double().cpu()
        size = float(group.face_node_index.size(1))  # |f| = number of nodes of the cell
        pairs = pair_index(group, node_type, self_inverse)
        learned_pair = (w * m)[pairs.cell] / (2.0 * size)
        uniform_pair = m[pairs.cell] / (2.0 * size)
        log_w = w.log().tolist()
        for c, value_w, value_1, key in zip(pairs.cell.tolist(), learned_pair.tolist(),
                                            uniform_pair.tolist(), pairs.schema):
            learned[key] += value_w
            uniform[key] += value_1
            cells[key].add((g, c))
            log_w_of[(g, c)] = log_w[c]
    total_w, total_1 = sum(learned.values()), sum(uniform.values())
    out: dict[Signature, SchemaStats] = {}
    for key in learned:
        share, coverage = learned[key] / total_w, uniform[key] / total_1
        members = cells[key]
        out[key] = SchemaStats(
            signature=key, span=(len(key) - 1) // 3, coverage=coverage, share=share,
            ratio=share / coverage if coverage > 0 else None,
            mean_g=sum(log_w_of[cell] for cell in members) / len(members),
            num_cells=len(members))
    return out


# ----------------------------------------------------------------------------- names
def signature_of(types: tuple[int, ...], hops: tuple[tuple[int, int], ...]) -> Signature:
    """Canonical signature (the smaller orientation, as `wisp_readouts._canonical`)."""
    forward: list[int] = [types[0]]
    for (relation, direction), t in zip(hops, types[1:]):
        forward += [relation, direction + 1, t]
    rtypes = types[::-1]
    backward: list[int] = [rtypes[0]]
    for (relation, direction), t in zip(reversed(hops), rtypes[1:]):
        backward += [relation, -direction + 1, t]
    return min(tuple(forward), tuple(backward))


def literature_signature(template: MetaPathConfig, layer: FactLayer,
                         relation_tuples: tuple[tuple[str, str, str], ...],
                         node_type_names: tuple[str, ...]) -> Signature:
    """Signature of one listed meta-path, by its relation and type names; raises if a name does
    not resolve (`FactLayer.resolve`: unknown relation; `resolve_schema`: a hop that does not join
    the named types) or a type is not in the graph."""
    schema = resolve_schema(template, layer, relation_tuples)
    missing = [t for t in schema.node_types if t not in node_type_names]
    if missing:
        raise KeyError(f"meta-path {template.name!r}: node type(s) {missing} not in "
                       f"{node_type_names}.")
    types = tuple(node_type_names.index(t) for t in schema.node_types)
    return signature_of(types, schema.hops)


def literature_signatures(dataset_name: str, layer: FactLayer,
                          relation_tuples: tuple[tuple[str, str, str], ...],
                          node_type_names: tuple[str, ...],
                          templates: tuple[MetaPathConfig, ...] | None = None
                          ) -> dict[str, Signature]:
    """`{listed name: signature}` for the dataset's literature list (or `templates`); empty
    only for a dataset without a list (`LITERATURE_METAPATHS`)."""
    listed = LITERATURE_METAPATHS.get(dataset_name.lower(), ()) if templates is None else templates
    out = {t.name: literature_signature(t, layer, relation_tuples, node_type_names)
           for t in listed}
    if len(set(out.values())) != len(out):
        raise ValueError(f"two listed meta-paths of {dataset_name!r} map to one signature: {out}.")
    return out


def schema_name(signature: Signature, node_type_names: tuple[str, ...],
                relation_names: tuple[str, ...]) -> tuple[str, str]:
    """(`paper-author-paper`, `paper -[paper__to__author +]-> author -[... -]-> paper`)."""
    types = [node_type_names[t] for t in signature[0::3]]
    parts = [types[0]]
    for j in range((len(signature) - 1) // 3):
        relation, direction = signature[3 * j + 1], signature[3 * j + 2] - 1
        mark = "+" if direction > 0 else "-" if direction < 0 else "~"
        parts.append(f"-[{relation_names[relation]} {mark}]-> {types[j + 1]}")
    return "-".join(types), " ".join(parts)


# ----------------------------------------------------------------------------- runs
def run_record(stats: dict[Signature, SchemaStats], node_type_names: tuple[str, ...],
               relation_names: tuple[str, ...], literature: dict[str, Signature],
               meta: dict) -> dict:
    """The JSON of one run: every sub-path schema (sorted by I_sigma), the listed meta-paths."""
    def entry(stat: SchemaStats) -> dict:
        short, full = schema_name(stat.signature, node_type_names, relation_names)
        return {"name": short, "relations": full, "signature": list(stat.signature),
                "span": stat.span, "c": stat.coverage, "I": stat.share, "rho": stat.ratio,
                "mean_g": stat.mean_g, "num_cells": stat.num_cells}

    ordered = sorted(stats.values(), key=lambda s: (-s.share, s.signature))
    listed = []
    for name, signature in literature.items():
        if signature in stats:
            listed.append({"listed": name, "in_draw": True, **entry(stats[signature])})
        else:  # in the list, absent from this draw: stated, never dropped
            short, full = schema_name(signature, node_type_names, relation_names)
            listed.append({"listed": name, "in_draw": False, "name": short, "relations": full,
                           "signature": list(signature), "span": (len(signature) - 1) // 3,
                           "c": 0.0, "I": 0.0, "rho": None, "mean_g": None, "num_cells": 0})
    return {
        **meta,
        "num_distinct_schemas": len(stats),
        "num_distinct_length2_schemas": sum(1 for s in stats.values() if s.span == 2),
        "schemas": [entry(s) for s in ordered],
        "literature": listed if literature else None,
    }


def kendall_matrix(records: list[dict], span: int | None = None, permutations: int = 1000
                   ) -> list[dict]:
    """Kendall tau-b (and permutation p) of the I_sigma rankings between every pair of runs, over
    the union of their schemas (missing = 0); `span` restricts to sub-paths of that length."""
    def scores(record: dict) -> dict[Signature, float]:
        return {tuple(s["signature"]): s["I"] for s in record["schemas"]
                if span is None or s["span"] == span}

    out = []
    for a in range(len(records)):
        for b in range(a + 1, len(records)):
            tau, p = kendall_tau(scores(records[a]), scores(records[b]), permutations=permutations)
            out.append({"a": records[a]["run"], "b": records[b]["run"],
                        "tau": None if math.isnan(tau) else tau, "p": p})
    return out


__all__ = ["SchemaStats", "kendall_matrix", "literature_signature", "literature_signatures",
           "run_record", "schema_name", "schema_statistics", "signature_of"]
