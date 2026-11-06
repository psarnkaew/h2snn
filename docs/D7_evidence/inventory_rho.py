"""Part D pre-step, inventory item 2: the exact inverse relation rho per dataset.

Independent of `src/` fact-layer code (none exists yet): rho is computed here from the
definition only,

    rho(r) = r'  iff  E(r') == {(v, u) : (u, v) in E(r)}   (sets, exact; r' = r allowed)

and every count is derived from the stored edge lists the pipeline builds (`flatten_edges`
for HGB / AmazonBook, the TU loader's `edge_type` for TU graphs).

Run from `H2S-NN/`:  python ../docs/D7_evidence/inventory_rho.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2] / "H2S-NN"
sys.path.insert(0, str(REPO))


def _keys(src: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    return src.long() * n + dst.long()


def analyse(name: str, edge_index: torch.Tensor, edge_type: torch.Tensor,
            relation_names: list[str], relation_types: list[tuple[str, str]],
            num_nodes: int) -> dict:
    R = len(relation_names)
    n = int(num_nodes)
    per_rel = []
    key_sets = []
    rev_sets = []
    for r in range(R):
        mask = edge_type == r
        s, t = edge_index[0, mask], edge_index[1, mask]
        k = _keys(s, t, n)
        uniq = torch.unique(k)
        key_sets.append(uniq)
        rev_sets.append(torch.unique(_keys(t, s, n)))
        loops = int((s == t).sum())
        # reciprocal facts inside r: unordered pairs {u, v}, u != v, with both (u,v), (v,u) in E(r)
        non_loop = uniq[(uniq // n) != (uniq % n)]
        rev_non_loop = (non_loop % n) * n + (non_loop // n)
        reciprocal_pairs = int(torch.isin(non_loop, rev_non_loop).sum()) // 2
        per_rel.append({
            "relation": relation_names[r],
            "types": relation_types[r],
            "stored_edges": int(mask.sum()),
            "distinct_edges": int(uniq.numel()),
            "duplicate_edges": int(mask.sum()) - int(uniq.numel()),
            "self_loops": loops,
            "reciprocal_pairs_within": reciprocal_pairs,
        })

    rho: dict[int, list[int]] = {}
    near: list[dict] = []
    for r in range(R):
        partners = []
        for r2 in range(R):
            # type-compatible: src(r') = dst(r), dst(r') = src(r)
            if relation_types[r2] != (relation_types[r][1], relation_types[r][0]):
                continue
            a, b = rev_sets[r], key_sets[r2]
            if a.numel() == b.numel() and torch.equal(a, b):
                partners.append(r2)
            else:
                inter = int(torch.isin(a, b).sum())
                if inter > 0 and r2 >= r:
                    near.append({
                        "r": relation_names[r], "r_prime": relation_names[r2],
                        "reversed_E_r": int(a.numel()), "E_r_prime": int(b.numel()),
                        "intersection": inter,
                        "missing_from_r_prime": int(a.numel()) - inter,
                        "extra_in_r_prime": int(b.numel()) - inter,
                    })
        rho[r] = partners

    # facts after tying (distinct stored edges up to inversion), one rule for all relations
    facts_after = 0
    seen = set()
    for r in range(R):
        if r in seen:
            continue
        partners = rho[r]
        if len(partners) > 1:
            raise RuntimeError(f"{name}: relation {relation_names[r]} has {len(partners)} exact partners")
        if not partners:  # rho(r) undefined: every edge is its own fact
            facts_after += per_rel[r]["distinct_edges"]
        elif partners[0] == r:  # self-inverse: one fact per unordered pair (loops count once)
            loops = per_rel[r]["self_loops"]
            distinct = per_rel[r]["distinct_edges"]
            facts_after += (distinct - loops) // 2 + loops
        else:  # mirror pair: one fact per edge of the pair
            facts_after += per_rel[r]["distinct_edges"]
            seen.add(partners[0])
        seen.add(r)

    for r in range(R):
        partners = rho[r]
        per_rel[r]["rho"] = relation_names[partners[0]] if partners else None
        per_rel[r]["unmatched_edges"] = 0 if partners else per_rel[r]["distinct_edges"]

    return {
        "dataset": name,
        "num_nodes": n,
        "num_relations": R,
        "facts_before": int(edge_index.size(1)),
        "facts_after": facts_after,
        "self_inverse": [relation_names[r] for r in range(R) if rho[r] == [r]],
        "mirror_pairs": sorted({tuple(sorted((relation_names[r], relation_names[rho[r][0]])))
                                for r in range(R) if rho[r] and rho[r][0] != r}),
        "untied": [relation_names[r] for r in range(R) if not rho[r]],
        "near_partners": near,
        "relations": per_rel,
    }


def hgb(name: str) -> dict:
    from src.configs.config import TrainConfig
    from src.data.dataset import _load_hetero_data
    from src.data.hetero_flatten import flatten_edges, stack_node_features

    config = TrainConfig.from_dict({"data": {"dataset_name": name, "root": "data"}})
    data = _load_hetero_data(config)
    x, node_type, names, slices, dims = stack_node_features(data)
    edge_index, edge_type, rel_names, rel_tuples = flatten_edges(data, slices)
    return analyse(name, edge_index, edge_type, list(rel_names),
                   [(a, c) for a, _, c in rel_tuples], x.size(0))


def amazonbook() -> dict:
    from torch_geometric.datasets import AmazonBook

    from src.data.hetero_flatten import flatten_edges, stack_node_features

    data = AmazonBook(root=str(Path("data") / "amazonbook"))[0]
    x, node_type, names, slices, dims = stack_node_features(data)
    edge_index, edge_type, rel_names, rel_tuples = flatten_edges(data, slices)
    return analyse("amazonbook", edge_index, edge_type, list(rel_names),
                   [(a, c) for a, _, c in rel_tuples], x.size(0))


def tu(name: str) -> dict:
    """Union of all graphs (node ids made disjoint), with the pipeline's edge types and,
    as a cross-check, PyG's own edge_attr labels (argmax of the one-hot), which PyG keeps
    aligned with its edge_index."""
    from torch_geometric.datasets import TUDataset

    from src.data.graph_classification_dataset import _TUD_NAMES, _load_tu_type_tensors

    tud = _TUD_NAMES[name]
    root = Path("data")
    dataset = TUDataset(root=str(root / "tud"), name=tud)
    data_list = [dataset[i].clone() for i in range(len(dataset))]
    _, edge_types, num_node_types, num_relations = _load_tu_type_tensors(root, tud, data_list)
    edges, pipeline_types, pyg_types = [], [], []
    offset = 0
    for data, et in zip(data_list, edge_types):
        edges.append(data.edge_index + offset)
        pipeline_types.append(et)
        if data.edge_attr is not None and data.edge_attr.numel():
            pyg_types.append(data.edge_attr.argmax(dim=-1))
        offset += int(data.num_nodes)
    edge_index = torch.cat(edges, dim=1)
    etype = torch.cat(pipeline_types)
    rel_names = [f"edge_label_{r}" for r in range(num_relations)]
    rel_types = [("node", "node")] * num_relations  # TU: every label joins any node labels
    result = analyse(name, edge_index, etype, rel_names, rel_types, offset)
    result["num_graphs"] = len(data_list)
    result["pipeline_num_relations"] = num_relations
    if pyg_types:
        pyg = torch.cat(pyg_types)
        result["pyg_edge_attr_available"] = True
        result["pipeline_vs_pyg_label_mismatch"] = int((pyg != etype).sum())
        result["pyg_label_analysis"] = analyse(
            name + "_pyg_edge_attr", edge_index, pyg, rel_names, rel_types, offset
        )
    else:
        result["pyg_edge_attr_available"] = False
    return result


def main(argv: list[str]) -> None:
    targets = argv or ["acm", "dblp", "imdb", "mutag", "proteins", "enzymes", "nci1",
                       "amazonbook"]
    out = []
    for name in targets:
        try:
            if name in {"acm", "dblp", "imdb"}:
                result = hgb(name)
            elif name == "amazonbook":
                result = amazonbook()
            else:
                result = tu(name)
        except Exception as exc:  # report, do not hide: each dataset is independent
            result = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}
        out.append(result)
        print(json.dumps(result, indent=1, default=str))
    Path(__file__).with_name("inventory_rho_output.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
