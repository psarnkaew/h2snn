"""G2 data facts for Lemma N (documented, not tested): self-loop facts per dataset, and path cells
with a node repeated inside the cell (the evaluation draw, seed 42). Run from H2S-NN/.
Writes docs/D12_evidence/g2_lemma_n_counts.json."""
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402

out = {}
for dataset in ("acm", "dblp", "imdb"):
    raw = yaml.safe_load((ROOT / "configs" / "experiments" / f"{dataset}_h2snn.yaml").read_text(encoding="utf-8"))
    raw["device"] = "cpu"
    data = build_dataset(TrainConfig.from_dict(raw))
    src, tgt = data.edge_index
    loops = src == tgt
    excluded = set(src[loops].tolist())
    cells_repeated, nodes_repeated = {}, set()
    for group in data.face_groups:
        nodes = group.face_node_index
        sorted_nodes = nodes.sort(dim=1).values
        repeated = (sorted_nodes[:, 1:] == sorted_nodes[:, :-1]).any(dim=1)
        cells_repeated[group.template_name] = [int(repeated.sum()), group.num_faces]
        for row in nodes[repeated].tolist():
            nodes_repeated |= {v for v in row if row.count(v) > 1}
    out[dataset] = {
        "facts": int(src.numel()), "self_loop_facts": int(loops.sum()),
        "nodes_with_self_loop_fact": len(excluded),
        "self_loop_facts_by_relation": {data.relation_names[r]: int((data.edge_type[loops] == r).sum())
                                        for r in range(len(data.relation_names))},
        "cells_with_repeated_node_by_schema [repeated, total]": cells_repeated,
        "nodes_repeated_inside_a_cell": len(nodes_repeated),
        "nodes_outside_lemma_N": len(excluded | nodes_repeated), "num_nodes": int(data.num_nodes),
    }
    print(dataset, json.dumps(out[dataset]), flush=True)
(ROOT.parent / "docs" / "D12_evidence" / "g2_lemma_n_counts.json").write_text(json.dumps(out, indent=1),
                                                                               encoding="utf-8")
