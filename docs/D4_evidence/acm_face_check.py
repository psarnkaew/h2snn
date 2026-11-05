"""D4 pre-step inventory item 2: face groups on real ACM.

Checks, per face group: validate_face_orientation (path edges stored along the path, simple
paths), relation sequence constant, closure edge = (v_L, v_0), lengths and counts. Also
reports graph sizes and nodes with no observed edge (D_v = 0 candidates).

    python <this>        (run from the H2S-NN repo root, h2snn env)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig  # noqa: E402
from src.data.face_geometry import (  # noqa: E402
    is_simple,
    validate_face_orientation,
    validate_relation_sequence_constant,
)

CACHE = Path(__file__).resolve().parent / "acm_prepared.pt"


def main() -> None:
    with open("configs/experiments/acm_h2snn.yaml", encoding="utf-8") as handle:
        cfg = TrainConfig.from_dict(yaml.safe_load(handle))
    if CACHE.exists():
        data = torch.load(CACHE, weights_only=False)
    else:
        from src.data.dataset import build_dataset

        data = build_dataset(cfg)
        torch.save(data, CACHE)
    report: dict[str, object] = {
        "N": int(data.num_nodes),
        "M": int(data.num_edges),
        "num_relations": len(data.relation_names),
        "discovery_is_none": data.discovery is None,
        "groups": [],
    }
    degree = torch.zeros(data.num_nodes, dtype=torch.long)
    degree.index_add_(0, data.edge_index[0], torch.ones(data.num_edges, dtype=torch.long))
    degree.index_add_(0, data.edge_index[1], torch.ones(data.num_edges, dtype=torch.long))
    report["nodes_without_observed_edges"] = int((degree == 0).sum())
    for group in data.face_groups:
        entry: dict[str, object] = {
            "name": group.template_name,
            "faces": int(group.num_faces),
            "L": int(group.face_edge_index.size(1)),
        }
        try:
            validate_face_orientation(
                group.face_node_index, group.face_edge_index, data.edge_index,
                template_name=group.template_name,
            )
            entry["orientation_ok"] = True
        except ValueError as exc:
            entry["orientation_ok"] = f"FAIL: {exc}"
        entry["simple"] = bool(is_simple(group.face_node_index))
        try:
            validate_relation_sequence_constant(group.relation_index, group.template_name)
            entry["relation_sequence_constant"] = True
        except ValueError as exc:
            entry["relation_sequence_constant"] = f"FAIL: {exc}"
        entry["relation_sequence"] = group.relation_index[0].tolist()
        expected_closure = torch.stack(
            [group.face_node_index[:, -1], group.face_node_index[:, 0]], dim=1
        )
        entry["closure_is_vL_v0"] = bool(torch.equal(group.closure_edge_index, expected_closure))
        entry["edge_types_match_relation_index"] = bool(
            torch.equal(data.edge_type[group.face_edge_index], group.relation_index)
        )
        report["groups"].append(entry)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
