"""H0 data facts (read-only): parallel facts after the tied fact layer, (relation, direction)
actions per node type, anchors, and the schemas-run incidence budget (eval draw, seed 42).
Run from H2S-NN/: python ../docs/WISP_evidence/h0_data_facts.py"""
import json, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path.cwd()))
from src.configs.config import TrainConfig
from src.data.dataset import build_dataset
from src.data.path_sampler import build_half_edges

out = {}
for ds in ("acm", "dblp", "imdb"):
    raw = yaml.safe_load(Path(f"configs/experiments/{ds}_h2snn.yaml").read_text(encoding="utf-8"))
    raw["device"] = "cpu"
    data = build_dataset(TrainConfig.from_dict(raw))
    src, dst = data.edge_index
    key = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst), data.edge_type], 1)
    parallel = int(key.size(0) - torch.unique(key, dim=0).size(0))
    layer = data.fact_layer
    half = build_half_edges(data.edge_index, data.edge_type, data.num_nodes, layer.self_inverse, True)
    action = half.relation * 3 + (half.direction + 1)
    actions_by_type = {}
    for t, name in enumerate(data.node_type_names):
        mask = data.node_type[half.node] == t
        actions_by_type[name] = int(torch.unique(action[mask]).numel())
    incidences = sum(g.num_faces * (g.length + 1) for g in data.face_groups)
    tgt = data.target_node_type
    out[ds] = {"num_nodes": int(data.num_nodes), "facts": int(src.numel()),
               "parallel_facts_same_pair_same_relation": parallel,
               "self_loop_facts": int((src == dst).sum()),
               "relations": list(data.relation_names), "self_inverse": list(layer.self_inverse),
               "actions_per_node_type": actions_by_type,
               "anchors_target_type": f"{tgt}: {data.node_slices[tgt][1] - data.node_slices[tgt][0]}",
               "schemas_eval_draw": {g.template_name: [g.length, g.num_faces] for g in data.face_groups},
               "schemas_incidences": incidences}
    print(ds, json.dumps(out[ds]), flush=True)
(Path.cwd().parent / "docs" / "WISP_evidence" / "h0_data_facts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
