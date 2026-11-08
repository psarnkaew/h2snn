"""H5 S1' diagnosis (measurement only; no redesign). For the trained S1' runs (last checkpoint),
how much of each target paper's coupling comes from path cells versus fact edges, and how much of
the path part comes from the relevant cell:
  edge_deg(P)   = sum over P's fact edges of c^2            (Delta_0 part of the scalar degree)
  path_deg(P)   = sum_f w_f m_f (1 - 1/|f|) s_{P,f}^2         (L_P part)
  relevant_share = path_deg from the relevant cell / path_deg(P)
Run from H2S-NN/:  python ../docs/WISP_evidence/h5_s1p_diagnosis.py   Writes h5_s1p_diagnosis.json"""
import dataclasses
import json
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "docs" / "WISP_evidence"))
import h5_synthetic as h5  # noqa: E402
import src.data.dataset as dataset_module  # noqa: E402
from src.builders import build_model  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import H2SNN  # noqa: E402

graph, info = h5.graph_s1p()
dataset_module._load_hetero_data = lambda _config: graph.clone()
out = {}
for arm in ("oracle_weight", "full"):
    for seed in (42, 43, 44):
        cfg = dataclasses.replace(h5.config("s1p", arm, seed, info, 100), device="cpu")
        data = build_dataset(cfg)
        model = build_model(cfg, data)
        state = torch.load(cfg.exp_dir / "checkpoints" / "last.pt", map_location="cpu")["state_dict"]
        model.load_state_dict(state)
        model.eval()
        relevant = h5.relevance_of_cells(data, info)
        original = H2SNN._path_weights
        if arm == "oracle_weight":
            sizes = {int(r.numel()): r.float() for r in relevant}
            H2SNN._path_weights = lambda self, latent, group, like: sizes[group.num_faces].to(like.dtype)
        try:
            with torch.no_grad():
                h0 = model.encode_nodes(data.x, data.node_type)
                op = model.build_operator(h0, data.edge_index, data.edge_type, data.face_groups,
                                          data.num_nodes)
        finally:
            H2SNN._path_weights = original
        lap = op.laplacian
        targets = data.target_indices
        edge_deg = lap.edge_degree()[targets]
        path_deg = sum(t.diagonal(data.num_nodes) for t in lap.path_terms)[targets]
        term = lap.path_terms[0]
        rel = relevant[0]
        keep = term.weights * rel.to(term.weights.dtype)
        per = (1.0 - 1.0 / term.cell_size) * keep.unsqueeze(1) * term.incidence.squared_scales()
        rel_deg = per.new_zeros(data.num_nodes).index_add_(
            0, term.incidence.face_node_index.reshape(-1), per.reshape(-1))[targets]
        log_w = torch.cat(op.path_weights).clamp_min(1e-30).log()
        out[f"{arm}/s{seed}"] = {
            "path_share_of_degree_median": float((path_deg / (edge_deg + path_deg)).median()),
            "relevant_share_of_path_degree_median": float((rel_deg / path_deg.clamp_min(1e-30)).median()),
            "relevant_share_of_total_degree_median": float((rel_deg / (edge_deg + path_deg)).median()),
            "log_w_relevant_mean": float(log_w[rel].mean()) if arm == "full" else None,
            "log_w_irrelevant_mean": float(log_w[~rel].mean()) if arm == "full" else None,
        }
        print(arm, seed, json.dumps(out[f"{arm}/s{seed}"]), flush=True)
(ROOT.parent / "docs" / "WISP_evidence" / "h5_s1p_diagnosis.json").write_text(
    json.dumps(out, indent=1), encoding="utf-8")
