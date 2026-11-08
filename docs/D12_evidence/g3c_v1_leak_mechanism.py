"""G3c v1 leak evidence: how no_path_cells (seed 42, selected checkpoint) binds the lead bit without
path cells. Per target P: route strength Q -> A -> P through the lead author vs the non-lead author,
    route = g_P * |Dhat_PA| * b_A * g_A * |Dhat_AQ| * b_Q,   |Dhat_uv| = c_u c_v / sqrt(D_u D_v)
(per-edge maps are scaled orthogonal frames, so the block norm is that scalar), plus the gates alone.
Run from H2S-NN/:  python ../docs/D12_evidence/g3c_v1_leak_mechanism.py   (CPU, eval mode)"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "docs" / "D12_evidence"))
import g3c_binding as g3c  # noqa: E402
import src.data.dataset as dataset_module  # noqa: E402
from src.data.dataset import build_dataset  # noqa: E402
from src.models.h2snn_model import build_model  # noqa: E402

graph, truth, _ = g3c.graph_v1()
dataset_module._load_hetero_data = lambda _config: graph.clone()
cfg = g3c.config("v1", "no_path_cells", 42)
data = build_dataset(cfg)
model = build_model(cfg, data)
model.load_state_dict(torch.load(cfg.exp_dir / "checkpoints" / "best.pt", map_location="cpu")["state_dict"])
model.eval()
with torch.no_grad():
    h0 = model.encode_nodes(data.x, data.node_type)
    operator = model.build_operator(h0, data.edge_index, data.edge_type, data.face_groups, data.num_nodes)
    g, b = operator.listen, operator.broadcast
    lap = operator.laplacian
    degree = lap.degree()
    c_src, c_tgt = lap.edge_maps.src.scale, lap.edge_maps.tgt.scale
    src, tgt = lap.src_idx, lap.tgt_idx
    block = c_src * c_tgt / (degree[src] * degree[tgt]).sqrt()
edge_of = {(min(u, v), max(u, v)): e for e, (u, v) in enumerate(zip(src.tolist(), tgt.tolist()))}
paper0, author0 = data.node_slices["paper"][0], data.node_slices["author"][0]


def route(t: int, middle: str, far: str) -> float:
    p, a, q = paper0 + t, author0 + int(truth[middle][t]), paper0 + int(truth[far][t])
    pa, aq = edge_of[(min(p, a), max(p, a))], edge_of[(min(a, q), max(a, q))]
    return float(g[p] * block[pa] * b[a] * g[a] * block[aq] * b[q])


targets = range(g3c.NT)
lead = torch.tensor([route(t, "L", "qstar") for t in targets])
other = torch.tensor([route(t, "N", "qprime") for t in targets])
L_ids, N_ids = author0 + truth["L"], author0 + truth["N"]
out = {
    "run": "v1/no_path_cells/s42, selected checkpoint",
    "route_strength_mean": {"through_lead_author": float(lead.mean()), "through_non_lead": float(other.mean())},
    "route_ratio_lead_over_non_lead": {"median": float((lead / other).median()), "min": float((lead / other).min())},
    "author_gates_mean": {"lead": {"listen_g": float(g[L_ids].mean()), "broadcast_b": float(b[L_ids].mean())},
                          "non_lead": {"listen_g": float(g[N_ids].mean()), "broadcast_b": float(b[N_ids].mean())}},
    "edge_block_mean": {
        "P-lead": float(torch.tensor([block[edge_of[tuple(sorted((paper0 + t, author0 + int(truth['L'][t]))))]]
                                      for t in targets]).mean()),
        "P-non_lead": float(torch.tensor([block[edge_of[tuple(sorted((paper0 + t, author0 + int(truth['N'][t]))))]]
                                          for t in targets]).mean())},
}


def test_accuracy() -> float:
    with torch.no_grad():
        logits = model(data).logits
    index = data.target_indices[data.target_test_mask]
    return float((logits[index].argmax(-1) == data.y[index]).float().mean())


# Intervention: the same checkpoint with the relational context c set to 0 (maps, gates read [h0, 0]).
out["test_acc_checkpoint"] = test_accuracy()
original = model.relational_context.forward
model.relational_context.forward = lambda h, ei, et: torch.zeros_like(original(h, ei, et))
out["test_acc_context_zeroed"] = test_accuracy()
model.relational_context.forward = original
print(json.dumps(out, indent=1))
(ROOT.parent / "docs" / "D12_evidence" / "g3c_v1_leak_mechanism.json").write_text(json.dumps(out, indent=1),
                                                                                  encoding="utf-8")
