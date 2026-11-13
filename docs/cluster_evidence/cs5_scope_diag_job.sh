#!/bin/bash
# CS5 diagnostic (no code change): size of the difference in
# test_partMB_scopes.test_featureless_node_in_two_scopes_has_its_global_h0 on this platform.
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn || exit 2
echo "job=$SLURM_JOB_ID node=$(hostname) cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//')"
python - <<'PY'
import torch
from tests.test_partMB_scopes import example_graph, example_cells, cells_of_paper_1, build_scope_batch
from src.configs.config import TrainConfig
from src.models.h2snn_model import H2SNN, ModelDims
print("torch", torch.__version__, "mkldnn", torch.backends.mkldnn.is_available(), "threads", torch.get_num_threads())
graph = example_graph()
cfg = TrainConfig.from_dict({"device": "cpu", "data": {"dataset_name": "acm", "wisp_walks_per_anchor": 2},
    "model": {"d": 8, "d_r": 4, "edge_mlp_hidden_dim": 16, "face_mlp_hidden_dim": 16,
              "classifier_hidden_dim": 16, "propagation": "anchored", "arm": "full"},
    "ablation": {"coupling": "symmetric"}})
for threads in (torch.get_num_threads(), 1):
    torch.set_num_threads(threads); torch.manual_seed(0)
    model = H2SNN(cfg, ModelDims.from_prepared_data(graph)).eval()
    batch = build_scope_batch(graph, torch.tensor([0, 1]), [example_cells(), cells_of_paper_1()])
    with torch.no_grad():
        whole = model.encode_nodes(graph.x, graph.node_type)
        scoped = model.encode_nodes(batch.data.x, batch.data.node_type, batch.data.global_node_id)
    ref = whole[batch.global_node_id]
    diff = (scoped - ref).abs()
    print(f"threads={threads} equal={torch.equal(scoped, ref)} max|diff|={diff.max().item():.3e} rows_differing={(diff.amax(1) > 0).sum().item()}/{diff.shape[0]}")
PY
