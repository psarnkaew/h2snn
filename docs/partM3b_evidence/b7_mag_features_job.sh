#!/bin/bash
# Part M3b B7: is the B2 loader's ogbn-mag paper feature the same array as OGB's own (data/NC/mag/ogbn_mag)?
#   srun --jobid=<allocation> --overlap bash docs/partM3b_evidence/b7_mag_features_job.sh
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
python - <<'PY' 2>&1 | grep -vE "GBWarning|warnings.warn|FutureWarning|weights_only|An experimental|allocation pattern|See https|and set the|if you want to disable"
import torch
from src.data.h2gb_nc import load_h2gb_nc
ogb = torch.load("data/NC/mag/ogbn_mag/processed/geometric_data_processed.pt", weights_only=False)
graph = ogb[0]
print("OGB processed object:", type(graph).__name__, "keys:", sorted(graph.keys()) if hasattr(graph, "keys") else "?")
x_ogb = graph["x_dict"]["paper"] if isinstance(graph, dict) else graph.x_dict["paper"]
h2gb = load_h2gb_nc("ogbn-mag", "data")
x_h2gb = h2gb["paper"].x
print("paper x shapes", tuple(x_ogb.shape), tuple(x_h2gb.shape), "dtypes", x_ogb.dtype, x_h2gb.dtype)
print("paper x identical (torch.equal):", bool(torch.equal(x_ogb.to(x_h2gb.dtype), x_h2gb)))
nodes = graph["num_nodes_dict"] if isinstance(graph, dict) else graph.num_nodes_dict
print("OGB num_nodes_dict:", nodes)
print("OGB types with features:", list((graph["x_dict"] if isinstance(graph, dict) else graph.x_dict).keys()))
print("B2 loader feature dims:", {t: tuple(h2gb[t].x.shape) for t in h2gb.node_types})
PY
echo "end=$(date -Is)"
