#!/bin/bash
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 1
echo "job=$SLURM_JOB_ID node=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi | head -4
gcc --version | head -1; g++ --version | head -1
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn
which python; python --version
echo "--- curl data.dgl.ai from compute node"
curl -sI --max-time 20 https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html | head -3; echo "curl_exit=${PIPESTATUS[0]}"
echo "--- imports"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "torch.version.cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0)); x=torch.ones(3,device="cuda"); print("cuda_tensor_sum", float(x.sum()))
for m in ["torch_geometric","torch_scatter","torch_sparse","torch_cluster","dgl","torch_householder"]:
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e:
        print(m, "IMPORT FAILED:", type(e).__name__, str(e)[:300])
try:
    from src.baselines.runtime._dgl_compat import dgl_supports_cuda
    print("dgl_supports_cuda", dgl_supports_cuda())
    from src.baselines.runtime.dependencies import _dgl_cuda_defect
    print("_dgl_cuda_defect", _dgl_cuda_defect())
except Exception as e:
    print("dgl check FAILED:", type(e).__name__, str(e)[:300])
PY
pip freeze > ../docs/cluster_evidence/server_pip_freeze.txt; echo "pip_freeze_lines=$(wc -l < ../docs/cluster_evidence/server_pip_freeze.txt)"
