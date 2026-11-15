#!/bin/bash
# M3a-A3b: stale DGL install messages replaced (messages only): DglCudaGateTests and RequirementGroupTests on a
# GPU node, the real dgl_cuda_report() of h2snn-b, then the gate. With E exported.
#   srun --jobid=<allocation> --overlap bash docs/partM3a_evidence/a3b_job.sh   (or sbatch --gres=gpu:1)
#SBATCH --job-name=m3a-a3b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:20:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
nvidia-smi --query-gpu=name --format=csv,noheader
git status --short -- . | grep -v '^??' | sed 's/^/working tree: /'
echo "===== tests"
python -m unittest -v tests.test_baselines.DglCudaGateTests tests.test_baselines.RequirementGroupTests 2>&1
echo "tests_exit=$?"
echo "===== the messages on this machine (real DGL 2.4.0+cu124 with CUDA) and with a CPU-only DGL patched in"
python - <<'EOF' 2>&1 | grep -vE "GBWarning|expandable|allocation|PYTORCH_CUDA|gb_warning|^$|An experimental|See https|and set the"
import sys
from unittest import mock
from src.baselines.runtime import dependencies as dep
print("real:", dep.dgl_cuda_report())
with mock.patch("src.baselines.runtime._dgl_compat.dgl_supports_cuda", return_value=False), \
        mock.patch.object(sys, "platform", "linux"):
    print("cpu-only DGL, linux, report:", dep.dgl_cuda_report()[1])
    print("cpu-only DGL, linux, defect:", dep._dgl_cuda_defect())
EOF
echo
bash ../docs/partM3a_evidence/gate.sh
echo "end=$(date -Is)"
