#!/bin/bash
# LR-5: v3b_golden.py compare and mb0_reference.py compare (E, 1 thread set by the scripts) on one node.
#   srun --jobid=<allocation> --overlap bash docs/legacy_evidence/lr5_gates.sh        (g2n01)
#   sbatch --nodelist=g3an01 docs/legacy_evidence/lr5_gates.sh                         (g3an01)
#SBATCH --job-name=lr5-gates
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/legacy_evidence/lr5_gates_%N_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
F='GBWarning|expandable|allocation pattern|PYTORCH_CUDA|gb_warning|An experimental|See https|and set the|FutureWarning|autocast|_flatten\("", torch.load'
echo "===== v3b_golden.py compare"
python ../docs/legacy_evidence/v3b_golden.py compare ../docs/legacy_evidence/v3b_golden_cluster.pt 2>&1 | grep -vE "${F}"
echo "v3b_compare_exit=${PIPESTATUS[0]}"
echo "===== mb0_reference.py compare"
python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "${F}"
echo "mb0_compare_exit=${PIPESTATUS[0]}"
echo "end=$(date -Is)"
