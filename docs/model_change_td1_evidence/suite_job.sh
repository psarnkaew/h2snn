#!/bin/bash
# Model change TD-1, M5: the full suite on one node (g2n01, the base suite's node), one GPU not shared,
# environment E, in the worktree itself. --mem = the base suite's recorded peak (job 17648, 7.4 GB) + 30 %.
#SBATCH --job-name=td1-suite
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH --nodelist=g2n01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/m5_suite_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
lscpu | grep 'Model name'
python -m unittest discover -s tests -v 2>&1 | grep -E " \.\.\. |^Ran |^OK|^FAILED|^ERROR:|^FAIL:|Error|assert" | grep -vE "Warning" 
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
