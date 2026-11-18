#!/bin/bash
# Model change TD-1: targeted unittest modules on CPU (environment E), no GPU.
#   sbatch docs/model_change_td1_evidence/targeted_job.sh <label> tests.test_x [tests.test_y ...]
#SBATCH --job-name=td1-tgt
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/targeted_%j.txt
set -uo pipefail
LABEL="${1:?label}"; shift
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "label=${LABEL} job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
python -m unittest "$@" 2>&1 | grep -vE "GBWarning|FutureWarning|warnings.warn|UserWarning" | tail -n 120
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
