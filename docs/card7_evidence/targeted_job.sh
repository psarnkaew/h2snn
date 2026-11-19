#!/bin/bash
# Card 7: targeted unittest modules on CPU (environment E), no GPU, in the worktree itself (a snapshot with a
# symlinked vendor/ breaks vendor isolation, model-td1 M8 job 17756).
#   sbatch docs/card7_evidence/targeted_job.sh <label> tests.test_x [tests.test_y ...]
# --mem: no recorded peak for the card 7 tests; the TD-1 targeted CPU jobs peaked at 0.4 GB (17689, 17693).
#SBATCH --job-name=c7-tgt
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/targeted_%j.txt
set -uo pipefail
LABEL="${1:?label}"; shift
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MPLBACKEND=Agg
echo "label=${LABEL} job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_sha256=$(git diff HEAD | sha256sum | cut -c1-16)"
git status --short | grep -v '^??' | sed 's/^/working tree: /'
python -m unittest -v "$@" 2>&1 | grep -vE "GBWarning|FutureWarning|warnings.warn|UserWarning" | tail -n 150
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
