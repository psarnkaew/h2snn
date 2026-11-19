#!/bin/bash
# Card 7 amendments 9-10: targeted unittest modules on CPU (environment E) in the worktree itself, full verbose log
# (targeted_job.sh keeps the last 150 lines only), then baselines1_vs_card5_capture.py.
#   sbatch docs/card7_evidence/targeted_full_job.sh <label> tests.test_x [tests.test_y ...]
# --mem: job 17885 (the same modules) peaked at 0.9 GB; 4G as targeted_job.sh.
#SBATCH --job-name=c7-tgtF
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
python -m unittest -v "$@" 2>&1 | grep -vE "GBWarning|FutureWarning|warnings.warn|UserWarning"
echo "tests exit=${PIPESTATUS[0]}"
echo "== baselines-1 list vs card 5's capture"
python ../docs/card7_evidence/baselines1_vs_card5_capture.py 2>&1 | grep -vE "Warning|warn"
echo "end=$(date -Is)"
