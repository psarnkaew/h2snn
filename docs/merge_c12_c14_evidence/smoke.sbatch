#!/bin/bash
# Merge c12+c14 step 5: one-epoch smoke on the merged code with card 14's smoke.py (finite loss; PCGConvergenceError
# is recorded as the outcome).
#   sbatch --gres=gpu:1 --mem=<peak+30%> docs/merge_c12_c14_evidence/smoke.sbatch --config <yaml> --name <n> [--set k=v]
#SBATCH --job-name=mc-smoke
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-merge/H2S-NN/logs/slurm-%j.out
set -uo pipefail
cd /mnt/data/peeradon/h2snn-merge/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
start=$(date +%s)
python -u scripts/diagnostics/card14/smoke.py --out /mnt/data/peeradon/merge_c12_c14_runs/smoke "$@"
status=$?
echo "end=$(date -Is) exit=${status} wall_s=$(( $(date +%s) - start ))"
exit "${status}"
