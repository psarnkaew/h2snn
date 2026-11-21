#!/bin/bash
# Card 11: one diagnosis job. Usage (GPU type, mem and time are given at submission):
#   sbatch --gres=gpu:3090:1 --mem=24G --time=1:00:00 --job-name=c11-x \
#       scripts/diagnostics/card11/job.sbatch <python script under scripts/diagnostics/card11> [args...]
# Outputs (checkpoints, logs) go to /mnt/data/peeradon/card11_runs (outside the repository).
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/data/peeradon/card11_runs/logs/slurm-%j.out

set -uo pipefail
cd /mnt/data/peeradon/h2snn-card11/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
md5sum scripts/diagnostics/card11/*.py | sed "s/^/md5 /"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
script="$1"; shift
start=$(date +%s)
python -u "scripts/diagnostics/card11/${script}" "$@"
status=$?
echo "end=$(date -Is) exit=${status} wall_s=$(( $(date +%s) - start ))"
exit "${status}"
