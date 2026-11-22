#!/bin/bash
# Card 14: one job (bench, smoke or check). GPU type, mem and time are given at submission:
#   sbatch --gres=gpu:3090:1 --mem=24G --time=1:00:00 --job-name=c14-x \
#       scripts/diagnostics/card14/job.sbatch <script under scripts/diagnostics/card14> [args...]
# The script runs with the card14 worktree's interpreter env; bench.py imports the tree given by --root.
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/data/peeradon/card14_runs/logs/slurm-%j.out

set -uo pipefail
cd /mnt/data/peeradon/h2snn-card14/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
md5sum scripts/diagnostics/card14/*.py | sed "s/^/md5 /"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
script="$1"; shift
start=$(date +%s)
python -u "scripts/diagnostics/card14/${script}" "$@"
status=$?
echo "end=$(date -Is) exit=${status} wall_s=$(( $(date +%s) - start ))"
exit "${status}"
