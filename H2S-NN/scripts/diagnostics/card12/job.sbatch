#!/bin/bash
# Card 12: one diagnosis job. Usage (GPU type, mem and time are given at submission):
#   sbatch --gres=gpu:3090:1 --mem=24G --time=1:00:00 --job-name=c12-x \
#       scripts/diagnostics/card12/job.sbatch <python script under scripts/diagnostics/card1{1,2}> [args...]
# The script path is relative to scripts/diagnostics (e.g. card11/run_from_config.py).
# Outputs (checkpoints, logs) go to /mnt/data/peeradon/card12_runs (outside the repository).
# CARD12_PINNED=1 sets the pinned CPU kernel environment E (as run_tests.sbatch).
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/data/peeradon/card12_runs/logs/slurm-%j.out

set -uo pipefail
cd /mnt/data/peeradon/h2snn-card12/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export WANDB_MODE=disabled
if [ "${CARD12_PINNED:-0}" = "1" ]; then
    export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
fi
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
md5sum src/models/anchored_solve.py scripts/diagnostics/card11/*.py scripts/diagnostics/card12/*.py | sed "s/^/md5 /"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
script="$1"; shift
start=$(date +%s)
python -u "scripts/diagnostics/${script}" "$@"
status=$?
echo "end=$(date -Is) exit=${status} wall_s=$(( $(date +%s) - start ))"
exit "${status}"
