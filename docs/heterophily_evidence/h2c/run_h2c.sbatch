#!/bin/bash
# Card 6 H2c: one heterophily_stats.py call per array task, from the given worktree; CPU only (no GPU requested).
# Each line of the list file holds the arguments of one call. The script stops itself 30 min before the SLURM limit.
#   sbatch --array=1-N%8 --mem=16G --time=12:00:00 docs/heterophily_evidence/h2c/run_h2c.sbatch <worktree root> <list>
#SBATCH --job-name=s5-het-h2c
#SBATCH --cpus-per-task=8
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-heterophily-h2c/docs/heterophily_evidence/h2c/logs/%x_%A_%a.txt

set -uo pipefail
ROOT="$1"; LIST="$2"
cd "$ROOT/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export TQDM_MININTERVAL=300
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
LIMIT_S=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*86400+$2*3600+$3*60+$4); else if (NF==3) print ($1*3600+$2*60+$3); else print ($1*60+$2) }')
DEADLINE_H=$(awk -v s="$LIMIT_S" 'BEGIN { printf "%.3f", (s - 1800) / 3600 }')
echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) mem=${SLURM_MEM_PER_NODE:-?}M cpus=${SLURM_CPUS_PER_TASK}"
git status --short -- src scripts tests | sed 's/^/status: /'
echo "args: ${ARGS} deadline_h=${DEADLINE_H}"
python scripts/heterophily_stats.py ${ARGS} --deadline-h "$DEADLINE_H"
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
