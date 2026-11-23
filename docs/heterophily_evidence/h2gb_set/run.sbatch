#!/bin/bash
# Card 19: one h2gb_set_measures.py call (or the two card tests) per array task, CPU only (no GPU requested).
#   sbatch --array=1-N%2 --mem=16G --time=24:00:00 docs/heterophily_evidence/h2gb_set/run.sbatch <list file>
# Each line of the list file: the arguments of one call, or "TESTS". The script stops itself 30 min before the limit.
#SBATCH --job-name=c19-het
#SBATCH --cpus-per-task=8
#SBATCH --exclude=g3an01,g2tn01
#SBATCH --output=/mnt/data/peeradon/h2snn-card19/docs/heterophily_evidence/h2gb_set/logs/%x_%A_%a.txt

set -uo pipefail
cd /mnt/data/peeradon/h2snn-card19/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export TQDM_MININTERVAL=300
LIST="$1"
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
LIMIT_S=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*86400+$2*3600+$3*60+$4); else if (NF==3) print ($1*3600+$2*60+$3); else print ($1*60+$2) }')
DEADLINE_H=$(awk -v s="$LIMIT_S" 'BEGIN { printf "%.3f", (s - 1800) / 3600 }')
echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) mem=${SLURM_MEM_PER_NODE:-?}M cpus=${SLURM_CPUS_PER_TASK}"
git status --short -- src scripts tests | sed 's/^/status: /'
echo "args: ${ARGS} deadline_h=${DEADLINE_H}"
if [ "$ARGS" = "TESTS" ]; then
  python -m unittest -v tests.test_card19_h2gb_set
else
  python scripts/heterophily/h2gb_set_measures.py ${ARGS} --deadline-h "$DEADLINE_H"
fi
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
