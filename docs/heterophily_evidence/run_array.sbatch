#!/bin/bash
# Card 6 H1/H2: one (dataset, task, measure) per array task, CPU only (no GPU requested; D-b).
#   sbatch --array=1-N%3 --mem=<=56G --time=<=12:00:00 docs/heterophily_evidence/run_array.sbatch <list file>
# Each line of the list file: "<dataset> <task> <measure> [<out dir>]" (default out dir: docs/heterophily_evidence). The script stops itself 30 min before the SLURM limit
# (--deadline-h) and writes the record with the reason; memory beyond --mem is reported from sacct.
#SBATCH --job-name=s5-het
#SBATCH --cpus-per-task=8
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-heterophily/docs/heterophily_evidence/logs/%x_%A_%a.txt

set -uo pipefail
cd /mnt/data/peeradon/h2snn-heterophily/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export TQDM_MININTERVAL=300

LIST="$1"
read -r DATASET TASK MEASURE OUT < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
LIMIT_S=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*86400+$2*3600+$3*60+$4); else if (NF==3) print ($1*3600+$2*60+$3); else print ($1*60+$2) }')
DEADLINE_H=$(awk -v s="$LIMIT_S" 'BEGIN { printf "%.3f", (s - 1800) / 3600 }')
echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) mem=${SLURM_MEM_PER_NODE:-?}M cpus=${SLURM_CPUS_PER_TASK}"
echo "task: dataset=${DATASET} task=${TASK} measure=${MEASURE} out=${OUT:-default} deadline_h=${DEADLINE_H}"
python scripts/heterophily_stats.py --dataset "$DATASET" --task "$TASK" --measure "$MEASURE" --deadline-h "$DEADLINE_H" ${OUT:+--out-dir "$OUT"}
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
