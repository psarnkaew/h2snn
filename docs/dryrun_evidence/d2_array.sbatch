#!/bin/bash
# Card 2 D2: dry run, one cell per array task.  sbatch [resources] --array=1-N d2_array.sbatch <cell list> <attempt>
# Run from /mnt/data/peeradon/h2snn-dryrun/H2S-NN. Records go to $DRYRUN_OUT/records (outside the repository);
# logs (which hold the trainers' own console lines) stay outside the repository too.
#SBATCH --job-name=dryrun
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/data/peeradon/dryrun_scratch/logs/%x_%A_%a.out
set -u
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
LIST="$1"; ATTEMPT="${2:-24GB}"
DRYRUN_OUT="${DRYRUN_OUT:-/mnt/data/peeradon/dryrun_scratch/d2}"
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")"
CELL="${LINE%% *}"; STEP=""   # optional second field: axis=value (card 2 step-down after an OOM)
case "$LINE" in *" "*) STEP="--step_down ${LINE#* }";; esac
echo "head=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} cell=$CELL attempt=$ATTEMPT $STEP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export WANDB_SILENT=true
python ../docs/dryrun_evidence/d2_dryrun.py --cell "$CELL" --out "$DRYRUN_OUT" --attempt "$ATTEMPT" $STEP
