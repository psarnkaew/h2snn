#!/bin/bash
# Card 2 D2: run lines FIRST..LAST of a cell list sequentially inside the author's allocation (srun --jobid --overlap),
# same driver and records as d2_array.sbatch. Run from /mnt/data/peeradon/h2snn-dryrun/H2S-NN.
set -u
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
LIST="$1"; FIRST="$2"; LAST="$3"; ATTEMPT="${4:-24GB}"
DRYRUN_OUT="${DRYRUN_OUT:-/mnt/data/peeradon/dryrun_scratch/d2}"
export WANDB_SILENT=true
INC=1; [ "$FIRST" -gt "$LAST" ] && INC=-1   # FIRST > LAST: from the end of the list (claims avoid duplicates)
for i in $(seq "$FIRST" "$INC" "$LAST"); do
  LINE="$(sed -n "${i}p" "$LIST")"
  CELL="${LINE%% *}"; STEP=""   # optional second field: axis=value (card 2 step-down after an OOM)
  case "$LINE" in *" "*) STEP="--step_down ${LINE#* }";; esac
  LOG=/mnt/data/peeradon/dryrun_scratch/logs/alloc_${SLURM_JOB_ID}_${ATTEMPT}_line${i}.out
  { echo "head=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID step=$SLURM_STEP_ID line=$i cell=$CELL attempt=$ATTEMPT $STEP"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    timeout 3h python ../docs/dryrun_evidence/d2_dryrun.py --cell "$CELL" --out "$DRYRUN_OUT" --attempt "$ATTEMPT" $STEP
    echo "exit=$?"; } > "$LOG" 2>&1
  grep -h "^RESULT" "$LOG" || echo "NO RESULT line $i $CELL (see $LOG)"
done
