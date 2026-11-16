#!/bin/bash
# Card 2 D6: one task per line of a D6 list (d6_dryrun.py; records in $DRYRUN_OUT/records, outside the repository).
#   sbatch [resources] --array=1-N%2 d6_array.sbatch <list>          (from /mnt/data/peeradon/h2snn-dryrun/H2S-NN)
# Line forms (whitespace separated; '#' lines are skipped as no-ops):
#   RUN    <cell> <attempt> [d6_dryrun.py options...]          one run (e.g. --point central)
#   LADDER <cell> <attempt> <set_1> <set_2> ...                D6.3: run --set set_1, set_2, ... in order (one grid
#                                                              step at a time) and stop at the first OK
#   CLB    <cell> <attempt> [options...]                       one run with CUDA_LAUNCH_BLOCKING=1 (D6.5)
#SBATCH --job-name=dryrun-d6
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/data/peeradon/dryrun_scratch/logs/%x_%A_%a.out
set -u
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
LIST="$1"
DRYRUN_OUT="${DRYRUN_OUT:-/mnt/data/peeradon/dryrun_scratch/d6}"
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")"
echo "head=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} line=$LINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export WANDB_SILENT=true
read -r KIND CELL ATTEMPT REST <<< "$LINE"
DRV=../docs/dryrun_evidence/d6_dryrun.py
case "$KIND" in
  ''|'#'*) echo "comment line: nothing to run"; exit 0;;
  RUN) python $DRV --cell "$CELL" --out "$DRYRUN_OUT" --attempt "$ATTEMPT" $REST;;
  CLB) CUDA_LAUNCH_BLOCKING=1 python $DRV --cell "$CELL" --out "$DRYRUN_OUT" --attempt "$ATTEMPT" $REST;;
  LADDER)
    step=0
    for SET in $REST; do
      step=$((step + 1))
      echo "LADDER $CELL step $step --set $SET"
      OUT=$(python $DRV --cell "$CELL" --out "$DRYRUN_OUT" --attempt "${ATTEMPT}-s${step}" --set "$SET" 2>&1 | tee /dev/stderr | grep '^RESULT')
      echo "$OUT"
      case "$OUT" in *" OK "*) echo "LADDER $CELL fits at step $step: $SET"; break;; esac
      case "$OUT" in *" OOM "*) ;; *) echo "LADDER $CELL step $step not OOM and not OK: stop the ladder"; break;; esac
    done;;
  *) echo "unknown line kind $KIND"; exit 2;;
esac
