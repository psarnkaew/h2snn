#!/bin/bash
# Model change TD-1, M6: card 4's known-answer reruns on the 1.25 model, exactly card 4's frozen
# settings (scripts/known_answer/common.py), seeds and run counts; only the model changed.
#   KIND=audit  : s3_audit.py (new audit (ii): [x_u, x_v, rel])                   (one job)
#   KIND=s3     : s3.py --task $SLURM_ARRAY_TASK_ID (0-11), after the audit       (array)
#   KIND=s1pb   : s1pb.py --task $SLURM_ARRAY_TASK_ID (0-8 = the `full` runs; TASKS in s1pb.py) (array)
# Runs in a snapshot (snapshot_submit.sh) of the committed head; results go to the worktree's
# docs/model_change_td1_evidence/{k3,k4}. --mem = card 4's recorded host peak 7.6 GB + 30 %.
#SBATCH --job-name=td1-ka
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence/jobs/ka_%x_%A_%a.out
set -uo pipefail
cd "${SNAP:?run through snapshot_submit.sh}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
EVID=/mnt/data/peeradon/h2snn-model/docs/model_change_td1_evidence
echo "kind=${KIND:?} job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} node=$(hostname) start=$(date -Is)"
echo "snapshot=${SNAP} $(cat ${SNAP}/SNAPSHOT.txt)"
echo "git=$(git -C /mnt/data/peeradon/h2snn-model rev-parse HEAD)"
sha256sum scripts/known_answer/*.py src/models/*.py src/configs/config.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "${KIND}" in
  audit) mkdir -p "$EVID/k4"; python scripts/known_answer/s3_audit.py --out "$EVID/k4/s3_audit.json" ;;
  s3)    python scripts/known_answer/s3.py --task "${SLURM_ARRAY_TASK_ID:?}" --audit "$EVID/k4/s3_audit.json" --out_dir "$EVID/k4" ;;
  s1pb)  mkdir -p "$EVID/k3"; python scripts/known_answer/s1pb.py --task "${SLURM_ARRAY_TASK_ID:?}" --out_dir "$EVID/k3" ;;
esac
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
