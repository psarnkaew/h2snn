#!/bin/bash
# Card 4 K4 (S3, tasks 0-11) and K5 (S2', tasks 12-17): one known-answer run per array task, from the
# detached snapshot worktree /mnt/data/peeradon/h2snn-ablprep-run (src at the committed K2 head f1601b8,
# scripts/known_answer copied, sha256 printed) so that K6 edits in the working worktree never reach a
# queued run. Results go to the working worktree's evidence folder.
#   sbatch --array=0-17 docs/card4_evidence/jobs/known_answer_array.sbatch
#SBATCH --job-name=card4-ka
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-ablprep-run/H2S-NN/logs/ka-%A_%a.out
set -uo pipefail
RUN=/mnt/data/peeradon/h2snn-ablprep-run/H2S-NN
EVID=/mnt/data/peeradon/h2snn-ablprep/docs/card4_evidence
cd "$RUN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) dirty: $(git status --short -- src scripts | tr '\n' ' ')"
sha256sum scripts/known_answer/*.py src/models/*.py src/configs/config.py
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
t=${SLURM_ARRAY_TASK_ID:?}
if [ "$t" -lt 12 ]; then
    python scripts/known_answer/s3.py --task "$t" --audit "$EVID/k4/s3_audit.json" --out_dir "$EVID/k4"
else
    python scripts/known_answer/s2p.py --task $((t - 12)) --out_dir "$EVID/k5"
fi
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
