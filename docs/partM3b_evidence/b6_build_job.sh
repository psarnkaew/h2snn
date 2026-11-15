#!/bin/bash
# Part M3b B6 data job: build one large LP task (split, message-graph check, HGB negatives).
#   sbatch --job-name=m3b-b6-<task> docs/partM3b_evidence/b6_build_job.sh <task>
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/partM3b_evidence/b6_build_%x_%j.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK="$1"
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) task=${TASK} start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
python scripts/lp/build_large_lp.py --root data --task "$TASK" --report ../docs/partM3b_evidence/b6_build_${TASK}.json --head-chunk 512 2>&1 | grep -vE "GBWarning|warnings.warn|FutureWarning|weights_only|An experimental|allocation pattern|See https|and set the|if you want to disable"
status=${PIPESTATUS[0]}
ls -la "data/LP/large/${TASK}" 2>/dev/null
echo "end=$(date -Is) exit=${status}"
exit "$status"
