#!/bin/bash
# Part M3b B2 data check: count one H2GB NC dataset (or --compare-mag) as H2GB loads it.
#   sbatch --job-name=m3b-b2-<key> docs/partM3b_evidence/b2_counts_job.sh <key>|compare-mag
# Output: docs/partM3b_evidence/b2_counts_<key>.json and this job's log; peak RSS recorded by the script (getrusage) and by sacct MaxRSS.
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/partM3b_evidence/b2_counts_%x_%j.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KEY="$1"
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) key=${KEY} start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
OUT=../docs/partM3b_evidence/b2_counts_${KEY}.json
if [ "$KEY" = compare-mag ]; then ARGS=(--compare-mag); else ARGS=("$KEY"); fi
python scripts/data/h2gb_nc_counts.py --root data --out "$OUT" "${ARGS[@]}" 2>&1 | grep -vE "GBWarning|warnings.warn|FutureWarning|weights_only|An experimental|allocation pattern|See https|and set the|if you want to disable"
status=${PIPESTATUS[0]}
echo "end=$(date -Is) exit=${status}"
exit "$status"
