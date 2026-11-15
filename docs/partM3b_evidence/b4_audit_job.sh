#!/bin/bash
# Part M3b B4 data check: venue-leak audit of the three OAG loaders.
#   sbatch docs/partM3b_evidence/b4_audit_job.sh
#SBATCH --job-name=m3b-b4-audit
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/partM3b_evidence/b4_audit_%j.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
python scripts/data/h2gb_leak_audit.py --root data --out ../docs/partM3b_evidence/b4_audit.json oag-cs oag-eng oag-chem 2>&1 | grep -vE "GBWarning|warnings.warn|FutureWarning|weights_only|An experimental|allocation pattern|See https|and set the|if you want to disable"
status=${PIPESTATUS[0]}
echo "end=$(date -Is) exit=${status}"
exit "$status"
