#!/bin/bash
# Part M3b B5 data check: PDNS through H2GB's own preformat_PDNS (seed 0) and its time windows.
#   sbatch docs/partM3b_evidence/b5_pdns_job.sh
#SBATCH --job-name=m3b-b5-pdns
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=3:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/partM3b_evidence/b5_pdns_%j.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
echo "data/NC/PDNS before: $(ls data/NC/PDNS | tr '\n' ' ')"
python scripts/data/h2gb_pdns_check.py --root data --out ../docs/partM3b_evidence/b5_pdns_check.json --seed 0 2>&1 | grep -vE "GBWarning|warnings.warn|FutureWarning|weights_only|An experimental|allocation pattern|See https|and set the|if you want to disable"
status=${PIPESTATUS[0]}
echo "data/NC/PDNS after: $(ls data/NC/PDNS | tr '\n' ' ')"
echo "end=$(date -Is) exit=${status}"
exit "$status"
