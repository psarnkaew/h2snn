#!/bin/bash
# Part M3b B7: enumeration tests and the H-Pokec meta-path cost (schema only; no data loaded).
#   srun --jobid=<allocation> --overlap bash docs/partM3b_evidence/b7_job.sh
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
F="GBWarning|warnings.warn|FutureWarning|An experimental|allocation pattern|See https|and set the|if you want to disable"
python -m unittest -v tests.test_metapath_counts 2>&1 | grep -vE "$F" | tail -n 6
python scripts/baselines/h2gb_metapath_cost.py --counts ../docs/partM3b_evidence/h2gb_nc_counts.json --out ../docs/partM3b_evidence/b7_metapath_cost.json 2>&1 | grep -vE "$F"
echo "end=$(date -Is)"
