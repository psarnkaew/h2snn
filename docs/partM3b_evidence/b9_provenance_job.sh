#!/bin/bash
# Part M3b B9 item 7: the H2GB section of data/PROVENANCE.md (sha256 of the raw files; reads data/, writes evidence).
#   srun --jobid=<allocation> --overlap bash docs/partM3b_evidence/b9_provenance_job.sh
set -euo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
python scripts/data/h2gb_provenance.py --root data --counts ../docs/partM3b_evidence/h2gb_nc_counts.json \
    --out ../docs/partM3b_evidence/h2gb_provenance_section.md
echo "end=$(date -Is)"
