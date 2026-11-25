#!/bin/bash
# Card 23: mb0_reference.py compare against the cluster reference on b0f13fb (runs checkout, read only) and on
# card23, to show whether the differences of job 19004 predate card 23. run_tests.sbatch environment.
#SBATCH --job-name=c23-mb0
#SBATCH --partition=lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card23/docs/card23_evidence/job%j_mb0_base.log
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export PYTHONDONTWRITEBYTECODE=1
REF=/mnt/data/peeradon/h2snn-card23/docs/partMB_evidence/mb0_reference_cluster.pt
TMP=$(mktemp -d)
echo "job=${SLURM_JOB_ID} node=$(hostname)"
for tree in /mnt/data/peeradon/h2snn-runs/b0f13fb43bcc77bdf15f73a2caa026140179e441 /mnt/data/peeradon/h2snn-card23; do
  cd $tree/H2S-NN && echo "=== tree=$tree git=$(git rev-parse HEAD)"
  python ../docs/partMB_evidence/mb0_reference.py compare $REF 2>/dev/null | grep -v "^  DIFF" ; echo "exit=${PIPESTATUS[0]}"
  python ../docs/partMB_evidence/mb0_reference.py save $TMP/$(basename $tree).pt 2>/dev/null; echo "save exit=$?"
done
echo "=== card23 compare against the b0f13fb capture of this job"
cd /mnt/data/peeradon/h2snn-card23/H2S-NN && python ../docs/partMB_evidence/mb0_reference.py compare $TMP/b0f13fb43bcc77bdf15f73a2caa026140179e441.pt 2>/dev/null; echo "exit=$?"
