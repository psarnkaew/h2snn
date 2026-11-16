#!/bin/bash
# Card 3 Part A: unittest targets in a CPU step (no GPU needed); prints HEAD, node, job, sha256 of the given files.
#   srun --jobid=<allocation> --overlap bash docs/hetsheaf_evidence/test_job.sh "<files to hash>" <targets...>
set -uo pipefail
FILES="$1"; shift
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
for f in $FILES; do sha256sum "$f"; done
python -m unittest -v "$@" 2>&1 | grep -vE "Warning|warn\(|^\s*$|An experimental"
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
