#!/bin/bash
# Card 5: one CPU-only Python step from H2S-NN/ of this worktree (sbatch, no GPU).
set -uo pipefail
cd /mnt/data/peeradon/h2snn-ablprep/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
echo "dirty: $(git status --short -- src tests scripts | tr '\n' ' ')"
python "$@"; status=$?
echo "end=$(date -Is) exit=${status}"; exit $status
