#!/bin/bash
# Card 9: targeted unit tests (CPU, environment h2snn-b) on the worktree; the full suite is suite_job.sh.
#   sbatch docs/card9_evidence/tests_job.sh <test id> [<test id> ...]
#SBATCH --job-name=c9-tests
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card9_evidence/tests_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b MPLBACKEND=Agg
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "dirty: $(git status --short -- scripts tests | tr '\n' ' ')"
python -m unittest -v "$@"
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
