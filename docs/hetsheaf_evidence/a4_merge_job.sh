#!/bin/bash
# Card 3 A4: after merging v3b 7df909d: regenerate the notebooks with the unchanged generator, then unit tests (CPU only).
#   sbatch docs/hetsheaf_evidence/a4_merge_job.sh "<files to hash>" <targets...>
#SBATCH --job-name=hetsheaf-merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/H2S-NN/logs/slurm-%j.out
set -uo pipefail
FILES="$1"; shift
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) merging=$(cat ../.git/MERGE_HEAD 2>/dev/null || git rev-parse MERGE_HEAD 2>/dev/null)"
for f in $FILES; do sha256sum "$f"; done
python scripts/generate_baseline_notebooks.py && echo "generator exit=0"
git status --short -- notebooks | sed 's/^/notebooks: /'
python -m unittest -v "$@" 2>&1 | grep -vE "Warning|warn\(|^\s*$|An experimental"
echo "exit=${PIPESTATUS[0]} end=$(date -Is)"
