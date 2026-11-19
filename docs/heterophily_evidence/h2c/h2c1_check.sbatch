#!/bin/bash
# Card 6 H2c-1: clique condition on ieee-cis-g and the real-data check (h2c1_check.py). CPU only (no GPU requested).
#   sbatch docs/heterophily_evidence/h2c/h2c1_check.sbatch <worktree root> <records dir>
#SBATCH --job-name=s5-het-h2c1-check
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-heterophily-h2c/docs/heterophily_evidence/h2c/logs/h2c1_check_%j.txt
set -uo pipefail
ROOT="$1"; REC="$2"
cd "$ROOT/H2S-NN" || exit 2
mkdir -p ../docs/heterophily_evidence/h2c/check
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- src scripts tests | sed 's/^/status: /'
status=0
for W in survey $(cd "$REC" && ls nc__*__mde.json | sed 's/^nc__//; s/__mde.json$//'); do
    echo "=== $W"
    python ../docs/heterophily_evidence/h2c/h2c1_check.py "$REC" "../docs/heterophily_evidence/h2c/check/h2c1_$W.json" "$W" \
        2>&1 | grep -vE "Warning|warn|torch.load|InMemoryDataset|^\s"
    s=${PIPESTATUS[0]}; [ "$s" -ne 0 ] && { echo "EXIT $W: $s"; status=1; }
done
echo "end=$(date -Is) exit=$status"
exit $status
