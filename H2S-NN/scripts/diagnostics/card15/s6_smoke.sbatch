#!/bin/bash
# Card 15 tests: one smoke epoch on acm_pp for H2S-NN and for GCN (shared LP loop). One GPU.
#SBATCH --job-name=c15-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card15/docs/card15_evidence/slurm-%j-smoke.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card15/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain -- src scripts | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for m in h2snn gcn; do
  python scripts/diagnostics/card15/smoke_one_epoch.py --run "$m:lp:acm_pp" --device cuda \
    --out ../docs/card15_evidence/smoke/${m}_acm_pp.json
  echo "smoke $m exit=$?"
done
echo "end=$(date -Is)"
