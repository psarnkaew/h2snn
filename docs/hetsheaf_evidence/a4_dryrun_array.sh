#!/bin/bash
# Card 3 A4 dry run, one cell per array task on a 24 GB GPU not shared (design session A4 plan): CELLS[$SLURM_ARRAY_TASK_ID]
# at the largest point; an OOM runs once more with hidden_channels 32 in the same job. No metric.
#   sbatch --array=0-18%3 docs/hetsheaf_evidence/a4_dryrun_array.sh <out dir>
#SBATCH --job-name=hetsheaf-a4
#SBATCH --gres=gpu:1
#SBATCH --exclude=g2n01,g3an01
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/H2S-NN/logs/slurm-%A_%a.out
set -uo pipefail
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none} id=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
sha256sum src/baselines/adapters/hetsheaf.py ../docs/hetsheaf_evidence/a4_dryrun.py
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
mkdir -p "$1"
python ../docs/hetsheaf_evidence/a4_dryrun.py --index "$SLURM_ARRAY_TASK_ID" --out "$1/cell_${SLURM_ARRAY_TASK_ID}.jsonl"
echo "end=$(date -Is)"
