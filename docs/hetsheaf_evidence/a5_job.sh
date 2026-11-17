#!/bin/bash
# Card 3 A5: one cell per job (A4's Rule). Mode `ladder` (A5.2) or `central` (A5.3); resources set at sbatch time
# (--mem = A4 host peak + 30 %, >= 16G; 60G only for H-Pokec; g3an01 excluded; g2n01 only for the GC cells).
#   sbatch --mem=16G --exclude=g2n01,g3an01 docs/hetsheaf_evidence/a5_job.sh ladder 0 docs/hetsheaf_evidence/a5_ladder
#SBATCH --job-name=hetsheaf-a5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/H2S-NN/logs/slurm-%j.out
set -uo pipefail
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is) mode=$1 index=$2"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
sha256sum src/baselines/adapters/hetsheaf.py ../docs/hetsheaf_evidence/a4_dryrun.py ../docs/hetsheaf_evidence/a5_dryrun.py
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
mkdir -p "../$3"
python ../docs/hetsheaf_evidence/a5_dryrun.py "--$1" "$2" --out "../$3/${1}_$2.jsonl"
echo "end=$(date -Is)"
