#!/bin/bash
# Card 3 A4 dry run (one epoch + validation per HetSheaf cell; no metric).
#   srun --jobid=16741 --overlap -n1 ... bash docs/hetsheaf_evidence/a4_dryrun.sh <cells|all> <out.jsonl>
#   sbatch --gres=gpu:a6000:1 ... docs/hetsheaf_evidence/a4_dryrun.sh <cells> <out.jsonl>   (OOM retry)
#SBATCH --job-name=hetsheaf-a4
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/H2S-NN/logs/slurm-%j.out
set -uo pipefail
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
sha256sum src/baselines/adapters/hetsheaf.py ../docs/hetsheaf_evidence/a4_dryrun.py
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python ../docs/hetsheaf_evidence/a4_dryrun.py --cells "$1" --out "$2"
echo "end=$(date -Is)"
