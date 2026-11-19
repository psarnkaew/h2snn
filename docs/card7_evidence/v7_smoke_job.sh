#!/bin/bash
# Card 7 V7 Phase A: the timing script's lab-GPU smoke on IMDB, 1 epoch, H2S-NN and one baseline (GCN), both at the
# card 2 central point (nothing is tuned yet); then the method lists of ogbn-mag and IMDB. One GPU not shared; never
# the A6000. --mem: no recorded peak for an IMDB timing run (16G chosen; sacct peak in the report).
#SBATCH --job-name=c7-time
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/v7_smoke_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b WANDB_MODE=disabled
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MPLBACKEND=Agg
E=/mnt/data/peeradon/h2snn-model/docs/card7_evidence
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_sha256=$(git diff HEAD | sha256sum | cut -c1-16)"
git status --short | grep -v '^??' | sed 's/^/working tree: /'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
F='GBWarning|FutureWarning|warnings.warn|UserWarning|DeprecationWarning|import torch_geometric'
for m in h2snn gcn; do
  python scripts/timing/time_methods.py --method $m --dataset imdb --epochs 1 --device cuda --out $E/v7_smoke/imdb_${m}.json 2>&1 | grep -vE "$F" | tail -n 15
  echo "${m}_exit=${PIPESTATUS[0]}"
done
for d in imdb ogbn-mag; do echo "== main-table methods on $d"; python scripts/timing/time_methods.py --list --dataset $d 2>&1 | grep -vE "$F"; done
echo "end=$(date -Is)"
