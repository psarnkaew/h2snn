#!/bin/bash
# Card 15 step 2: card 12 unit L degree check (a1_data_checks.py + l_flags.py, copied unchanged) on acm_pp,
# dblp_ap alongside as reference (card 11's reference trial config, as in card 12 job 18249).
#SBATCH --job-name=c15-degree
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=0:30:00
#SBATCH --exclude=g2tn01,g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card15/docs/card15_evidence/slurm-%j-degree.log
set -euo pipefail
cd /mnt/data/peeradon/h2snn-card15/H2S-NN
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain -- src scripts configs | wc -l)"
md5sum scripts/diagnostics/card15/a1_data_checks.py scripts/diagnostics/card15/l_flags.py
OUT=../docs/card15_evidence/L
python scripts/diagnostics/card15/a1_data_checks.py --out "$OUT" \
  --task acm_pp=configs/experiments/acm_pp_h2snn_v3b.yaml \
  --task dblp_ap=../docs/card11_inputs/h2snn_link_prediction_dblp_ap_tune_run0_daily-sweep-73/config.yaml
python scripts/diagnostics/card15/l_flags.py --in "$OUT" --out "$OUT"
echo "end=$(date -Is)"
