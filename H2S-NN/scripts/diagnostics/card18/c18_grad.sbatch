#!/bin/bash
# Card 18 follow-up: H2G-former grad census at the first backward (acm_pp LP, ACM NC, IEEE-CIS-G NC). One GPU.
# --mem: measured peaks 3.7 GB (acm_pp, 18295), 11.0 GB (oag-eng, 18353), 17.0 GB (pdns, 18304 cancelled in MetaPath2Vec) -> 17 GB + 30 % = 23G.
#SBATCH --job-name=c18-grad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=23G
#SBATCH --time=3:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card18/docs/card18_evidence/slurm-%j-grad.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card18/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) dirty=$(git status --porcelain -- src | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
EV=../docs/card18_evidence
SPECS=("lp acm_pp" "nc acm" "nc ieee-cis-g"); [ $# -gt 0 ] && SPECS=("$@")   # e.g. sbatch c18_grad.sbatch "nc ieee-cis-g"
for spec in "${SPECS[@]}"; do
  set -- $spec
  echo "== $1 $2"
  python scripts/diagnostics/card18/grad_census.py --task $1 --dataset $2 --out $EV/grad_census_$2.json 2>&1 | grep -v -i warn | tail -60
  echo "census $2 exit=${PIPESTATUS[0]}"
done
echo "end=$(date -Is)"
