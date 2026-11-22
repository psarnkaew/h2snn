#!/bin/bash
# Card 15 step 5: heterophily records of acm_pp with the card 6 harness (scripts/heterophily_stats.py), as the acm_pa
# records were made (docs/heterophily_evidence/run_array.sbatch: one call per measure, CPU, 8 threads); then --collect.
#SBATCH --job-name=c15-het
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card15/docs/card15_evidence/slurm-%j-heterophily.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card15/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export TQDM_MININTERVAL=300
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain -- src scripts | wc -l)"
for m in task_h_edge mde; do
  python scripts/heterophily_stats.py --dataset acm_pp --task lp --measure "$m" --deadline-h 0.4 || echo "measure $m exit $?"
done
python scripts/heterophily_stats.py --collect
echo "end=$(date -Is)"
