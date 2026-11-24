#!/bin/bash
#SBATCH --job-name=c21-c1
#SBATCH --partition=lab
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card21/H2S-NN/logs/slurm-%j.out
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card21/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID} node=$(hostname) git=$(git rev-parse HEAD) start=$(date -Is)"
C=/mnt/data/peeradon/card21_runs/hinormer_seq_cache_${SLURM_JOB_ID}
echo "=== process 1 (seed 0)"; python ../docs/card21_evidence/c1_hpokec_timing.py 0 $C
echo "=== process 2 (seed 0, same cache)"; python ../docs/card21_evidence/c1_hpokec_timing.py 0 $C
ls -la $C
echo "end=$(date -Is)"
