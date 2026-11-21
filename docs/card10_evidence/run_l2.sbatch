#!/bin/bash
# Card 10 L2 + A2.2: test_fit_check + its two mutations, CPU only (no GPU, no real data).
#SBATCH --job-name=card10-l2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card10/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export WANDB_MODE=disabled
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
git status --short --untracked-files=no
python -m unittest -v tests.test_fit_check; a=$?
python ../docs/card10_evidence/mutation_l2.py; b=$?
python ../docs/card10_evidence/mutation_a22.py; c=$?
echo "tests=${a} mutation_l2=${b} mutation_a22=${c} end=$(date -Is)"
