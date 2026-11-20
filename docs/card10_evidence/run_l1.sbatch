#!/bin/bash
# Card 10 L1: test_lab_slots + its mutation, CPU only.
#SBATCH --job-name=card10-l1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card10/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
git status --short --untracked-files=no
python -m unittest -v tests.test_lab_slots; a=$?
python ../docs/card10_evidence/mutation_l1.py; b=$?
echo "tests=${a} mutation=${b} end=$(date -Is)"
