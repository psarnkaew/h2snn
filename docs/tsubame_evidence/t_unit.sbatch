#!/bin/bash
# TSUBAME card: a unit's targeted tests / mutation on the lab cluster, CPU only (no --gres), from H2S-NN/.
#   cd H2S-NN && sbatch --output=../docs/tsubame_evidence/<name>.txt ../docs/tsubame_evidence/t_unit.sbatch '<command>'
#SBATCH --job-name=t4-unit
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
echo "head=$(git rev-parse HEAD) host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
echo "worktree: $(git status --short | tr '\n' ' ')"
echo "command: $1"
t0=${SECONDS}
bash -c "$1"
code=$?
echo "exit=${code} seconds=$((SECONDS - t0))"
exit ${code}
