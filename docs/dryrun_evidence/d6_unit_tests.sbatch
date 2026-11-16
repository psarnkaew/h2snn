#!/bin/bash
# Card 2 D6: the given unittest targets in one short CPU job (no GPU). Run from /mnt/data/peeradon/h2snn-dryrun/H2S-NN.
#SBATCH --job-name=d6-tests
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/data/peeradon/dryrun_scratch/logs/%x_%j.out
set -u
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b
echo "head=$(git rev-parse HEAD) host=$(hostname) job=$SLURM_JOB_ID"
git status --short -- scripts tests configs | sed 's/^/worktree: /'
python -m unittest -v "$@" 2>&1 | grep -E ' \.\.\. |^(Ran|OK|FAILED|FAIL:|ERROR:)|Error|assert' | grep -viE 'acc|f1 |auc|mrr|loss='
