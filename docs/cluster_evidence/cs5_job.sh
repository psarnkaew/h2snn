#!/bin/bash
#SBATCH --job-name=cs5-suite
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/cluster_evidence/cs5_full_suite.txt
# CS5: the full suite through scripts/slurm/run_tests.sbatch (run inside this job), then the
# Part MB reference compare on CPU, reported as information only.
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
bash scripts/slurm/run_tests.sbatch
suite=$?
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn || exit 2
echo "=== python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference.pt (CPU)"
CUDA_VISIBLE_DEVICES="" python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference.pt; echo "mb0_exit=$?"
echo "suite_exit=${suite}"
