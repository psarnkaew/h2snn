#!/bin/bash
# CS7-1c: DGL known-answer checks on one GPU type. Submit with the GPU type and output file, e.g.
#   sbatch --gres=gpu:2080ti:1 --output=docs/cluster_evidence2/cs7_1c_dgl_2080ti.txt docs/cluster_evidence2/cs7_1c_job.sh
#SBATCH --job-name=cs7-1c
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
python -W default ../docs/cluster_evidence2/cs7_dgl_check.py 2>&1
status=$?
echo "dgl_check_exit=${status}"
echo "end=$(date -Is)"
exit ${status}
