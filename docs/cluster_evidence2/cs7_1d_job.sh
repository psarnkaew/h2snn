#!/bin/bash
# CS7-1d: torch_householder JIT compile and known answer in h2snn-b (CPU code; run on any node).
#   srun --jobid=<allocation> --overlap bash docs/cluster_evidence2/cs7_1d_job.sh   (or sbatch)
#SBATCH --job-name=cs7-1d
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
echo "g++: $(g++ --version | head -n1)"; echo "ninja: $(ninja --version)"
python ../docs/cluster_evidence2/cs7_householder_check.py 2>&1
status=$?
echo "householder_check_exit=${status} end=$(date -Is)"
exit ${status}
