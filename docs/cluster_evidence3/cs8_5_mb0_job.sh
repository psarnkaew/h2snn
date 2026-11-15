#!/bin/bash
# CS8-5: mb0_reference.py compare against the cluster reference, with E (as run_tests.sbatch exports it), CPU only.
#   sbatch --nodelist=<node> --output=docs/cluster_evidence3/cs8_5_mb0_<node>.txt docs/cluster_evidence3/cs8_5_mb0_job.sh
#   or as a step of an allocation: srun --jobid=<id> --overlap bash docs/cluster_evidence3/cs8_5_mb0_job.sh
#SBATCH --job-name=cs8-5-mb0
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=0:20:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES=""
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "FutureWarning|autocast"
echo "mb0_compare_exit=${PIPESTATUS[0]} end=$(date -Is)"
