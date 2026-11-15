#!/bin/bash
# CS8-0: CPU model and flags (avx2 / avx512f) of one node, and torch's CPU capability without and with E.
#   sbatch --nodelist=<node> --output=docs/cluster_evidence3/cs8_0_cpu_<node>.txt docs/cluster_evidence3/cs8_0_cpuinfo_job.sh
#   or as a step of an allocation: srun --jobid=<id> --overlap bash docs/cluster_evidence3/cs8_0_cpuinfo_job.sh
#SBATCH --job-name=cs8-0-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:10:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
echo "flag avx2=$(grep -m1 -o -w avx2 /proc/cpuinfo || echo no) avx512f=$(grep -m1 -o -w avx512f /proc/cpuinfo || echo no)"
PY='import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
echo -n "without E: "; python -c "${PY}"
echo -n "with E:    "; ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 python -c "${PY}"
echo "end=$(date -Is)"
