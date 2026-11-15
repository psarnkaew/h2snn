#!/bin/bash
# CS8-2: the bit-exact checks against the cluster references, with or without the pinned kernel environment E.
#   sbatch --nodelist=<node> --output=<out> docs/cluster_evidence3/cs8_2_job.sh E|noE [mb0]
#   E    : export E, run the two golden tests (+ the CS7-4 compare test); with "mb0" also mb0_reference.py compare
#   noE  : E not exported (the node's own kernels): the same tests, which must fail with the pinned-kernel message
#SBATCH --job-name=cs8-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=0:30:00

set -uo pipefail
MODE="${1:?usage: cs8_2_job.sh E|noE [mb0]}"
MB0="${2:-}"
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES=""
unset ATEN_CPU_CAPABILITY MKL_CBWR ONEDNN_MAX_CPU_ISA
if [ "${MODE}" = "E" ]; then
    export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
fi
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) mode=${MODE} start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
echo "E: ATEN_CPU_CAPABILITY=${ATEN_CPU_CAPABILITY:-unset} MKL_CBWR=${MKL_CBWR:-unset} ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA:-unset}"
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
git status --short -- tests ../docs/partMB_evidence ../docs/heads_evidence | sed 's/^/working tree: /'

echo "===== golden tests"
python -m unittest -v \
    tests.test_task_heads.NodeClassificationGoldenTests \
    tests.test_cooperative.SymmetricIsThePreviousModelTests 2>&1 | grep -vE "FutureWarning|autocast"
echo "golden_tests_exit=${PIPESTATUS[0]}"

if [ "${MB0}" = "mb0" ]; then
    echo "===== tests.test_mb0_reference_compare"
    python -m unittest -v tests.test_mb0_reference_compare 2>&1 | grep -vE "FutureWarning|autocast"
    echo "mb0_test_exit=${PIPESTATUS[0]}"
    echo "===== mb0_reference.py compare mb0_reference_cluster.pt"
    python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "FutureWarning|autocast"
    echo "mb0_compare_exit=${PIPESTATUS[0]}"
fi
echo "end=$(date -Is)"
