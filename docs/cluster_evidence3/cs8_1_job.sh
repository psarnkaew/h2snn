#!/bin/bash
# CS8-1: does the pinned kernel environment E make the golden / mb0 captures portable? CPU only, h2snn-b.
#   capture : 1 thread -> docs/cluster_evidence3/v2_golden_E.pt, mb0_reference_E.pt (node A = g2n01)
#   check   : capture again into node-local scratch with 1 thread, then with the test job's default
#             threads (submit with --cpus-per-task=8), and compare each with the node-A files
#   sbatch --nodelist=<node> --cpus-per-task=8 --output=docs/cluster_evidence3/cs8_1_check_<node>.txt docs/cluster_evidence3/cs8_1_job.sh check
#SBATCH --job-name=cs8-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=0:40:00

set -uo pipefail
MODE="${1:?usage: cs8_1_job.sh capture|check}"
REPO=/mnt/data/peeradon/h2snn
EV=${REPO}/docs/cluster_evidence3
DIFF=${REPO}/docs/cluster_evidence2/cs7_tensor_diff.py
cd ${REPO}/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES=""
# Pinned kernel environment E (cluster-setup3 card), exported before Python starts.
export ATEN_CPU_CAPABILITY=avx2
export MKL_CBWR=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2

echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) mode=${MODE} cpus_per_task=${SLURM_CPUS_PER_TASK:-?} start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
echo "E: ATEN_CPU_CAPABILITY=${ATEN_CPU_CAPABILITY} MKL_CBWR=${MKL_CBWR} ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA}"

capture() {  # $1 = output directory, $2 = label; threads from the caller's environment
    python -c 'import torch; print("['"$2"'] torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
    python ../docs/heads_evidence/v2_golden.py recapture ../docs/heads_evidence/v2_golden.pt "$1/v2_golden_E.pt" 2>&1 | grep -vE "FutureWarning|autocast"
    python ../docs/partMB_evidence/mb0_reference.py save "$1/mb0_reference_E.pt" 2>&1 | grep -vE "FutureWarning|autocast"
    sha256sum "$1/v2_golden_E.pt" "$1/mb0_reference_E.pt"
}

if [ "${MODE}" = "capture" ]; then
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 capture "${EV}" "capture, 1 thread"
else
    for THREADS in 1 default; do
        SCR=${TMPDIR:-/tmp}/cs8_1_$$_${THREADS}; mkdir -p ${SCR}
        echo; echo "######## threads=${THREADS}"
        if [ "${THREADS}" = "1" ]; then
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 capture ${SCR} "threads=1"
        else
            (unset OMP_NUM_THREADS MKL_NUM_THREADS; capture ${SCR} "threads=default")
        fi
        echo "===== golden: node A capture vs this node, threads=${THREADS}"
        python ${DIFF} ${EV}/v2_golden_E.pt ${SCR}/v2_golden_E.pt | head -n 60; echo "golden_diff_exit=${PIPESTATUS[0]}"
        echo "===== mb0: node A capture vs this node, threads=${THREADS}"
        python ${DIFF} ${EV}/mb0_reference_E.pt ${SCR}/mb0_reference_E.pt | head -n 60; echo "mb0_diff_exit=${PIPESTATUS[0]}"
    done
fi
echo "end=$(date -Is)"
