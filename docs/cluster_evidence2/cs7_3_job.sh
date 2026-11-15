#!/bin/bash
# CS7-3: golden and mb0 captures in h2snn-b, CPU only, 1 thread.
#   MODE=capture : write docs/cluster_evidence2/v2_golden_nodeA.pt and mb0_reference_nodeA.pt (node A)
#   MODE=check   : capture again into node-local scratch and compare with the node-A files
# Usage: bash cs7_3_job.sh capture|check   (as an srun step of an allocation, or sbatch with the argument)
#SBATCH --job-name=cs7-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00

set -uo pipefail
MODE="${1:?usage: cs7_3_job.sh capture|check}"
REPO=/mnt/data/peeradon/h2snn
EV=${REPO}/docs/cluster_evidence2
cd ${REPO}/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) mode=${MODE} start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-) flags_avx512f=$(grep -m1 -c avx512f /proc/cpuinfo)"
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'cuda_available', torch.cuda.is_available()); print(torch.__config__.show())"

if [ "${MODE}" = "capture" ]; then
    OUT_G=${EV}/v2_golden_nodeA.pt; OUT_M=${EV}/mb0_reference_nodeA.pt
else
    SCR=${TMPDIR:-/tmp}/cs7_3_$$; mkdir -p ${SCR}
    OUT_G=${SCR}/v2_golden_$(hostname).pt; OUT_M=${SCR}/mb0_reference_$(hostname).pt
fi
python ../docs/heads_evidence/v2_golden.py recapture ../docs/heads_evidence/v2_golden.pt ${OUT_G}; echo "golden_capture_exit=$?"
python ../docs/partMB_evidence/mb0_reference.py save ${OUT_M}; echo "mb0_save_exit=$?"
sha256sum ${OUT_G} ${OUT_M}

if [ "${MODE}" = "check" ]; then
    echo; echo "===== golden: node A capture vs this node"
    python ${EV}/cs7_tensor_diff.py ${EV}/v2_golden_nodeA.pt ${OUT_G}; echo "golden_diff_exit=$?"
    echo; echo "===== mb0: node A capture vs this node"
    python ${EV}/cs7_tensor_diff.py ${EV}/mb0_reference_nodeA.pt ${OUT_M}; echo "mb0_diff_exit=$?"
fi
echo "end=$(date -Is)"
