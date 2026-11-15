#!/bin/bash
# CS7-6: copy of scripts/slurm/run_tests.sbatch that activates h2snn-b (cluster-setup2 card), then the
# mb0 compares (Windows reference and node-A capture), pip check, and the freeze of h2snn-b.
#   sbatch docs/cluster_evidence2/cs7_6_run_tests_b.sbatch
#   or, as a step of an allocation: srun --jobid=<id> --overlap bash docs/cluster_evidence2/cs7_6_run_tests_b.sbatch
#SBATCH --job-name=h2snn-b-tests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/cluster_evidence2/cs7_6_full_suite.txt

set -uo pipefail

REPO=/mnt/data/peeradon/h2snn
cd ${REPO}/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b

echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD 2>/dev/null) branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) env=${CONDA_DEFAULT_ENV}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python -m unittest discover -s tests -v
status=$?
echo "suite_end=$(date -Is) suite_exit=${status}"

echo; echo "===== mb0_reference.py compare (CPU, 1 thread) vs the Windows reference"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference.pt 2>&1 | grep -vE "FutureWarning|autocast"
echo "mb0_compare_windows_exit=${PIPESTATUS[0]}"
echo; echo "===== mb0_reference.py compare (CPU, 1 thread) vs mb0_reference_nodeA.pt (g2n01)"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python ../docs/partMB_evidence/mb0_reference.py compare ../docs/cluster_evidence2/mb0_reference_nodeA.pt 2>&1 | grep -vE "FutureWarning|autocast"
echo "mb0_compare_nodeA_exit=${PIPESTATUS[0]}"

echo; echo "===== pip check"
python -m pip check; echo "pip_check_exit=$?"

FREEZE=requirements/linux-lock-b.freeze.txt
{
    echo "# pip freeze of h2snn-b after the CS7-6 full suite (cluster-setup2 card; author decision 7: transitive packages frozen)."
    echo "# date $(date -Is); node $(hostname); job ${SLURM_JOB_ID:-none} step ${SLURM_STEP_ID:-none}; git $(git rev-parse HEAD)"
    echo "# Record, not an install file: dgl comes from the conda package (scripts/setup/install_env_b.sh), and"
    echo "# 'packaging @ file:///...' is the conda-provided build of the env's base packages."
    python -m pip freeze
} > ${FREEZE}
echo "freeze_lines=$(grep -vc '^#' ${FREEZE}) -> ${FREEZE}"
echo "end=$(date -Is)"
exit "${status}"
