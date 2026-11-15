#!/bin/bash
# Card baselines-h2gb: run one command in a SLURM job or allocation step, with the evidence header.
#   srun --jobid=16741 --overlap bash docs/baselines_h2gb_evidence/job.sh <label> <command...>
#   sbatch --mem=60G --time=2:00:00 docs/baselines_h2gb_evidence/job.sh <label> <command...>
# Runs from H2S-NN/ (or from ${BH_TREE}/H2S-NN, a `git archive` export, when BH_TREE is set) in env h2snn-b
# with the pinned CPU kernel environment E; prints commit, host, job id, and the peak RSS of the command.
#SBATCH --job-name=bh2gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/H2S-NN/logs/slurm-%j.out
set -uo pipefail
LABEL="$1"; shift
cd "${BH_TREE:-/mnt/data/peeradon/h2snn}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "label=${LABEL} job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
if [ -n "${BH_TREE:-}" ]; then echo "tree=$(pwd) (git archive export) commit=${BH_COMMIT:-?}"
else echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
     git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'; fi
echo "env=${CONDA_DEFAULT_ENV} cmd=$*"
python /mnt/data/peeradon/h2snn/docs/baselines_h2gb_evidence/peak.py "$@"
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
