#!/bin/bash
# Part M3b gate after a unit that touches src/: both bit-exact compares (E, 1 thread set by the scripts) and the
# given unittest targets.
#   srun --jobid=<allocation> --overlap bash docs/partM3b_evidence/m3b_gate_job.sh <unit> [targets...]
# With M3B_TREE=<dir> M3B_COMMIT=<sha> it runs in a `git archive <sha>` export of the repository (a unit's commit
# alone, without later units' uncommitted edits; no git metadata there, so the commit is printed from M3B_COMMIT).
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
set -uo pipefail
UNIT="$1"; shift
cd "${M3B_TREE:-/mnt/data/peeradon/h2snn}/H2S-NN" || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "unit=${UNIT} job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
if [ -n "${M3B_TREE:-}" ]; then echo "tree=$(pwd) (git archive export) commit=${M3B_COMMIT:-?} env=${CONDA_DEFAULT_ENV}"
else echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"; fi
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads(), "cuda", torch.cuda.is_available())'
[ -z "${M3B_TREE:-}" ] && git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
F='GBWarning|expandable|allocation pattern|PYTORCH_CUDA|gb_warning|An experimental|See https|and set the|FutureWarning|autocast|_flatten\("", torch.load|if you want to disable'
echo "===== v3b_golden.py compare"
CUDA_VISIBLE_DEVICES="" python ../docs/legacy_evidence/v3b_golden.py compare ../docs/legacy_evidence/v3b_golden_cluster.pt 2>&1 | grep -vE "${F}"
echo "v3b_compare_exit=${PIPESTATUS[0]}"
echo "===== mb0_reference.py compare"
CUDA_VISIBLE_DEVICES="" python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "${F}"
echo "mb0_compare_exit=${PIPESTATUS[0]}"
if [ "$#" -gt 0 ]; then
  echo "===== unittest $*"
  python -m unittest "$@" > "${TMPDIR:-/tmp}/gate.$$" 2>&1; echo "tests_exit=$?"
  grep -vE "${F}" "${TMPDIR:-/tmp}/gate.$$" | grep -E "^(FAIL|ERROR):|^Ran|^(OK|FAILED)"
  grep -vE "${F}" "${TMPDIR:-/tmp}/gate.$$" | awk '/^={50,}/{p=1} p' | head -120
fi
echo "end=$(date -Is)"
