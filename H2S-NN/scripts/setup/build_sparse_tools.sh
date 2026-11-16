#!/bin/bash
# Build sparse_tools' CPU extensions in place, inside its vendor clone (card 1d H1, author decision D1-b).
#
#   srun --jobid=<alloc> --overlap bash scripts/setup/build_sparse_tools.sh      (or sbatch)
#
# sparse_tools (vendor/VENDORS.tsv row; https://github.com/Yangxc13/sparse_tools, commit 0596f00) is never installed into
# h2snn-b: `setup.py build_ext --inplace` compiles `_spspmm_diag_cpu.so` and `_spspmm_diag_unsym_cpu.so` next to its
# sources with h2snn-b's torch (no pip, no build isolation). The adapter imports it from the clone with a stub `sortednp`
# (src/baselines/adapters/_pshgcn_mag.py). `pip freeze` is printed before and after and must be identical (exit 1 if not).
#SBATCH --job-name=build-sparse-tools
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
set -uo pipefail
# under sbatch the script runs from SLURM's spool copy: submit from H2S-NN/ (SLURM_SUBMIT_DIR); with srun/bash its path is used
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SRC="${REPO_ROOT}/vendor/sparse_tools"
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "repo git=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[ -d "${SRC}/.git" ] || { echo "no clone at ${SRC}: run scripts/setup/clone_vendors.sh --only sparse_tools" >&2; exit 2; }
echo "sparse_tools commit=$(git -C "${SRC}" rev-parse HEAD)"
python -c "import torch; print('torch', torch.__version__)"
before=$(mktemp); after=$(mktemp)
python -m pip freeze > "${before}" 2>/dev/null
( cd "${SRC}" && python setup.py build_ext --inplace ) > "${SRC}/build.log" 2>&1; status=$?
echo "build_ext exit=${status} (log: ${SRC}/build.log)"; tail -3 "${SRC}/build.log"
python -m pip freeze > "${after}" 2>/dev/null
if diff "${before}" "${after}" > /dev/null; then echo "pip freeze: identical before and after"
else echo "pip freeze CHANGED:"; diff "${before}" "${after}"; status=1; fi
echo "extensions:"; find "${SRC}/sparse_tools" -maxdepth 1 -name "*.so" -printf "  %p %s bytes\n"
rm -f "${before}" "${after}"
echo "end=$(date -Is) exit=${status}"
exit "${status}"
