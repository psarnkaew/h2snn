#!/bin/bash
# LR-3 gate: v3-B golden and mb0 compares (differing none), the given unittest targets (default: the whole
# suite) and, with MODE=mutate, the two card mutations of the fail-closed registry (restored inside this job).
#   srun --jobid=<allocation> --overlap bash docs/legacy_evidence/lr3_job.sh [gate|suite|mutate] [targets...]
#SBATCH --job-name=lr3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

set -uo pipefail
MODE="${1:-gate}"; shift || true
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "mode=${MODE} job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads(), "cuda", torch.cuda.is_available())'
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
F='GBWarning|expandable|allocation pattern|PYTORCH_CUDA|gb_warning|An experimental|See https|and set the|FutureWarning|autocast|_flatten\("", torch.load'

if [ "${MODE}" = gate ] || [ "${MODE}" = suite ]; then
  echo "===== v3b_golden.py compare"
  CUDA_VISIBLE_DEVICES="" python ../docs/legacy_evidence/v3b_golden.py compare ../docs/legacy_evidence/v3b_golden_cluster.pt 2>&1 | grep -vE "${F}"
  echo "v3b_compare_exit=${PIPESTATUS[0]}"
  echo "===== mb0_reference.py compare"
  CUDA_VISIBLE_DEVICES="" python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "${F}"
  echo "mb0_compare_exit=${PIPESTATUS[0]}"
fi
if [ "${MODE}" = gate ]; then
  echo "===== unittest $*"
  python -m unittest "$@" 2>&1 | grep -vE "${F}" | grep -E "^(FAIL|ERROR):|Error|^Ran|^(OK|FAILED)|^  File \"/mnt/data/peeradon/h2snn/H2S-NN/(tests|src)" | head -400
  echo "tests_exit=${PIPESTATUS[0]}"
fi
if [ "${MODE}" = suite ]; then
  echo "===== unittest discover (whole suite)"
  python -m unittest discover -s tests -v > "${TMPDIR:-/tmp}/lr3_suite.$$" 2>&1; echo "suite_exit=$?"
  grep -vE "${F}" "${TMPDIR:-/tmp}/lr3_suite.$$" | grep -E "^(FAIL|ERROR):|^Ran|^(OK|FAILED)"
  grep -vE "${F}" "${TMPDIR:-/tmp}/lr3_suite.$$" | awk '/^={50,}/{p=1} p' | head -300
fi
if [ "${MODE}" = mutate ]; then
  S=src/configs/config.py
  cp ${S} "${TMPDIR:-/tmp}/config.orig.$$"
  echo "===== mutation 1: the raise for one value removed (sigmoid_gamma dropped from REMOVED_VALUES)"
  python - <<'EOF'
from pathlib import Path
p = Path("src/configs/config.py"); t = p.read_text()
old = '    ("model", "path_weight"): ("sigmoid_gamma",),\n'
assert t.count(old) == 1; p.write_text(t.replace(old, ''))
EOF
  python -m unittest tests.test_legacy_removal 2>&1 | grep -E "^(FAIL|ERROR):|^AssertionError|^Ran|^(OK|FAILED)"
  cp "${TMPDIR:-/tmp}/config.orig.$$" ${S}; cmp ${S} "${TMPDIR:-/tmp}/config.orig.$$" && echo "restored: identical"
  echo "===== mutation 2: the raise for one key removed (encoder dropped from REMOVED_KEYS)"
  python - <<'EOF'
from pathlib import Path
p = Path("src/configs/config.py"); t = p.read_text()
old = '    "model": ("K", "layer_dropout", "eta_init", "gamma_init", "coupling_gate_init", "encoder",\n'
assert t.count(old) == 1, old
p.write_text(t.replace(old, '    "model": ("K", "layer_dropout", "eta_init", "gamma_init", "coupling_gate_init",\n'))
EOF
  python -m unittest tests.test_legacy_removal 2>&1 | grep -E "^(FAIL|ERROR):|^AssertionError|^Ran|^(OK|FAILED)"
  cp "${TMPDIR:-/tmp}/config.orig.$$" ${S}; cmp ${S} "${TMPDIR:-/tmp}/config.orig.$$" && echo "restored: identical"
  python -m unittest tests.test_legacy_removal 2>&1 | tail -n 1
fi
echo "end=$(date -Is)"
