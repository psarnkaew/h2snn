#!/bin/bash
# LR-2 gate: v3-B golden and mb0 compares (differing none), tests.test_legacy_removal + test_v3b_golden, and the
# card mutation (ModelConfig default back to chebyshev -> the new default test fails; restored inside this job).
#   srun --jobid=<allocation> --overlap bash docs/legacy_evidence/lr2_job.sh   (or sbatch)
#SBATCH --job-name=lr2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
F='GBWarning|expandable|allocation pattern|PYTORCH_CUDA|gb_warning|An experimental|See https|and set the|FutureWarning|autocast|_flatten\("", torch.load'

echo "===== v3b_golden.py compare"
python ../docs/legacy_evidence/v3b_golden.py compare ../docs/legacy_evidence/v3b_golden_cluster.pt 2>&1 | grep -vE "${F}"
echo "v3b_compare_exit=${PIPESTATUS[0]}"
echo "===== mb0_reference.py compare"
python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 | grep -vE "${F}"
echo "mb0_compare_exit=${PIPESTATUS[0]}"
echo "===== tests.test_legacy_removal tests.test_v3b_golden"
python -m unittest -v tests.test_legacy_removal tests.test_v3b_golden 2>&1 | grep -vE "${F}"; echo "tests_exit=${PIPESTATUS[0]}"

S=src/configs/config.py
cp ${S} "${TMPDIR:-/tmp}/config.orig.$$"
echo "===== mutation: the ModelConfig default back to chebyshev"
python - <<'EOF'
from pathlib import Path
p = Path("src/configs/config.py"); t = p.read_text()
for old, new in (('    propagation: str = "anchored"', '    propagation: str = "chebyshev"'),
                 ('propagation=str(raw.get("propagation", "anchored"))',
                  'propagation=str(raw.get("propagation", "chebyshev"))')):
    assert t.count(old) == 1, old
    t = t.replace(old, new)
p.write_text(t)
EOF
python -m unittest tests.test_legacy_removal 2>&1 | grep -E "^AssertionError|Error:|^Ran|^(OK|FAILED)"
cp "${TMPDIR:-/tmp}/config.orig.$$" ${S}; cmp ${S} "${TMPDIR:-/tmp}/config.orig.$$" && echo "restored: identical"
python -m unittest tests.test_legacy_removal 2>&1 | tail -n 1
echo "end=$(date -Is)"
