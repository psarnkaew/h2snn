#!/bin/bash
# LR-1: v3-B golden on the unchanged code. E exported; the script itself computes with 1 thread.
#   MODE=capture : write docs/legacy_evidence/v3b_golden_cluster.pt (g2n01)
#   MODE=check   : v3b_golden.py compare, tests.test_v3b_golden, mb0 compare (any node)
#   MODE=mutate  : the covering_only degree constant changed in src/models/sheaf_energy.py -> the golden test fails
#                  naming that arm's tensors; restored inside this job
#   srun --jobid=<allocation> --overlap bash docs/legacy_evidence/lr1_job.sh <mode>
#   sbatch --nodelist=g3an01 --output=... docs/legacy_evidence/lr1_job.sh check
#SBATCH --job-name=lr1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00

set -uo pipefail
MODE="${1:?usage: lr1_job.sh capture|check|mutate}"
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) mode=${MODE} start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
G=../docs/legacy_evidence/v3b_golden.py
REF=../docs/legacy_evidence/v3b_golden_cluster.pt
F='GBWarning|expandable|allocation pattern|PYTORCH_CUDA|gb_warning|An experimental|See https|and set the|FutureWarning|autocast|_flatten\("", torch.load'

if [ "${MODE}" = "capture" ]; then
    python ${G} capture ${REF} 2>&1 | grep -vE "${F}"; echo "capture_exit=${PIPESTATUS[0]}"
    sha256sum ${REF}
elif [ "${MODE}" = "check" ]; then
    echo "===== v3b_golden.py compare"
    python ${G} compare ${REF} 2>&1 | grep -vE "${F}"; echo "v3b_compare_exit=${PIPESTATUS[0]}"
    echo "===== tests.test_v3b_golden"
    python -m unittest -v tests.test_v3b_golden 2>&1 | grep -vE "${F}"; echo "v3b_test_exit=${PIPESTATUS[0]}"
    echo "===== mb0_reference.py compare mb0_reference_cluster.pt"
    python ../docs/partMB_evidence/mb0_reference.py compare ../docs/partMB_evidence/mb0_reference_cluster.pt 2>&1 \
        | grep -vE "${F}"; echo "mb0_compare_exit=${PIPESTATUS[0]}"
else
    S=src/models/sheaf_energy.py
    cp ${S} "${TMPDIR:-/tmp}/sheaf_energy.orig.$$"
    python - <<'EOF'
from pathlib import Path
p = Path("src/models/sheaf_energy.py"); t = p.read_text()
old = "(g.length + 1 if self.node_cell else 0)"
assert t.count(old) == 1
p.write_text(t.replace(old, "(g.length + 1 if self.node_cell else 1)"))
EOF
    echo "===== mutation: covering_only degree term (node_cell off: 0 -> 1)"
    python -m unittest tests.test_v3b_golden 2>&1 | grep -E "^AssertionError|differing tensors|^Ran|^(OK|FAILED)" | cut -c1-4000
    cp "${TMPDIR:-/tmp}/sheaf_energy.orig.$$" ${S}
    cmp ${S} "${TMPDIR:-/tmp}/sheaf_energy.orig.$$" && echo "restored: identical"
    git diff --quiet -- ${S} && echo "git diff ${S}: clean"
fi
echo "end=$(date -Is)"
