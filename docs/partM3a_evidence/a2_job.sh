#!/bin/bash
# M3a-A2: CSR scope builder: equality tests, every Part MB / MB-fix test unchanged, the card mutation
# (index without self-loop facts; applied and restored inside this job), the timing (informational),
# then the gate. With E exported.
#   srun --jobid=<allocation> --overlap bash docs/partM3a_evidence/a2_job.sh   (or sbatch)
#SBATCH --job-name=m3a-a2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
git status --short -- . | grep -v '^??' | sed 's/^/working tree: /'
TMP=${TMPDIR:-/tmp}/m3a_a2_$$; mkdir -p ${TMP}

echo "===== A2 tests and every Part MB / MB-fix test"
MB_TESTS=$(ls tests/test_partMB_*.py tests/test_partMBfix_*.py | sed 's#/#.#; s#\.py$##' | tr '\n' ' ')
echo "modules: tests.test_partM3a_scopes ${MB_TESTS}"
python -m unittest -v tests.test_partM3a_scopes ${MB_TESTS} 2>&1 | grep -vE "FutureWarning|autocast"
echo "tests_exit=${PIPESTATUS[0]}"

F=src/data/scopes.py
cp ${F} ${TMP}/scopes.orig
echo; echo "===== mutation: the index built without self-loop facts -> test_equal_to_the_reference_builder"
python - <<'EOF'
from pathlib import Path
p = Path("src/data/scopes.py"); t = p.read_text()
old = "    nodes = torch.cat([src, dst[~loop]])\n    facts = torch.cat([fact, fact[~loop]])\n"
new = "    nodes = torch.cat([src[~loop], dst[~loop]])\n    facts = torch.cat([fact[~loop], fact[~loop]])\n"
assert t.count(old) == 1
p.write_text(t.replace(old, new))
EOF
python -m unittest tests.test_partM3a_scopes.ReferenceEqualityTests.test_equal_to_the_reference_builder 2>&1 \
    | grep -E "^AssertionError|Error:|^Ran|^(OK|FAILED)"
cp ${TMP}/scopes.orig ${F}; cmp ${F} ${TMP}/scopes.orig && echo "restored: identical"
python -m unittest tests.test_partM3a_scopes 2>&1 | tail -n 1

echo; echo "===== timing (informational, not a gate): python ../docs/partM3a_evidence/a2_timing.py"
python ../docs/partM3a_evidence/a2_timing.py

echo
bash ../docs/partM3a_evidence/gate.sh
echo "end=$(date -Is)"
