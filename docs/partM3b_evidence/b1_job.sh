#!/bin/bash
# Part M3b B1: tests.test_vendor_patches, then the card mutation (the script reapplies an applied patch ->
# test_second_run_is_a_no_op fails), restored inside this job.
#   srun --jobid=<allocation> --overlap bash docs/partM3b_evidence/b1_job.sh      (or sbatch)
#SBATCH --job-name=m3b-b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:20:00
set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
echo "===== tests.test_vendor_patches"
python -m unittest -v tests.test_vendor_patches 2>&1 | tail -n 12
S=scripts/setup/apply_vendor_patches.sh
cp "$S" "${TMPDIR:-/tmp}/apply.orig.$$"
echo "===== mutation: the 'already applied' branch removed (the script reapplies an applied patch)"
python - <<'EOF'
from pathlib import Path
p = Path("scripts/setup/apply_vendor_patches.sh"); t = p.read_text()
old = 'if git -C "$dest" apply --reverse --check "$patch" 2>/dev/null; then'
assert t.count(old) == 1
p.write_text(t.replace(old, 'if false; then'))
EOF
python -m unittest tests.test_vendor_patches.ApplyVendorPatchesTests.test_second_run_is_a_no_op 2>&1 | grep -E "^(FAIL|ERROR):|^AssertionError|^Ran|^(OK|FAILED)"
cp "${TMPDIR:-/tmp}/apply.orig.$$" "$S"; cmp "$S" "${TMPDIR:-/tmp}/apply.orig.$$" && echo "restored: identical"
python -m unittest tests.test_vendor_patches 2>&1 | tail -n 1
echo "end=$(date -Is)"
