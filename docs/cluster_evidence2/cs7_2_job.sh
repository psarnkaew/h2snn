#!/bin/bash
# CS7-2: vendors present, manifest + ignore tests (with the ignore mutation), and the PSHGCN/ChebNetII tests of CS5, in h2snn-b on one GPU.
#   srun --jobid=<allocation> --overlap bash docs/cluster_evidence2/cs7_2_job.sh   (or sbatch --gres=gpu:1)
#SBATCH --job-name=cs7-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=1:00:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo; echo "===== available_vendors()"
python -c "
from src.baselines.runtime.vendor_paths import available_vendors
v = available_vendors(); print(v); print('all_true', all(v.values()), 'false:', [k for k, ok in v.items() if not ok])"

echo; echo "===== test_vendor_manifest"
python -m unittest -v tests.test_vendor_manifest 2>&1; echo "manifest_exit=$?"

echo; echo "===== mutation: one kind=clone row (PSHGCN) left out of the ignore list"
cp .gitignore "${TMPDIR:-/tmp}/gitignore.orig.$$"
sed -i '\#^/vendor/PSHGCN/$#d' .gitignore
grep -c '^/vendor/PSHGCN/$' .gitignore
python -m unittest tests.test_vendor_manifest.VendorIgnoreTests 2>&1 | grep -E "AssertionError|^(OK|FAILED)"
cp "${TMPDIR:-/tmp}/gitignore.orig.$$" .gitignore
git diff --quiet HEAD -- .gitignore; echo "gitignore_vs_HEAD_diff=$? (1 = the CS7-2 edit is present, as expected before the commit)"
cmp .gitignore "${TMPDIR:-/tmp}/gitignore.orig.$$" && echo "restored: identical"
python -m unittest tests.test_vendor_manifest.VendorIgnoreTests 2>&1 | tail -n 1

echo; echo "===== PSHGCN / ChebNetII tests that skipped or errored in CS5"
python -m unittest -v \
    tests.test_partP_derived_lp.FitTests.test_every_lp_adapter_runs_one_epoch \
    tests.test_partP_freebase.TinyReleaseTests.test_every_nc_adapter_runs_one_epoch \
    tests.test_baselines.VendorIsolationTests.test_all_declared_vendor_clones_are_present \
    tests.test_baselines.VendorIsolationTests.test_colliding_vendor_module_names_do_not_leak \
    tests.test_baselines.ReproducibilityTests.test_vendor_model_init_is_reproducible_under_seed_everything 2>&1
echo "vendor_tests_exit=$? end=$(date -Is)"
