#!/bin/bash
# CS8-3: h2snn-b as the default: script and env tests, check_env.py with its new default lock, and the
# mutation (one E line removed from run_tests.sbatch -> the script test fails), restored in this job.
#   srun --jobid=<allocation> --overlap bash docs/cluster_evidence3/cs8_3_job.sh   (or sbatch)
#SBATCH --job-name=cs8-3
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
lscpu | grep 'Model name'
python -c 'import torch; print("torch", torch.__version__, "cpu_capability", torch.backends.cpu.get_cpu_capability(), "threads", torch.get_num_threads())'

echo "===== bash -n scripts/slurm/run_tests.sbatch"; bash -n scripts/slurm/run_tests.sbatch; echo "bash_n_exit=$?"
echo "===== tests"
python -m unittest -v tests.test_slurm_scripts tests.test_check_env 2>&1; echo "tests_exit=$?"
echo "===== python scripts/setup/check_env.py (default lock)"
python scripts/setup/check_env.py; echo "check_env_exit=$?"

echo "===== mutation: 'export MKL_CBWR=AVX2' removed from run_tests.sbatch"
F=scripts/slurm/run_tests.sbatch
cp ${F} "${TMPDIR:-/tmp}/run_tests.sbatch.orig.$$"
sed -i '/^export MKL_CBWR=AVX2$/d' ${F}
grep -c '^export MKL_CBWR=AVX2$' ${F}
python -m unittest tests.test_slurm_scripts.SbatchScriptTests.test_script_follows_cluster_rules 2>&1 | grep -E "AssertionError|^(OK|FAILED)"
cp "${TMPDIR:-/tmp}/run_tests.sbatch.orig.$$" ${F}
cmp ${F} "${TMPDIR:-/tmp}/run_tests.sbatch.orig.$$" && echo "restored: identical"
python -m unittest tests.test_slurm_scripts 2>&1 | tail -n 1
echo "end=$(date -Is)"
