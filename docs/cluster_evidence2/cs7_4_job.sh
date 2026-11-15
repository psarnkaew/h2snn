#!/bin/bash
# CS7-4: mb0_reference.py compare exit code; test and mutation (the exit-code line removed), CPU only.
#   srun --jobid=<allocation> --overlap bash docs/cluster_evidence2/cs7_4_job.sh   (or sbatch)
#SBATCH --job-name=cs7-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00

set -uo pipefail
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES=""
echo "job=${SLURM_JOB_ID:-none} step=${SLURM_STEP_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) env=${CONDA_DEFAULT_ENV}"
S=../docs/partMB_evidence/mb0_reference.py

echo "===== test"
python -m unittest -v tests.test_mb0_reference_compare 2>&1 | grep -vE "FutureWarning|autocast"; echo "test_exit=${PIPESTATUS[0]}"

echo; echo "===== mutation: the exit-code line removed"
cp ${S} "${TMPDIR:-/tmp}/mb0_reference.py.orig.$$"
sed -i '/sys.exit(1 if bad else 0)/d' ${S}
grep -c "sys.exit(1 if bad else 0)" ${S}
python -m unittest tests.test_mb0_reference_compare 2>&1 | grep -E "AssertionError|^(OK|FAILED)"
cp "${TMPDIR:-/tmp}/mb0_reference.py.orig.$$" ${S}
cmp ${S} "${TMPDIR:-/tmp}/mb0_reference.py.orig.$$" && echo "restored: identical"
python -m unittest tests.test_mb0_reference_compare 2>&1 | tail -n 1
echo "end=$(date -Is)"
