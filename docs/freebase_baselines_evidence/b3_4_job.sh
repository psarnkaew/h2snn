#!/bin/bash
# Card 3 B3.4: targeted tests as one CPU job (no --gres, 16G) + the B3.3 mutation. No full suite.
#   sbatch docs/freebase_baselines_evidence/b3_4_job.sh
#SBATCH --job-name=fb-b3-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/docs/freebase_baselines_evidence/b3_4_tests_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "unit=b3_4 job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --short -- . ../docs | grep -v '^??' | sed 's/^/working tree: /'
sha256sum src/baselines/adapters/h2sgnn.py src/baselines/adapters/pshgcn.py src/baselines/minibatch.py \
  src/evaluation/validation_subset.py tests/test_freebase_baselines.py ../docs/freebase_baselines_evidence/b3_mutation.py
python -m unittest tests.test_freebase_baselines tests.test_b1a_validation_subset tests.test_baselines \
  tests.test_baselines_minibatch tests.test_baselines_large_lp tests.test_partP_freebase tests.test_budget_table \
  tests.test_baseline_final_seeds 2>&1 | grep -vE "Warning|warn\(|^\s*$|An experimental" | tail -40
echo "tests_exit=${PIPESTATUS[0]}"
echo "===== mutation"
python ../docs/freebase_baselines_evidence/b3_mutation.py 2>&1 | grep -E "^(mutant|real|mutation check)"
echo "mutation_exit=${PIPESTATUS[0]} end=$(date -Is)"
