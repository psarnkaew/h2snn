#!/bin/bash
# CS1: unit test of the SLURM script, then the mutation (--time removed -> the test must fail).
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn || exit 2
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m unittest -v tests.test_slurm_scripts 2>&1 | tail -6
echo "=== mutation: #SBATCH --time removed"
S=scripts/slurm/run_tests.sbatch
cp "$S" /tmp/run_tests.sbatch.orig
sed -i '/^#SBATCH --time/d' "$S"
python -m unittest tests.test_slurm_scripts.SbatchScriptTests.test_script_follows_cluster_rules 2>&1 | grep -E "AssertionError|missing|FAILED|OK" | head -5
cp /tmp/run_tests.sbatch.orig "$S"
echo "restored: $(grep -c '^#SBATCH --time' "$S") --time line"
