#!/bin/bash
#SBATCH --job-name=cs4-counts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/cluster_evidence/cs4_count_tests.txt
# CS4: Part M M1 gate (m1_data_counts.py) and the loaders' count tests on the server's data/.
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn || exit 2
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date -Is)"
echo "=== python ../docs/partM_evidence/m1_data_counts.py"
python ../docs/partM_evidence/m1_data_counts.py 2>&1 | tail -40; echo "m1_exit=${PIPESTATUS[0]}"
echo "=== M1 test modules"
python -m unittest tests.test_data_layout tests.test_data tests.test_imdb_protocol tests.test_lp_derived tests.test_heart_negatives tests.test_lp_feature_rule tests.test_lp_feature_rebuild tests.test_lp_feature_rules tests.test_lp_valid_files tests.test_paths_and_facts tests.test_sheaf_energy tests.test_task_heads -v 2>&1 | grep -vE ' \.\.\. ok$'
echo "end=$(date -Is)"
