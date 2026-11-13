#!/bin/bash
#SBATCH --job-name=cs3-vendors
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn/docs/cluster_evidence/cs3_vendor_tests.txt
# CS3: manifest tests, available_vendors(), then every test that skipped for a missing clone.
set -u
cd /mnt/data/peeradon/h2snn/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn || exit 2
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date -Is)"
echo "=== available_vendors()"
python -c "from src.baselines.runtime import available_vendors as a; v=a(); [print(f'{k:12s} {p}') for k,p in v.items()]; print('all present:', all(v.values()))"
echo "=== tests.test_vendor_manifest"
python -m unittest -v tests.test_vendor_manifest 2>&1
for t in tests.test_baselines tests.test_heterophily_metrics tests.test_hgb_oracle tests.test_partP_freebase tests.test_partP_derived_lp tests.test_partP_h2gformer tests.test_partPfix_h2gformer_width tests.test_partPfix_data_root tests.test_partO_literature_row; do
  echo "=== $t"; python -m unittest -v "$t" 2>&1
done
echo "end=$(date -Is)"
