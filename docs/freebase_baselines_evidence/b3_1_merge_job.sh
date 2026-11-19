#!/bin/bash
# Card 3 B3.1: registry counts after the v3b merge (campaign_summary / iter_cells) and the LP-DGL notebook regenerated.
#SBATCH --job-name=fb-b3-1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-hetsheaf/docs/freebase_baselines_evidence/b3_1_merge.txt
cd /mnt/data/peeradon/h2snn-hetsheaf/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
echo "job=$SLURM_JOB_ID node=$(hostname) HEAD=$(git rev-parse HEAD) MERGE_HEAD=$(git rev-parse MERGE_HEAD 2>/dev/null) start=$(date -Is)"
sha256sum src/baselines/registry.py tests/test_baselines.py
python - <<'PY'
from src.baselines.registry import DATASETS, campaign_summary, iter_cells
print("runnable", campaign_summary()["runnable_cells"])
for task in DATASETS:
    print(task, sum(1 for c in iter_cells() if c["task"] == task and c["runnable"]))
PY
python scripts/generate_baseline_notebooks.py 2>&1 | tail -3
python scripts/generate_baseline_notebooks.py --check 2>&1 | tail -1
echo "end=$(date -Is)"
