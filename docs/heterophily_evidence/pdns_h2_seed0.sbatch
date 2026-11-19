#!/bin/bash
# Card 6 H2: pdns ℋ² with fill seed 0 only (author decision B, 2026-11-18). Five seeds need ~11 h 40 min
# (5 x ~2 h 12 min: one large meta-path ~2 h per fill) and do not fit the 12 h limit; 17388_18 reached the 11.5 h
# deadline after 22 of 25 path passes (record kept in logs/nc__pdns__h2__17388_18_timeout.json). Seed 0 is the
# primary value (the single fill H2GB itself uses); no spread is reported for pdns. CPU only (no GPU requested).
#SBATCH --job-name=s5-het-pdns0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-heterophily/docs/heterophily_evidence/logs/%x_%j.txt

set -uo pipefail
cd /mnt/data/peeradon/h2snn-heterophily/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export TQDM_MININTERVAL=300
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) mem=${SLURM_MEM_PER_NODE:-?}M cpus=${SLURM_CPUS_PER_TASK}"
python scripts/heterophily_stats.py --dataset pdns --task nc --measure h2 --fill-seeds 0 --deadline-h 5.5
status=$?
echo "end=$(date -Is) exit=${status}"
exit "${status}"
