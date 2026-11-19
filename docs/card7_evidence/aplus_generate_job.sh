#!/bin/bash
# Card 7 Phase A+ (amendment 1): the new levels h = 0.30 and 0.50 on ACM and DBLP (0.20/0.40/0.60 stay as generated
# by job 17814), then --backfill: the edge-list fingerprint, C and the ceiling 1 - 1/C written into every ACM/DBLP
# graph.json (link.dat SHA-256 checked unchanged). CPU only; writes under the data root
# (H2S-NN/data/NC/<DATASET>_pert/). --mem: job 17814 peaked at 8.2 GB; + 30 % = 10.7 GB -> 11G.
#SBATCH --job-name=c7-pertA+
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
#SBATCH --time=4:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/aplus_generate_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
E=/mnt/data/peeradon/h2snn-model/docs/card7_evidence
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_sha256=$(git diff HEAD | sha256sum | cut -c1-16)"
git status --short | grep -v '^??' | sed 's/^/working tree: /'
F='GBWarning|FutureWarning|warnings.warn|UserWarning|DeprecationWarning|import torch_geometric|C = torch.sparse'
python scripts/perturbation/make_graphs.py --datasets acm dblp --levels 0.30 0.50 --out $E/aplus_reachability_new_levels.json 2>&1 | grep -vE "$F"
echo "generate_exit=${PIPESTATUS[0]} t=$(date -Is)"
python scripts/perturbation/make_graphs.py --datasets acm dblp --backfill --out $E/aplus_reachability.json 2>&1 | grep -vE "$F"
echo "backfill_exit=${PIPESTATUS[0]} end=$(date -Is)"
