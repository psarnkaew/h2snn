#!/bin/bash
# Card 7 V3: the perturbed graphs of ACM and DBLP (then IMDB, optional) and the reachability table. CPU only; writes
# under the data root (H2S-NN/data/NC/<DATASET>_pert/). --mem: no recorded peak for this script (16G chosen; sacct
# peak recorded in the report).
#SBATCH --job-name=c7-pert
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/v3_generate_%j.txt
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
python scripts/perturbation/make_graphs.py --datasets acm dblp --out $E/v3_reachability.json 2>&1 | grep -vE "$F"
echo "acm_dblp_exit=${PIPESTATUS[0]} t=$(date -Is)"
python scripts/perturbation/make_graphs.py --datasets --include_imdb --out $E/v3_reachability_imdb.json 2>&1 | grep -vE "$F"
echo "imdb_exit=${PIPESTATUS[0]} end=$(date -Is)"
