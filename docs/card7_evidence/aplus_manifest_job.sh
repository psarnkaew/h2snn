#!/bin/bash
# Card 7 Phase A+: the perturbation manifest on the real graphs (amendments 1, 3, 4, 7) with placeholder `full`
# configurations (the experiment files; nothing is tuned yet) and no baseline sweep outputs (HetSheaf and CSNN
# "needs tuned config"), to count rows; nothing is run. CPU. --mem: Phase A's manifest job used 4G (17819).
#SBATCH --job-name=c7-manA+
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/aplus_manifest_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
E=/mnt/data/peeradon/h2snn-model/docs/card7_evidence
W=/mnt/data/peeradon/h2snn-model-snap/card7_manifests_aplus; rm -rf $W; mkdir -p $W
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_sha256=$(git diff HEAD | sha256sum | cut -c1-16)"
F='GBWarning|FutureWarning|warnings.warn|UserWarning|DeprecationWarning|import torch_geometric'
FULL="acm=configs/experiments/acm_h2snn_v3b.yaml dblp=configs/experiments/dblp_h2snn_v3b.yaml"
python scripts/perturbation/manifest.py --graphs data/NC --full_config $FULL --baseline_output $W/no_baselines --out_dir $W/pert 2>&1 | grep -vE "$F" | cut -c1-400
cp $W/pert/missing.json $E/aplus_manifest_missing.json
echo "== rows per dataset, graph kind, configuration"
awk -F'\t' 'NR>1 {print $2, $1, $7}' $W/pert/manifest.tsv | sort | uniq -c
echo "== rows per dataset, level, kind (h, H2)"
awk -F'\t' 'NR>1 {print $2, $3, $1, $4, $5}' $W/pert/manifest.tsv | sort | uniq -c
echo "== baseline configuration mapping"
python -c "import sys; sys.path.insert(0, 'scripts/perturbation'); import baseline_configs as bc; print(bc.markdown_mapping())"
echo "end=$(date -Is)"
