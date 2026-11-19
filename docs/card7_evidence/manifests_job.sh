#!/bin/bash
# Card 7 Phase A: build the ablation and perturbation manifests with placeholder `full` configurations (the
# experiment files; nothing is tuned yet) to count rows and list the baselines without a tuned or published
# configuration; nothing is run. CPU. --mem: the card 7 targeted CPU jobs peaked at 0.36 GB (17803).
#SBATCH --job-name=c7-man
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/manifests_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
E=/mnt/data/peeradon/h2snn-model/docs/card7_evidence
W=/mnt/data/peeradon/h2snn-model-snap/card7_manifests; rm -rf $W; mkdir -p $W
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_sha256=$(git diff HEAD | sha256sum | cut -c1-16)"
F='GBWarning|FutureWarning|warnings.warn|UserWarning|DeprecationWarning|import torch_geometric'
for d in acm dblp imdb freebase; do
  mode=""; [ $d = freebase ] && mode="--freebase_mode minibatch"
  python scripts/ablation/seed_runs.py --dataset $d --full_config configs/experiments/${d}_h2snn_v3b.yaml \
    --tuning_base configs/experiments/${d}_h2snn_v3b.yaml --main_output_dir $W/main --out_dir $W/ablation $mode 2>&1 | grep -vE "$F"
done
cat $W/ablation/*/manifest.tsv | grep -v '^kind' | awk -F'\t' '{k=$1"/"$4; rows[k]++; runs[k]+=($1=="tune"?0:$7)} END {for (k in rows) print "ablation", k, "rows", rows[k], "runs", runs[k]}' | sort
FULL="acm=configs/experiments/acm_h2snn_v3b.yaml dblp=configs/experiments/dblp_h2snn_v3b.yaml imdb=configs/experiments/imdb_h2snn_v3b.yaml"
python scripts/perturbation/manifest.py --graphs data/NC --full_config $FULL --baseline_output $W/no_baselines --out_dir $W/pert_tuned_only 2>&1 | grep -vE "$F" | cut -c1-600
python scripts/perturbation/manifest.py --graphs data/NC --full_config $FULL --baseline_output $W/no_baselines --out_dir $W/pert_published --published_config 2>&1 | grep -vE "$F" | cut -c1-600
cp $W/pert_published/missing.json $E/v3_manifest_published_missing.json
cp $W/pert_tuned_only/missing.json $E/v3_manifest_tuned_only_missing.json
echo "== perturbation rows (published flag) per dataset, graph kind, method"
awk -F'\t' 'NR>1 {print $2, $1, $4}' $W/pert_published/manifest.tsv | sort | uniq -c
echo "end=$(date -Is)"
