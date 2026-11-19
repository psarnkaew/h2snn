#!/bin/bash
# Card 7 Phase B evidence on the real data (nothing is trained):
#  1. amendment 8: the original ACM / DBLP graphs as reference folders (make_graphs.py --original);
#  2. the perturbation manifest with the reference rows (placeholder `full` = the experiment files, no baseline sweep);
#  3. V8: the cell lists of the phases ablation (manifests of seed_runs.py on the placeholder files; Freebase full
#     batch as a placeholder, card 5 D1 open), perturbation and timing, with their line counts.
# CPU. --mem: Phase A+ generation peaked at 5.5 GB (17838) + 30 % -> 8G.
#SBATCH --job-name=c7-manB
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/phaseb_manifest_%j.txt
set -uo pipefail
cd /mnt/data/peeradon/h2snn-model/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
E=/mnt/data/peeradon/h2snn-model/docs/card7_evidence
W=/mnt/data/peeradon/h2snn-model-snap/card7_phaseb; rm -rf $W; mkdir -p $W
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --short -- src scripts tests | wc -l)"
F='GBWarning|FutureWarning|warnings.warn|UserWarning|DeprecationWarning|import torch_geometric|Processing|Done!'
echo "== 1. original graphs (amendment 8)"
python scripts/perturbation/make_graphs.py --datasets acm dblp --original --out $E/phaseb_original.json 2>&1 | grep -vE "$F"
echo "== 2. perturbation manifest"
FULL="acm=configs/experiments/acm_h2snn_v3b.yaml dblp=configs/experiments/dblp_h2snn_v3b.yaml"
python scripts/perturbation/manifest.py --graphs data/NC --full_config $FULL --baseline_output $W/no_baselines --out_dir $W/pert 2>&1 | grep -vE "$F" | cut -c1-300
cp $W/pert/missing.json $E/phaseb_manifest_missing.json
echo "rows per graph kind and configuration:"; awk -F'\t' 'NR>1 {print $2, $1, $7}' $W/pert/manifest.tsv | sort | uniq -c
echo "== 3. cell lists"
for d in acm dblp imdb freebase; do
  mode=""; [ "$d" = freebase ] && mode="--freebase_mode full_batch"
  python scripts/ablation/seed_runs.py --dataset $d --full_config configs/experiments/${d}_h2snn_v3b.yaml \
      --tuning_base configs/experiments/${d}_h2snn_v3b.yaml --main_output_dir $W/main --out_dir $W/abl $mode 2>&1 | grep -vE "$F"
done
python scripts/tsubame/make_cell_list.py --phase ablation --manifest $W/abl/{acm,dblp,imdb,freebase}/manifest.tsv --out $W/cells/cells_ablation_tuning.txt 2>&1 | grep -vE "$F"
python scripts/tsubame/make_cell_list.py --phase perturbation --manifest $W/pert/manifest.tsv --out $W/cells/cells_perturbation.txt 2>&1 | grep -vE "$F"
python scripts/tsubame/make_cell_list.py --phase timing --out $W/cells/cells_timing.txt 2>&1 | grep -vE "$F"
for f in $W/cells/*.txt; do echo "$(basename $f): $(grep -cvE '^\s*(#|$)' $f) runnable, $(grep -c '^#' $f) comment"; done
echo "manifest rows: ablation tune $(cat $W/abl/*/manifest.tsv | awk -F'\t' '$1=="tune"' | wc -l), runs $(cat $W/abl/*/manifest.tsv | awk -F'\t' '$1=="runs"' | wc -l); perturbation with command $(awk -F'\t' 'NR>1 && $14!=""' $W/pert/manifest.tsv | wc -l), without $(awk -F'\t' 'NR>1 && $14==""' $W/pert/manifest.tsv | wc -l)"
echo "== first line of each list"
for f in $W/cells/*.txt; do echo "-- $(basename $f)"; grep -vE '^\s*(#|$)' $f | head -n 1 | sed "s#$W#<W>#g"; done
echo "end=$(date -Is)"
