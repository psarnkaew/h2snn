#!/bin/bash
# Card 18 tests (a)-(c) and the before/after hop arrays on acm_pp. One GPU; job 18294 peak 3.8 GB + 30 %.
#SBATCH --job-name=c18-tests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=1:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-card18/docs/card18_evidence/slurm-%j-tests.log
set -uo pipefail
cd /mnt/data/peeradon/h2snn-card18/H2S-NN || exit 2
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b WANDB_MODE=disabled
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) dirty=$(git status --porcelain -- src scripts tests | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
EV=../docs/card18_evidence
echo "== hop arrays"; python scripts/diagnostics/card18/hop_arrays.py > $EV/hop_arrays_acm_pp.json; echo "exit=$?"
echo "== (a) (b)"; python -m unittest -v tests.test_card18_metapath_direction 2>&1 | grep -v -i warn
for m in magnn h2gformer; do
  echo "== (c) $m"; python scripts/diagnostics/card18/smoke_epoch.py --model $m --out $EV/smoke_${m}_acm_pp.json 2>&1 | grep -v -i warn | tail -12
  echo "smoke $m exit=${PIPESTATUS[0]}"
done
echo "== (c) h2gformer metapath_pe on the pre-card-18 code (card15 2e4881c worktree)"
OLD=/mnt/data/peeradon/h2snn-card15/H2S-NN
echo "old git=$(git -C $OLD rev-parse HEAD) dirty=$(git -C $OLD status --porcelain -- src scripts | wc -l)"
(cd $OLD && C18_PROJECT_ROOT=$OLD python /mnt/data/peeradon/h2snn-card18/H2S-NN/scripts/diagnostics/card18/smoke_epoch.py \
   --model h2gformer --out /mnt/data/peeradon/h2snn-card18/docs/card18_evidence/smoke_h2gformer_acm_pp_pre_card18.json 2>&1 | grep -v -i warn | tail -12)
echo "end=$(date -Is)"
