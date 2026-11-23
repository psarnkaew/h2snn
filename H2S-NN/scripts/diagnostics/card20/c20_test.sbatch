#!/bin/bash
# Card 20: the one CPU test on the fix and on the base 0b9873d (second checkout), plus the training-identity line.
#SBATCH --job-name=c20-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/data/peeradon/h2snn-card20/docs/card20_evidence/slurm-%j-test.log
set -uo pipefail
NEW=/mnt/data/peeradon/h2snn-card20/H2S-NN
OLD=/mnt/data/peeradon/h2snn-card20-base/H2S-NN
T=tests/test_card20_h2gformer_subset_metapath.py
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=
echo "job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Is)"
echo "fix  git=$(git -C $NEW rev-parse HEAD) dirty_src=$(git -C $NEW status --porcelain -- src | wc -l)"
echo "base git=$(git -C $OLD rev-parse HEAD) dirty_src=$(git -C $OLD status --porcelain -- src | wc -l)"
TMP=$(mktemp -d)
echo "== fix"
(cd $NEW && C20_LOSS_OUT=$TMP/fix python -m unittest -v tests.test_card20_h2gformer_subset_metapath 2>&1 | grep -v -i warn | tail -25)
echo "== base 0b9873d (test file copied into the base checkout's tests/, removed after)"
cp $NEW/$T $OLD/$T
(cd $OLD && C20_LOSS_OUT=$TMP/base python -m unittest -v tests.test_card20_h2gformer_subset_metapath 2>&1 | grep -v -i warn | tail -25)
rm -f $OLD/$T
echo "== training identity: first training-step loss"
echo "base=$(cat $TMP/base) fix=$(cat $TMP/fix) equal=$([ "$(cat $TMP/base)" = "$(cat $TMP/fix)" ] && echo yes || echo NO)"
echo "end=$(date -Is)"
