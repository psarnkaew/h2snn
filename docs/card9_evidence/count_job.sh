#!/bin/bash
# Card 9 W9: the number of test cases unittest discovers (loader only, nothing runs) at card 7's suite head 4d00154
# (reference 879, suite 17858), at the card base fc0ebaa and at card9 HEAD; each tree from `git archive` into a
# scratch folder (vendor and data linked from the worktree).
#SBATCH --job-name=c9-count
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card9_evidence/count_%j.txt
set -uo pipefail
WT=/mnt/data/peeradon/h2snn-model
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b MPLBACKEND=Agg
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is) git=$(git -C $WT rev-parse HEAD)"
TMPD=$(mktemp -d /mnt/data/peeradon/h2snn-model-snap/c9count.XXXX)
for rev in 4d00154e7e1f3c01174c256c4b2cffa24d961ed6 fc0ebaa605280f2d987c4848cc3109c7d3b66680 $(git -C $WT rev-parse HEAD); do
    d=$TMPD/$rev; mkdir -p $d
    git -C $WT archive $rev H2S-NN | tar -x -C $d
    ln -sfn $WT/H2S-NN/vendor $d/H2S-NN/vendor; ln -sfn $WT/H2S-NN/data $d/H2S-NN/data
    ( cd $d/H2S-NN && python -c "
import unittest
suite = unittest.defaultTestLoader.discover('tests')
def walk(s):
    for t in s:
        yield from (walk(t) if isinstance(t, unittest.TestSuite) else [t])
cases = list(walk(suite))
bad = [c for c in cases if type(c).__name__ == '_FailedTest']
print('$rev', 'cases', len(cases), 'import_failures', len(bad))
" 2>&1 | tail -1 )
done
rm -rf $TMPD
echo "end=$(date -Is)"
