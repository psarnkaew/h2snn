#!/bin/bash
# Card 5 T4 baselines-1 reference at v3b 1646efd (ops card T7.4), then the card 5 test file in the worktree (CPU).
# 1. a 1646efd tree (git archive of src, scripts, tests, configs) in the snapshot area; capture_1646efd.py from its
#    H2S-NN/ -> docs/card5_evidence/base_capture_1646efd.json; 2. capture_diff.py (894fd74 vs 1646efd);
# 3. python -m unittest -v tests.test_card5_protocol_hooks in the worktree itself.
# --mem: job 17887 (this file among others) peaked at 0.9 GB; + 30 % = 1.2 -> 2G.
#SBATCH --job-name=c7-cap
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=0:30:00
#SBATCH --exclude=g3an01
#SBATCH --output=/mnt/data/peeradon/h2snn-model/docs/card7_evidence/capture_1646efd_%j.txt
set -uo pipefail
. ~/miniforge3/etc/profile.d/conda.sh && conda activate h2snn-b || exit 2
export CUDA_VISIBLE_DEVICES="" TORCH_EXTENSIONS_DIR=~/.cache/torch_extensions/h2snn-b
export ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MPLBACKEND=Agg
W=/mnt/data/peeradon/h2snn-model; E=$W/docs/card7_evidence; S=/mnt/data/peeradon/h2snn-model-snap/cap1646efd
cd $W/H2S-NN || exit 2
echo "job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
echo "git=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_sha256=$(git diff HEAD | sha256sum | cut -c1-16)"
git status --short | grep -v '^??' | sed 's/^/working tree: /'
echo "== 1. capture at v3b 1646efd"
rm -rf $S && mkdir -p $S
git -C $W archive 1646efde6274255a712a0a0c3a425e0dd31e8ca0 H2S-NN/src H2S-NN/scripts H2S-NN/tests H2S-NN/configs | tar -x -C $S || exit 3
(cd $S/H2S-NN && python $E/capture_1646efd.py $W/docs/card5_evidence/base_capture_1646efd.json) 2>&1 | grep -vE "Warning|warn"
echo "== 2. 894fd74 vs 1646efd"
python $E/capture_diff.py
echo "== 3. tests.test_card5_protocol_hooks"
python -m unittest -v tests.test_card5_protocol_hooks 2>&1 | grep -vE "GBWarning|FutureWarning|warnings.warn|UserWarning"
echo "tests exit=${PIPESTATUS[0]} end=$(date -Is)"
rm -rf $S
